import os
import glob
import cv2
import json
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import albumentations.pytorch as Ap
import numpy as np
import sys
import torch.amp

from configs.config import get_default_config, Config
from src.models.end2end_lpr import End2EndLPR
from src.training.sr_loss import SRLoss
from src.training.trainer import decode_with_confidence
from src.data.transforms import get_degradation_transforms

class JointDataset(Dataset):
    """
    Dataset for End-to-End Joint Training.
    Returns: (lr_tensor, hr_tensor, targets, target_len, label, track_id)
    - lr_tensor: [5, 3, 16, 64]
    - hr_tensor: [3, 32, 128] (if available, else zero tensor)
    """
    def __init__(self, data_root, split_file, char2idx, mode='train', lr_h=16, lr_w=64, scale=2, seed=42):
        self.data_root = data_root
        self.mode = mode
        self.lr_h = lr_h
        self.lr_w = lr_w
        self.hr_h = lr_h * scale
        self.hr_w = lr_w * scale
        self.char2idx = char2idx
        self.seed = seed
        
        self.samples = []
        
        # Load valid tracks from split file
        with open(split_file, 'r') as f:
            val_ids = set(json.load(f))
            
        from src.data.dataset import MultiFrameDataset
        
        abs_root = os.path.abspath(data_root)
        all_tracks = sorted(glob.glob(os.path.join(abs_root, "**", "track_*"), recursive=True))
                    
        tracks = []
        for t in all_tracks:
            unique_id = MultiFrameDataset.get_track_unique_id(t)
            base_name = os.path.basename(t)
            is_val = unique_id in val_ids or (base_name in val_ids and "Scenario-B" in t)
            
            if mode == 'train' and not is_val:
                tracks.append(t)
            elif mode == 'val' and is_val:
                tracks.append(t)
                
        for t in tracks:
            json_path = os.path.join(t, "annotations.json")
            if not os.path.exists(json_path): continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list): data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label: continue
                
                track_id = os.path.basename(t)
                lr_files = sorted(glob.glob(os.path.join(t, "lr-*.png")) + glob.glob(os.path.join(t, "lr-*.jpg")))
                hr_files = sorted(glob.glob(os.path.join(t, "hr-*.png")) + glob.glob(os.path.join(t, "hr-*.jpg")))
                
                # Real LR samples
                if lr_files:
                    self.samples.append({
                        'paths': lr_files,
                        'hr_path': None,
                        'label': label,
                        'is_synthetic': False,
                        'track_id': track_id
                    })
                
                # Synthetic samples (only train)
                if hr_files and mode == 'train':
                    self.samples.append({
                        'paths': hr_files,
                        'hr_path': hr_files[len(hr_files)//2],
                        'label': label,
                        'is_synthetic': True,
                        'track_id': track_id
                    })
            except:
                pass

        # Transforms
        additional_targets = {f'image{i}': 'image' for i in range(1, 5)}
        self.degrade = get_degradation_transforms(4) if mode == 'train' else None
        
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(self.lr_h, self.lr_w),
                A.ColorJitter(p=0.5),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                Ap.ToTensorV2()
            ], additional_targets=additional_targets, is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(self.lr_h, self.lr_w),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                Ap.ToTensorV2()
            ], additional_targets=additional_targets, is_check_shapes=False)

        self.hr_transform = A.Compose([
            A.Resize(self.hr_h, self.hr_w),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            Ap.ToTensorV2()
        ], is_check_shapes=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        is_synthetic = item['is_synthetic']
        
        raw_images = []
        for p in item['paths']:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.hr_w, self.hr_h))
                raw_images.append(img)
        while len(raw_images) < 5:
            if len(raw_images) > 0: raw_images.append(raw_images[-1])
            else: raw_images.append(np.zeros((self.hr_h, self.hr_w, 3), dtype=np.uint8))
        raw_images = raw_images[:5]

        hr_img = np.zeros((self.hr_h, self.hr_w, 3), dtype=np.uint8)
        if item['hr_path']:
            img = cv2.imread(item['hr_path'], cv2.IMREAD_COLOR)
            if img is not None:
                hr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hr_img = cv2.resize(hr_img, (self.hr_w, self.hr_h))

        if is_synthetic and self.degrade:
            deg_targets = {f'image{i}': raw_images[i] for i in range(1, 5)}
            res = self.degrade(image=raw_images[0], **deg_targets)
            raw_images = [res['image']] + [res[f'image{i}'] for i in range(1, 5)]

        targets = {f'image{i}': raw_images[i] for i in range(1, 5)}
        res = self.transform(image=raw_images[0], **targets)
        lr_tensors = [res['image']] + [res[f'image{i}'] for i in range(1, 5)]
        lr_tensor = torch.stack(lr_tensors, dim=0)

        hr_tensor = self.hr_transform(image=hr_img)['image']
        
        label = item['label']
        target = [self.char2idx[c] for c in label if c in self.char2idx]
        if len(target) == 0: target = [0]
        
        return lr_tensor, hr_tensor, torch.tensor(target, dtype=torch.long), len(target), label, item['track_id'], is_synthetic

    @staticmethod
    def collate_fn(batch):
        lrs, hrs, targets, target_lengths, labels_text, track_ids, is_synth = zip(*batch)
        return (
            torch.stack(lrs, 0),
            torch.stack(hrs, 0),
            torch.cat(targets),
            torch.tensor(target_lengths, dtype=torch.long),
            labels_text,
            track_ids,
            torch.tensor(is_synth, dtype=torch.bool)
        )


def main():
    print("=== Task 2.1.5: Starting Two-Stage Decoupled Training ===")
    config = get_default_config()
    
    FROZEN_EPOCHS = 20
    JOINT_EPOCHS = 15
    TOTAL_EPOCHS = FROZEN_EPOCHS + JOINT_EPOCHS
    
    LR_OCR_STAGE1 = 3e-4
    LR_SR_STAGE2 = 1e-5
    LR_OCR_STAGE2 = 1e-4
    LAMBDA_SR = 0.1
    BATCH_SIZE = 16
    
    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Starting Decoupled Training with TOTAL_EPOCHS={TOTAL_EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
    
    if not os.path.exists(config.VAL_SPLIT_FILE):
        os.makedirs(os.path.dirname(config.VAL_SPLIT_FILE), exist_ok=True)
        from src.data.dataset import MultiFrameDataset
        _ = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode="train",
            val_split_file=config.VAL_SPLIT_FILE,
            split_ratio=config.SPLIT_RATIO,
            seed=config.SEED
        )
            
    train_dataset = JointDataset(config.DATA_ROOT, config.VAL_SPLIT_FILE, config.CHAR2IDX, mode="train",
                                 lr_h=16, lr_w=64, scale=2)
    val_dataset = JointDataset(config.DATA_ROOT, config.VAL_SPLIT_FILE, config.CHAR2IDX, mode="val",
                               lr_h=16, lr_w=64, scale=2)
    print(f"Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=JointDataset.collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=JointDataset.collate_fn, num_workers=2)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="restran", choices=["restran", "sr_vit"])
    args, _ = parser.parse_known_args()

    if args.model == "sr_vit":
        from src.models.sr_vit import MultiFrameSRViT
        print("🚀 Using MultiFrameSRViT (Flow 3) architecture")
        model = MultiFrameSRViT(
            num_classes=config.NUM_CLASSES,
            sr_num_blocks=8,
            patch_h=config.SR_VIT_PATCH_H,
            patch_w=config.SR_VIT_PATCH_W,
            embed_dim=config.SR_VIT_EMBED_DIM,
            num_heads=config.SR_VIT_NUM_HEADS,
            num_layers=config.SR_VIT_NUM_LAYERS,
            ff_dim=config.SR_VIT_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            max_seq_len=(config.IMG_HEIGHT // config.SR_VIT_PATCH_H) * (config.IMG_WIDTH // config.SR_VIT_PATCH_W)
        ).to(device)
    else:
        print("🚀 Using End2EndLPR (Flow 2) architecture")
        model = End2EndLPR(
            num_classes=config.NUM_CLASSES,
            sr_num_blocks=8,
            ocr_transformer_layers=2,
            pretrained=True
        ).to(device)
    
    # Warm-start SR module if available
    sr_weights_path = os.path.join(config.OUTPUT_DIR, "sr_net_best.pth")
    if os.path.exists(sr_weights_path):
        print(f"📦 Warm-starting SR module from {sr_weights_path}")
        # Use strict=False because the new Frame-Parallel SR has 3 input channels instead of 15
        model.sr_net.load_state_dict(torch.load(sr_weights_path, map_location=device), strict=False)
    
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    sr_loss_fn = SRLoss(w_mse=1.0, w_perceptual=0.1, w_edge=0.1, device=device)
    scaler = torch.amp.GradScaler(device.type) if device.type in ["cuda", "mps"] else None
    
    best_val_acc = 0.0
    optimizer = None
    scheduler = None
    
    for epoch in range(TOTAL_EPOCHS):
        if epoch == 0:
            print("\n" + "="*50)
            print("🚀 STAGE 1: OCR FINE-TUNING (FROZEN SR)")
            print("="*50)
            ocr_params = [p for n, p in model.named_parameters() if "sr_net" not in n]
            for param in model.sr_net.parameters():
                param.requires_grad = False
            for param in ocr_params:
                param.requires_grad = True
                
            optimizer = optim.AdamW(ocr_params, lr=LR_OCR_STAGE1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FROZEN_EPOCHS, eta_min=1e-6)
            
        elif epoch == FROZEN_EPOCHS:
            print("\n" + "="*50)
            print("🚀 STAGE 2: END-TO-END JOINT FINE-TUNING (UNFROZEN SR)")
            print("="*50)
            for param in model.sr_net.parameters():
                param.requires_grad = True
                
            ocr_params = [p for n, p in model.named_parameters() if "sr_net" not in n]
            optimizer = optim.AdamW([
                {"params": model.sr_net.parameters(), "lr": LR_SR_STAGE2, "weight_decay": 1e-4},
                {"params": ocr_params, "lr": LR_OCR_STAGE2, "weight_decay": 1e-4}
            ])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=JOINT_EPOCHS, eta_min=1e-6)
            
        model.train()
        # Stage 1: SR is frozen, so eval mode for SR is safer
        if epoch < FROZEN_EPOCHS:
            model.sr_net.eval()
            
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
        
        for lr, hr, targets, target_lengths, labels_text, track_ids, is_synth in pbar:
            lr = lr.to(device)
            hr = hr.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type) if scaler else torch.autocast("cpu", enabled=False):
                logits, sr_image = model(lr, return_sr=True)
                
                preds_permuted = logits.permute(1, 0, 2)
                input_lengths = torch.full((lr.size(0),), logits.size(1), dtype=torch.long)
                
                if device.type == "mps":
                    loss_ctc = ctc_loss_fn(preds_permuted.float().cpu(), targets.cpu(), input_lengths, target_lengths).to(device)
                else:
                    loss_ctc = ctc_loss_fn(preds_permuted.float(), targets, input_lengths, target_lengths)
                
                if epoch < FROZEN_EPOCHS:
                    loss = loss_ctc
                else:
                    if is_synth.any():
                        sr_image_synth = sr_image[is_synth]
                        hr_synth = hr[is_synth]
                        center_idx = sr_image_synth.size(1) // 2
                        loss_sr = sr_loss_fn(sr_image_synth[:, center_idx, :, :, :], hr_synth)
                    else:
                        loss_sr = torch.tensor(0.0, device=device)
                    loss = loss_ctc + LAMBDA_SR * loss_sr
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for lr_v, _, _, _, labels_v, _, _ in val_loader:
                lr_v = lr_v.to(device)
                with torch.amp.autocast(device_type=device.type) if scaler else torch.autocast("cpu", enabled=False):
                    logits_v = model(lr_v, return_sr=False)
                decoded_list = decode_with_confidence(logits_v, config.IDX2CHAR)
                for i, (pred_t, _) in enumerate(decoded_list):
                    if pred_t == labels_v[i]:
                        total_correct += 1
                    total_val_samples += 1
                    
        val_acc = (total_correct / total_val_samples) * 100 if total_val_samples > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        stage_str = "STAGE 1" if epoch < FROZEN_EPOCHS else "STAGE 2"
        print(f"📊 [{stage_str}] Epoch {epoch+1}/{TOTAL_EPOCHS} -> Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Acc: {max(best_val_acc, val_acc):.2f}% | LR: {current_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(config.OUTPUT_DIR, f"decoupled_{args.model}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Saved New Best Decoupled Model: {save_path} ({best_val_acc:.2f}%)")
        
    print(f"\n🎉 {TOTAL_EPOCHS} Epochs Decoupled Training Completed! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
