import os
import glob
import cv2
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import albumentations.pytorch as Ap
import numpy as np

from configs.config import get_default_config
from src.models.sr import StackedSRNet
from src.training.sr_loss import SRLoss
import torch.amp

class SRDataset(Dataset):
    """
    Dataset for standalone SR training.
    Loads 5 LR frames and 1 HR frame (center frame).
    """
    def __init__(self, data_root, split_file, mode='train', img_height=16, img_width=48, scale_factor=2):
        self.data_root = data_root
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width
        self.scale_factor = scale_factor
        self.hr_height = img_height * scale_factor
        self.hr_width = img_width * scale_factor
        
        self.samples = []
        
        # Load valid tracks from split file
        with open(split_file, 'r') as f:
            val_ids = set(json.load(f))
            
        from src.data.dataset import MultiFrameDataset
        
        # Get all tracks
        abs_root = os.path.abspath(data_root)
        all_tracks = sorted(glob.glob(os.path.join(abs_root, "**", "track_*"), recursive=True))
                    
        # Filter train/val
        tracks = []
        for t in all_tracks:
            unique_id = MultiFrameDataset.get_track_unique_id(t)
            base_name = os.path.basename(t)
            is_val = unique_id in val_ids or (base_name in val_ids and "Scenario-B" in t)
            
            if mode == 'train' and not is_val:
                tracks.append(t)
            elif mode == 'val' and is_val:
                tracks.append(t)
                
        # Index samples (only use tracks with HR images)
        for t in tracks:
            lr_files = sorted(glob.glob(os.path.join(t, "lr-*.png")) + glob.glob(os.path.join(t, "lr-*.jpg")))
            hr_files = sorted(glob.glob(os.path.join(t, "hr-*.png")) + glob.glob(os.path.join(t, "hr-*.jpg")))
            
            # We need HR for supervised SR training
            if len(hr_files) > 0 and len(lr_files) > 0:
                self.samples.append({
                    'lr_paths': lr_files,
                    'hr_path': hr_files[len(hr_files)//2] # Center frame as target HR
                })

        # Transforms
        if self.mode == 'train':
            self.transform = A.Compose([
                A.ColorJitter(p=0.5),
                A.Normalize(),
                Ap.ToTensorV2()
            ], additional_targets={f'image{i}': 'image' for i in range(1, 5)})
        else:
            self.transform = A.Compose([
                A.Normalize(),
                Ap.ToTensorV2()
            ], additional_targets={f'image{i}': 'image' for i in range(1, 5)})

        # Transform for HR (only normalization)
        self.hr_transform = A.Compose([
            A.Normalize(),
            Ap.ToTensorV2()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        lr_paths = item['lr_paths']
        hr_path = item['hr_path']
        
        # Load LR
        lr_images = []
        for p in lr_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_width, self.img_height))
                lr_images.append(img)
                
        # Pad to 5 frames
        while len(lr_images) < 5:
            if len(lr_images) > 0:
                lr_images.append(lr_images[-1])
            else:
                lr_images.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
        lr_images = lr_images[:5]

        # Load HR
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        if hr_img is not None:
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_img = cv2.resize(hr_img, (self.hr_width, self.hr_height))
        else:
            hr_img = np.zeros((self.hr_height, self.hr_width, 3), dtype=np.uint8)

        # Apply transforms synchronously to LR
        targets = {f'image{i}': lr_images[i] for i in range(1, 5)}
        res = self.transform(image=lr_images[0], **targets)
        lr_tensors = [res['image']] + [res[f'image{i}'] for i in range(1, 5)]
        lr_tensor = torch.stack(lr_tensors, dim=0) # [5, 3, H, W]

        # Apply transform to HR
        hr_tensor = self.hr_transform(image=hr_img)['image'] # [3, H*2, W*2]
        
        return lr_tensor, hr_tensor

def main():
    config = get_default_config()
    
    # Standalone SR config
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4
    
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Dummy creation of val split file if it doesn't exist for test
    if not os.path.exists(config.VAL_SPLIT_FILE):
        os.makedirs(os.path.dirname(config.VAL_SPLIT_FILE), exist_ok=True)
        with open(config.VAL_SPLIT_FILE, 'w') as f:
            json.dump([], f)
            
    # Mock data directories if they don't exist for testing
    if not os.path.exists(config.DATA_ROOT):
        print(f"⚠️ {config.DATA_ROOT} not found. Creating dummy data for testing...")
        os.makedirs(os.path.join(config.DATA_ROOT, "Scenario-A", "test_track"), exist_ok=True)
        dummy_img = np.zeros((16, 48, 3), dtype=np.uint8)
        dummy_hr = np.zeros((32, 96, 3), dtype=np.uint8)
        for i in range(5):
            cv2.imwrite(os.path.join(config.DATA_ROOT, "Scenario-A", "test_track", f"lr-{i}.png"), dummy_img)
            cv2.imwrite(os.path.join(config.DATA_ROOT, "Scenario-A", "test_track", f"hr-{i}.png"), dummy_hr)

    # Datasets
    train_dataset = SRDataset(config.DATA_ROOT, config.VAL_SPLIT_FILE, mode='train', 
                             img_height=config.IMG_HEIGHT//2, img_width=config.IMG_WIDTH//2) # 16x64
    val_dataset = SRDataset(config.DATA_ROOT, config.VAL_SPLIT_FILE, mode='val',
                             img_height=config.IMG_HEIGHT//2, img_width=config.IMG_WIDTH//2)

    # Note: LRLPR uses 32x128 for OCR, so LR is 16x64. Let's use 16x64 -> 32x128
    
    print(f"Train samples: {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Model
    model = StackedSRNet(num_frames=5, in_channels=3, base_channels=64, num_blocks=8, scale_factor=2).to(device)
    
    # Loss
    criterion = SRLoss(w_mse=1.0, w_perceptual=0.1, w_edge=0.1, device=device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Scaler for AMP
    scaler = torch.amp.GradScaler(device.type) if device.type in ['cuda', 'mps'] else None
    
    print("Starting SR Training...")
    
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type) if scaler else torch.autocast('cpu', enabled=False):
                sr = model(lr)
                loss = criterion(sr, hr)
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "sr_net_best.pth"))
            print(f"  ⭐ Saved Best SR Model (Loss: {best_loss:.4f})")
            
    print("✅ Standalone SR Training Complete!")

if __name__ == "__main__":
    main()
