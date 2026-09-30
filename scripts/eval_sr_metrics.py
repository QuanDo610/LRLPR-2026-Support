import os
import sys
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import get_default_config
from src.data.dataset import MultiFrameDataset
from src.models.sr import StackedSRNet
from src.utils.common import seed_everything

def denorm_to_uint8(tensor):
    """Convert tensor from [-1, 1] to uint8 [0, 255]."""
    tensor = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
    tensor = (tensor * 255.0).cpu().numpy().astype(np.uint8)
    return tensor

def evaluate_sr_metrics(model, dataloader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    print("🔍 Evaluating PSNR & SSIM on Synthetic Data...")
    with torch.no_grad():
        for lr, hr, labels, label_lengths, is_synth in tqdm(dataloader, desc="Eval SR"):
            # Only evaluate synthetic data since it has valid HR ground truth
            if not torch.tensor(is_synth).any():
                continue
                
            is_synth = torch.tensor(is_synth); lr_synth = lr[is_synth].to(device)
            hr_synth = hr[is_synth] # [B, 3, H, W]
            
            # Predict
            sr_image = model(lr_synth) # [B, F, 3, H, W]
            
            # Extract center frame
            center_idx = sr_image.size(1) // 2
            sr_synth_center = sr_image[:, center_idx].cpu()
            
            for i in range(sr_synth_center.size(0)):
                # Convert to HWC for skimage
                sr_img_np = denorm_to_uint8(sr_synth_center[i]).transpose(1, 2, 0)
                hr_img_np = denorm_to_uint8(hr_synth[i]).transpose(1, 2, 0)
                
                # Calculate metrics
                val_psnr = psnr(hr_img_np, sr_img_np)
                # Ensure win_size is smaller than image dims. For 32x128, win_size 7 is safe.
                val_ssim = ssim(hr_img_np, sr_img_np, channel_axis=-1, data_range=255, win_size=7)
                
                total_psnr += val_psnr
                total_ssim += val_ssim
                count += 1
                
    if count == 0:
        return 0.0, 0.0
        
    return total_psnr / count, total_ssim / count

def main():
    config = get_default_config()
    device = config.DEVICE
    seed_everything(config.SEED)
    
    # Load dataset
    val_dataset = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='val',
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        collate_fn=MultiFrameDataset.collate_fn, num_workers=2
    )
    
    # Load Model
    model = StackedSRNet(num_blocks=8, scale_factor=2).to(device)
    sr_weights_path = os.path.join(config.OUTPUT_DIR, "sr_net_best.pth")
    
    if os.path.exists(sr_weights_path):
        print(f"📦 Loading SR weights from {sr_weights_path}")
        # Use strict=False in case of architectural adjustments
        model.load_state_dict(torch.load(sr_weights_path, map_location=device), strict=False)
    else:
        print("⚠️ Warning: sr_net_best.pth not found. Evaluating with random weights.")
        
    avg_psnr, avg_ssim = evaluate_sr_metrics(model, val_loader, device)
    
    print("\n" + "="*50)
    print("📊 SUPER-RESOLUTION METRICS (Synthetic Val Set)")
    print("="*50)
    print(f"   - Average PSNR : {avg_psnr:.2f} dB")
    print(f"   - Average SSIM : {avg_ssim:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
