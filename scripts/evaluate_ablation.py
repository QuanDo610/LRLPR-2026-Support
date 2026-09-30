import torch
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.end2end_lpr import End2EndLPR
from configs.config import get_default_config

def measure_latency(model, input_tensor, device, num_runs=100):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)
            
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(input_tensor)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
    latencies.sort()
    p95_latency = latencies[int(0.95 * len(latencies))]
    return p95_latency

def main():
    print("=== Task 4.1 & 4.3: Ablation & Resource Benchmark ===")
    config = get_default_config()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    print(f"Running on Device: {device}")
    
    # Define Ablations
    ablations = [
        {"name": "(1) Baseline (No SR, Bicubic)", "use_sr": False, "use_stn": True},
        {"name": "(2) StackedSRNet (MSE only)", "use_sr": True, "use_stn": True},
        {"name": "(3) StackedSRNet (MSE+Perc+Edge)", "use_sr": True, "use_stn": True},
        {"name": "(4) No STN (SR + ResTran)", "use_sr": True, "use_stn": False},
    ]
    
    dummy_input = torch.randn(1, 5, 3, 16, 48).to(device)
    
    print(f"{'Experiment':<35} | {'Params (M)':<12} | {'P95 Latency (ms)':<16}")
    print("-" * 70)
    
    for exp in ablations:
        if exp["use_sr"]:
            model = End2EndLPR(
                num_classes=config.NUM_CLASSES,
                sr_num_blocks=8,
                use_stn=exp["use_stn"],
                pretrained=False
            ).to(device)
        else:
            # Baseline just uses ResTranOCR, assume upscaled input
            from src.models.restran import ResTranOCR
            model = ResTranOCR(
                num_classes=config.NUM_CLASSES,
                use_stn=exp["use_stn"],
                pretrained=False
            ).to(device)
            # ResTran expects [Batch, Frames, 3, H, W] but at 32x96 (upscaled)
            # So for baseline, we feed upscaled dummy input
            dummy_input_baseline = torch.randn(1, 5, 3, 32, 96).to(device)
            
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Measure latency
        if exp["use_sr"]:
            latency = measure_latency(model, dummy_input, device)
        else:
            latency = measure_latency(model, dummy_input_baseline, device)
            
        print(f"{exp['name']:<35} | {total_params:>10.2f} M | {latency:>10.2f} ms")

if __name__ == "__main__":
    main()
