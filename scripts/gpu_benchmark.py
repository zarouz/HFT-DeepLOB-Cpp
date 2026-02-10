"""
gpu_benchmark.py
================
Run this BEFORE training to verify:
  1. CUDA is found and A4000 is detected
  2. AMP (float16) is working correctly
  3. Actual throughput with batch_size=1024
  4. Estimated time per epoch for your dataset size

Run from ~/HFT/:
  python scripts/gpu_benchmark.py

If this fails, fix the environment before wasting time on training.
Typical output on A4000 with DeepLOB:
  ~80,000-150,000 samples/sec at batch=1024 with AMP
  Single epoch over 3M samples: ~20-40 seconds
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deeplob_model import DeepLOB, count_parameters
from config import USE_AMP, BATCH_SIZE
from torch.amp import autocast

def run_benchmark():
    print("\n" + "="*60)
    print("GPU BENCHMARK: DeepLOB on RTX A4000")
    print("="*60)

    # ---- Basic checks ----
    print("\n[1] CUDA diagnostics:")
    if not torch.cuda.is_available():
        print("  FATAL: CUDA not available.")
        print("  Check: nvidia-smi, nvcc --version, torch.cuda.is_available()")
        sys.exit(1)

    device    = torch.device('cuda')
    gpu_name  = torch.cuda.get_device_name(0)
    vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
    cuda_ver  = torch.version.cuda
    torch_ver = torch.__version__

    print(f"  GPU:         {gpu_name}")
    print(f"  VRAM:        {vram_gb:.1f} GB")
    print(f"  CUDA:        {cuda_ver}")
    print(f"  PyTorch:     {torch_ver}")
    print(f"  AMP enabled: {USE_AMP}")

    assert 'A4000' in gpu_name or 'NVIDIA' in gpu_name, \
        f"Unexpected GPU: {gpu_name}. Verify you are on the right machine."

    # ---- Model ----
    print(f"\n[2] Model:")
    model = DeepLOB(num_classes=3, T=100, D=40).to(device)
    n_params = count_parameters(model)
    print(f"  Parameters:  {n_params:,}")
    print(f"  Model VRAM:  {torch.cuda.memory_allocated()/1e9:.2f} GB")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- Warmup ----
    print(f"\n[3] Warmup (3 batches, batch_size={BATCH_SIZE})...")
    for _ in range(3):
        X = torch.randn(BATCH_SIZE, 1, 100, 40, device=device, dtype=torch.float16)
        y = torch.randint(0, 3, (BATCH_SIZE,), device=device)
        with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            out  = model(X)
            loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print("  Warmup done.")

    # ---- Throughput test ----
    print(f"\n[4] Throughput (forward + backward, 50 batches):")
    torch.cuda.reset_peak_memory_stats()
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    n_batches = 50
    t_start   = time.perf_counter()

    for _ in range(n_batches):
        X = torch.randn(BATCH_SIZE, 1, 100, 40, device=device, dtype=torch.float16)
        y = torch.randint(0, 3, (BATCH_SIZE,), device=device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            out  = model(X)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed      = t_end - t_start
    total_samp   = n_batches * BATCH_SIZE
    throughput   = total_samp / elapsed
    vram_peak    = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Batches:         {n_batches} x {BATCH_SIZE}")
    print(f"  Total samples:   {total_samp:,}")
    print(f"  Elapsed:         {elapsed:.2f}s")
    print(f"  Throughput:      {throughput:,.0f} samples/sec")
    print(f"  Peak VRAM:       {vram_peak:.2f} GB / {vram_gb:.1f} GB  "
          f"({100*vram_peak/vram_gb:.0f}% used)")

    # ---- Epoch time estimate ----
    print(f"\n[5] Epoch time estimate for your NVDA dataset:")
    for n_samples in [500_000, 1_000_000, 2_000_000, 3_000_000]:
        est_sec = n_samples / throughput
        print(f"  {n_samples/1e6:.1f}M samples -> {est_sec:.0f}s/epoch  "
              f"({est_sec/60:.1f} min)")

    # ---- Batch size recommendation ----
    print(f"\n[6] VRAM headroom analysis:")
    vram_per_sample  = vram_peak / BATCH_SIZE
    max_safe_batch   = int(0.85 * vram_gb / vram_per_sample / 100) * 100
    print(f"  VRAM per sample: {vram_per_sample*1000:.1f} MB")
    print(f"  Current batch:   {BATCH_SIZE}")
    print(f"  Max safe batch:  ~{max_safe_batch} (85% VRAM limit)")
    if max_safe_batch > BATCH_SIZE * 1.5:
        print(f"  SUGGESTION: You can increase BATCH_SIZE to {min(max_safe_batch, 2048)} "
              f"in config.py for better GPU utilization.")
    else:
        print(f"  Batch size {BATCH_SIZE} is well-tuned for your VRAM.")

    print(f"\n{'='*60}")
    if throughput > 50_000:
        print(f"  RESULT: GPU is working correctly. Ready to train.")
    else:
        print(f"  RESULT: Throughput lower than expected ({throughput:,.0f} < 50,000).")
        print(f"  Check: Is another process using the GPU? Run nvidia-smi.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_benchmark()