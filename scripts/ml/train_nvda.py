"""
train_nvda.py
=============
Full training loop for Config A (DeepLOB) on NVDA.
Run: python scripts/ml/train_nvda.py 2>&1 | tee ~/HFT/models/train_run1.log

Pipeline:
  1. Build multi-day dataset from 10 training days
  2. Chronological train/val split (last 15% of each day = val)
  3. WeightedRandomSampler to balance DOWN/FLAT/UP
  4. Training loop with AMP (CUDA) / standard (MPS/CPU)
  5. ReduceLROnPlateau + early stopping
  6. Save best val_acc checkpoint
  7. Test set evaluation with training DoubleAlpha

Batch size:
  Linux A4000 : 1024 (config.py default)
  M2 Pro MPS  : 256  (config.py auto-sets)
"""

import os
import sys
import time
import pickle
import random
import numpy as np
import torch
import importlib.util
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Load adaptive_labeler from scripts/ml/ explicitly (avoids old scripts/ version)
_al_spec = importlib.util.spec_from_file_location('adaptive_labeler_ml',
                                                    os.path.join(_HERE, 'adaptive_labeler.py'))
_al = importlib.util.module_from_spec(_al_spec); _al_spec.loader.exec_module(_al)
load_snapshots            = _al.load_snapshots
filter_session            = _al.filter_session
fit_double_alpha_from_files = _al.fit_double_alpha_from_files
label_fixed               = _al.label_fixed
normalize_features_zscore = _al.normalize_features_zscore
DoubleAlpha               = _al.DoubleAlpha

from config import (
    DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PREFETCH,
    USE_AMP, USE_COMPILE,
    WINDOW_SIZE, D_FEATURES, NUM_CLASSES,
    EPOCHS, LR, WEIGHT_DECAY, LR_PATIENCE, ES_PATIENCE, VAL_SPLIT,
    TICKER, DATA_DIR, MODEL_DIR, ALPHA_PKL,
    TRAIN_DATES, TEST_DATES, DOWNSAMPLE_STRIDE,
    print_hardware_summary,
)
from feature_label_builder import build_raw_features, build_sliding_windows, build_dataset
from deeplob_model import build_model, count_parameters


# ── Dataset ───────────────────────────────────────────────────────────────────
class LOBDataset(Dataset):
    """(M, 100, 40) windows → (M, 1, 100, 40) for DeepLOB."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).unsqueeze(1)   # (M, 1, T, F)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Utilities ─────────────────────────────────────────────────────────────────
def chronological_split(X: np.ndarray, y: np.ndarray, val_frac: float = VAL_SPLIT):
    """
    Chronological train/val split. Random splits leak future data.
    Last val_frac of the combined sequence is validation.
    """
    n = len(y)
    split = int(n * (1 - val_frac))
    return X[:split], y[:split], X[split:], y[split:]


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency sampling to balance DOWN/FLAT/UP during training."""
    classes, counts = np.unique(y, return_counts=True)
    freq = counts / counts.sum()
    weight_per_class = 1.0 / (freq + 1e-8)
    sample_weights = weight_per_class[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(y),
        replacement=True,
    )


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy loss."""
    _, counts = np.unique(y, return_counts=True)
    weights = 1.0 / (counts / counts.sum() + 1e-8)
    weights = weights / weights.sum() * NUM_CLASSES   # normalise to sum=num_classes
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, use_amp):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    # Per-class recall
    per_class_recall = {}
    for c, name in enumerate(['DOWN', 'FLAT', 'UP']):
        mask = labels == c
        per_class_recall[name] = preds[mask].mean() == c if mask.sum() > 0 else 0.0
        per_class_recall[name] = (preds[mask] == c).mean() if mask.sum() > 0 else 0.0

    return total_loss / total, correct / total, per_class_recall


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  DeepLOB Config A Training -- NVDA 100ms Horizon")
    print("=" * 60)
    print_hardware_summary()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Step 1: Load or fit DoubleAlpha ───────────────────────────────────────
    if os.path.exists(ALPHA_PKL):
        print(f"\nLoading DoubleAlpha from {ALPHA_PKL}")
        try:
            da = DoubleAlpha.load(ALPHA_PKL)
            # Sanity check: ensure loaded object has the right schema
            _ = da.alpha_down, da.alpha_up, da.n_samples, da.asymmetry
            print(f"  {da}")
        except (AttributeError, TypeError) as e:
            print(f"  ⚠ Cached pkl has wrong schema ({e})")
            print(f"  → Deleting stale pkl and re-fitting...")
            os.remove(ALPHA_PKL)
            da = None
    else:
        da = None

    if da is None:
        print("\nFitting DoubleAlpha from 10 training days...")
        train_paths = [
            os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin')
            for d in TRAIN_DATES
            if os.path.exists(os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin'))
        ]
        n_workers = min(len(train_paths), NUM_WORKERS)
        da = fit_double_alpha_from_files(train_paths, n_workers=n_workers)
        da.save(ALPHA_PKL)
        print(f"  {da}  → saved to {ALPHA_PKL}")

    # ── Step 2: Pre-scan days to compute class weights for loss ──────────────
    # Scan labels across all days without keeping windows (just label counts).
    # Peak RAM this step: one day of features ~1-2 GB, freed after each day.
    print(f"\nPre-scanning {len(TRAIN_DATES)} days for class weights (stride={DOWNSAMPLE_STRIDE})...")
    train_paths = [
        os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin')
        for d in TRAIN_DATES
        if os.path.exists(os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin'))
    ]

    label_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    valid_paths  = []
    for path, date in zip(train_paths, TRAIN_DATES):
        arr = load_snapshots(path)
        arr = filter_session(arr)
        if len(arr) < 200:
            continue
        lbl = label_fixed(arr, da)
        valid = lbl != -1
        lbl_v = lbl[valid]
        if len(lbl_v) < 200:
            continue
        for c in range(NUM_CLASSES):
            label_counts[c] += (lbl_v == c).sum()
        valid_paths.append((path, date))
        del arr, lbl, valid, lbl_v

    freq = label_counts / label_counts.sum()
    cw   = 1.0 / (freq + 1e-8)
    cw   = cw / cw.sum() * NUM_CLASSES
    class_weights = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    print(f"  Label counts: DOWN={label_counts[0]:,}  FLAT={label_counts[1]:,}  "
          f"UP={label_counts[2]:,}")
    print(f"  Class weights: DOWN={cw[0]:.3f}  FLAT={cw[1]:.3f}  UP={cw[2]:.3f}")
    print(f"  Valid days: {len(valid_paths)}/{len(TRAIN_DATES)}")

    # Hold out last day as the fixed validation set (chronological)
    # Rest are training days rotated through each epoch
    val_path, val_date   = valid_paths[-1]
    train_day_paths      = valid_paths[:-1]

    # Build validation set once (last train day, held out permanently)
    print(f"\nBuilding validation set from {val_date}...")
    arr_v   = load_snapshots(val_path)
    arr_v   = filter_session(arr_v)
    lbl_v   = label_fixed(arr_v, da)
    feat_v  = build_raw_features(arr_v)
    feat_v  = normalize_features_zscore(feat_v)
    valid_v = lbl_v != -1
    X_val, y_val = build_sliding_windows(feat_v[valid_v], lbl_v[valid_v],
                                          stride=DOWNSAMPLE_STRIDE)
    del arr_v, feat_v, lbl_v, valid_v
    val_ds     = LOBDataset(X_val, y_val)
    del X_val, y_val
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"  Val windows: {len(val_ds):,}  RAM: {val_ds.X.nbytes/1e9:.2f} GB")

    # ── Step 3: Model, loss, optimiser ───────────────────────────────────────
    model = build_model('A').to(DEVICE)
    if USE_COMPILE:
        print("Compiling model (torch.compile)...")
        model = torch.compile(model)

    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=0.5, mode='max')
    scaler     = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    print(f"\nModel: {count_parameters(model):,} trainable parameters")
    print(f"Training days per epoch: {len(train_day_paths)}  "
          f"(streaming, peak RAM = 1 day at a time)")
    print(f"DOWNSAMPLE_STRIDE={DOWNSAMPLE_STRIDE}  "
          f"→ ~{7_000_000//DOWNSAMPLE_STRIDE:,} windows/day")

    # ── Step 4: Epoch loop — stream one day at a time ────────────────────────
    # Each epoch iterates over all training days sequentially.
    # Only one day's windows exist in RAM at any moment.
    # Peak RAM: largest single day × 2 (numpy + tensor) ≈ 3-5 GB at stride=32
    best_val_acc  = 0.0
    best_epoch    = 0
    patience_left = ES_PATIENCE
    ckpt_path     = os.path.join(MODEL_DIR, f'{TICKER}_deeplob_best.pt')

    print(f"\n{'Epoch':>5}  {'Day':>14}  {'Loss':>8}  {'Acc':>7}  "
          f"{'Val Acc':>8}  {'DOWN':>6}  {'FLAT':>6}  {'UP':>6}  "
          f"{'LR':>8}  {'Time':>6}")
    print("-" * 100)

    for epoch in range(1, EPOCHS + 1):
        t_ep  = time.time()
        ep_loss, ep_correct, ep_total = 0.0, 0.0, 0

        # Shuffle day order each epoch so model doesn't overfit to day sequence
        day_order = list(train_day_paths)
        random.shuffle(day_order)

        for path, date in day_order:
            # Load one day
            arr   = load_snapshots(path)
            arr   = filter_session(arr)
            lbl   = label_fixed(arr, da)
            feat  = build_raw_features(arr)
            feat  = normalize_features_zscore(feat)
            valid = lbl != -1
            feat_v, lbl_v = feat[valid], lbl[valid]
            del arr, feat, lbl, valid

            if len(feat_v) < 200:
                continue

            X_day, y_day = build_sliding_windows(feat_v, lbl_v,
                                                   stride=DOWNSAMPLE_STRIDE)
            del feat_v, lbl_v

            # WeightedRandomSampler for this day
            day_ds      = LOBDataset(X_day, y_day)
            del X_day, y_day
            day_sampler = make_weighted_sampler(day_ds.y.numpy())
            # num_workers=0: avoids macOS spawn cost of 6 processes × 9 days × N epochs
            # The bottleneck is CPU preprocessing above, not DataLoader I/O
            day_loader  = DataLoader(
                day_ds, batch_size=BATCH_SIZE, sampler=day_sampler,
                num_workers=0, pin_memory=False,
            )

            d_loss, d_correct, d_total = train_epoch(
                model, day_loader, optimizer, criterion, scaler, USE_AMP)
            ep_loss    += d_loss * d_total
            ep_correct += d_correct * d_total
            ep_total   += d_total

            del day_ds, day_loader  # free GPU + RAM before next day

        # End of epoch: validate
        if ep_total == 0:
            continue
        tr_loss = ep_loss / ep_total
        tr_acc  = ep_correct / ep_total
        val_loss, val_acc, pcr = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)
        lr_now  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t_ep

        print(f"{epoch:>5}  {'all days':>14}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{val_acc:>8.4f}  {pcr['DOWN']:>6.3f}  {pcr['FLAT']:>6.3f}  "
              f"{pcr['UP']:>6.3f}  {lr_now:>8.2e}  {elapsed:>5.1f}s")

        # Checkpoint on val_acc improvement
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_epoch    = epoch
            patience_left = ES_PATIENCE
            state = {
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'da':         da,
                'config':     'A',
            }
            torch.save(state, ckpt_path)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best: {best_val_acc:.4f} @ epoch {best_epoch})")
                break

        # Warn if directional recall collapses after epoch 5
        if epoch == 5:
            dir_recall = pcr['DOWN'] + pcr['UP']
            if dir_recall < 0.40:
                print(f"\n⚠  Directional recall {dir_recall:.3f} < 0.40 after epoch 5.")
                print("   → Increase class weights ×2 and restart.")

    print(f"\nBest val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint  : {ckpt_path}")

    # ── Step 5: Test set evaluation (streaming, one day at a time) ───────────
    # Load best checkpoint first
    print(f"\nLoading best checkpoint (epoch {best_epoch}, val_acc={best_val_acc:.4f})...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])

    print(f"Evaluating on {len(TEST_DATES)} test days (streaming)...")
    test_paths = [
        os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin')
        for d in TEST_DATES
        if os.path.exists(os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin'))
    ]

    all_test_preds, all_test_labels = [], []
    test_loss_sum, test_total = 0.0, 0

    for path, date in zip(test_paths, TEST_DATES):
        arr   = load_snapshots(path)
        arr   = filter_session(arr)
        if len(arr) < 200:
            continue
        lbl   = label_fixed(arr, da)   # training da — never re-fit on test
        feat  = build_raw_features(arr)
        feat  = normalize_features_zscore(feat)
        valid = lbl != -1
        feat_v, lbl_v = feat[valid], lbl[valid]
        del arr, feat, lbl, valid
        if len(feat_v) < 200:
            continue

        X_d, y_d = build_sliding_windows(feat_v, lbl_v, stride=DOWNSAMPLE_STRIDE)
        del feat_v, lbl_v
        day_ds     = LOBDataset(X_d, y_d); del X_d, y_d
        day_loader = DataLoader(day_ds, batch_size=BATCH_SIZE * 2,
                                shuffle=False, num_workers=0, pin_memory=False)
        d_loss, d_acc, _ = eval_epoch(model, day_loader, criterion)

        # Collect predictions for confusion matrix
        model.eval()
        with torch.no_grad():
            for xb, yb in day_loader:
                logits = model(xb.to(DEVICE))
                all_test_preds.append(logits.argmax(1).cpu().numpy())
                all_test_labels.append(yb.numpy())
        test_loss_sum += d_loss * len(day_ds)
        test_total    += len(day_ds)
        del day_ds, day_loader
        print(f"  {date}: acc={d_acc:.4f}  loss={d_loss:.4f}  n={test_total:,}")

    preds  = np.concatenate(all_test_preds)
    labels = np.concatenate(all_test_labels)
    test_acc  = (preds == labels).mean()
    test_loss = test_loss_sum / max(test_total, 1)
    test_pcr  = {name: (preds[labels==c] == c).mean() if (labels==c).sum() > 0 else 0.0
                 for c, name in enumerate(['DOWN', 'FLAT', 'UP'])}

    test_ds     = None  # streaming — no monolithic test dataset
    test_loader = None

    print("\n" + "=" * 60)
    print(f"  TEST SET RESULTS (Config A, NVDA)")
    print("=" * 60)
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Loss     : {test_loss:.4f}")
    print(f"  DOWN recall: {test_pcr['DOWN']:.4f}")
    print(f"  FLAT recall: {test_pcr['FLAT']:.4f}")
    print(f"  UP   recall: {test_pcr['UP']:.4f}")
    print("=" * 60)

    # Save test results
    results_path = os.path.join(MODEL_DIR, f'{TICKER}_config_a_test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'test_acc': test_acc,
            'test_loss': test_loss,
            'per_class_recall': test_pcr,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'da': da,
            'config': 'A',
        }, f)
    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    main()