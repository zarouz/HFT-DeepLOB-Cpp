"""
train_nvda.py
=============
Full training loop for Config A (DeepLOB) on NVDA.
Optimized: Lazy Loading + Persistent Workers + Better Prefetching
Run: python3 scripts/ml/train_nvda.py 2>&1 | tee models/model1_run.log
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
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Load adaptive_labeler
_al_spec = importlib.util.spec_from_file_location('adaptive_labeler_ml', os.path.join(_HERE, 'adaptive_labeler.py'))
_al = importlib.util.module_from_spec(_al_spec); _al_spec.loader.exec_module(_al)
load_snapshots = _al.load_snapshots
filter_session = _al.filter_session
fit_double_alpha_from_files = _al.fit_double_alpha_from_files
label_fixed = _al.label_fixed
normalize_features_zscore = _al.normalize_features_zscore
DoubleAlpha = _al.DoubleAlpha

from config import (
    DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PREFETCH,
    USE_AMP, USE_COMPILE, WINDOW_SIZE, D_FEATURES, NUM_CLASSES,
    EPOCHS, LR, WEIGHT_DECAY, LR_PATIENCE, ES_PATIENCE, VAL_SPLIT,
    TICKER, DATA_DIR, MODEL_DIR, ALPHA_PKL, TRAIN_DATES, TEST_DATES,
    DOWNSAMPLE_STRIDE, print_hardware_summary,
)
from feature_label_builder import build_raw_features, build_dataset
from deeplob_model import build_model, count_parameters


# ── Dataset (LAZY LOADING) ──────────────────────────────────────────────────
class LOBDataset(Dataset):
    """
    Lazy Version: Keeps only the flat (N, 40) array in RAM.
    Slices windows [t-100 : t] on-the-fly during training.
    Reduces RAM usage by 100x.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 window_size: int = WINDOW_SIZE, stride: int = DOWNSAMPLE_STRIDE):
        # Keep features as FloatTensor in RAM (shared)
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.window_size = window_size
        
        # Calculate valid start indices
        N = len(features)
        self.starts = np.arange(window_size - 1, N, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        # On-the-fly slicing
        end_idx = self.starts[idx] + 1
        start_idx = end_idx - self.window_size
        
        # Extract window: (100, 40) -> (1, 100, 40)
        x_window = self.features[start_idx : end_idx].unsqueeze(0)
        y_label = self.labels[self.starts[idx]]
        
        return x_window, y_label


# ── Utilities ───────────────────────────────────────────────────────────────
def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency sampling to balance DOWN/FLAT/UP."""
    classes, counts = np.unique(y, return_counts=True)
    freq = counts / counts.sum()
    weight_per_class = 1.0 / (freq + 1e-8)
    weights_map = np.zeros(NUM_CLASSES)
    for c, w in zip(classes, weight_per_class):
        weights_map[c] = w
    sample_weights = weights_map[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(y),
        replacement=True,
    )


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy loss."""
    classes, counts = np.unique(y, return_counts=True)
    weights = np.zeros(NUM_CLASSES)
    total = counts.sum()
    for c, count in zip(classes, counts):
        weights[c] = 1.0 / (count / total + 1e-8)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ── Training loop (Optimized) ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, use_amp, desc="Training"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar with metrics
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch", ncols=120)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Update stats
        batch_size = len(y_batch)
        total_loss += loss.item() * batch_size
        batch_correct = (logits.argmax(1) == y_batch).sum().item()
        correct += batch_correct
        total += batch_size
        current_acc = batch_correct / batch_size
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.1%}'})
    
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    per_class_recall = {}
    for c, name in enumerate(['DOWN', 'FLAT', 'UP']):
        mask = labels == c
        if mask.sum() > 0:
            per_class_recall[name] = (preds[mask] == c).mean()
        else:
            per_class_recall[name] = 0.0
    
    return total_loss / total, correct / total, per_class_recall


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  DeepLOB Config A Training -- NVDA 100ms Horizon")
    print("  [Lazy Loading + Optimized Workers]")
    print("=" * 60)
    print_hardware_summary()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Optimized settings
    NUM_WORKERS_OPTIMIZED = 8
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 3
    
    print(f"\nDataLoader Optimizations:")
    print(f"  Workers            : {NUM_WORKERS_OPTIMIZED}")
    print(f"  Persistent Workers : {PERSISTENT_WORKERS}")
    print(f"  Prefetch Factor    : {PREFETCH_FACTOR}")
    
    # ── Step 1: Load or fit DoubleAlpha ─────────────────────────────────────
    if os.path.exists(ALPHA_PKL):
        print(f"\nLoading DoubleAlpha from {ALPHA_PKL}")
        try:
            da = DoubleAlpha.load(ALPHA_PKL)
            _ = da.alpha_down, da.alpha_up, da.n_samples, da.asymmetry
            print(f"  {da}")
        except (AttributeError, TypeError) as e:
            print(f"  ⚠ Cached pkl has wrong schema ({e})")
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
        print(f"  {da} → saved to {ALPHA_PKL}")
    
    # ── Step 2: Pre-scan days to compute class weights ──────────────────────
    print(f"\nPre-scanning {len(TRAIN_DATES)} days for class weights...")
    train_paths = [
        os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin')
        for d in TRAIN_DATES
        if os.path.exists(os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin'))
    ]
    
    label_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    valid_paths = []
    
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
    
    # Class weights
    total_samples = label_counts.sum()
    cw = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        if label_counts[i] > 0:
            cw[i] = 1.0 / (label_counts[i] / total_samples)
    cw = cw / cw.sum() * NUM_CLASSES
    class_weights = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    
    print(f"  Label counts: DOWN={label_counts[0]:,} FLAT={label_counts[1]:,} UP={label_counts[2]:,}")
    print(f"  Class weights: DOWN={cw[0]:.3f} FLAT={cw[1]:.3f} UP={cw[2]:.3f}")
    print(f"  Valid days: {len(valid_paths)}/{len(TRAIN_DATES)}")
    
    val_path, val_date = valid_paths[-1]
    train_day_paths = valid_paths[:-1]
    
    # ── Step 3: Build Validation Set (Lazy) ────────────────────────────────
    print(f"\nBuilding validation set from {val_date}...")
    arr_v = load_snapshots(val_path)
    arr_v = filter_session(arr_v)
    lbl_v = label_fixed(arr_v, da)
    feat_v = build_raw_features(arr_v)
    feat_v = normalize_features_zscore(feat_v)
    valid_v = lbl_v != -1
    
    feat_val = feat_v[valid_v]
    lbl_val = lbl_v[valid_v]
    del arr_v, feat_v, lbl_v, valid_v
    
    val_ds = LOBDataset(feat_val, lbl_val, window_size=WINDOW_SIZE, stride=DOWNSAMPLE_STRIDE)
    
    # Optimized validation loader
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE * 2, 
        shuffle=False,
        num_workers=NUM_WORKERS_OPTIMIZED,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )
    print(f"  Val samples: {len(val_ds):,}")
    
    # ── Step 4: Model & Optimizer ───────────────────────────────────────────
    model = build_model('A').to(DEVICE)
    
    if USE_COMPILE:
        print("Compiling model (torch.compile)...")
        print("  [NOTE] First epoch will be slow due to JIT compilation.")
        model = torch.compile(model)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, factor=0.5, mode='max')
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    print(f"\nModel: {count_parameters(model):,} parameters")
    print(f"Training days per epoch: {len(train_day_paths)}")
    
    # ── Step 5: Training Loop ───────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    patience_left = ES_PATIENCE
    ckpt_path = os.path.join(MODEL_DIR, f'{TICKER}_deeplob_best.pt')
    
    print(f"\n{'Epoch':>5} {'Day':>14} {'Loss':>8} {'Acc':>7} "
          f"{'Val Acc':>8} {'DOWN':>6} {'FLAT':>6} {'UP':>6} "
          f"{'LR':>8} {'Time':>6}")
    print("-" * 100)
    
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        ep_loss, ep_correct, ep_total = 0.0, 0.0, 0
        
        # Shuffle day order
        day_order = list(train_day_paths)
        random.shuffle(day_order)
        
        for i, (path, date) in enumerate(day_order):
            # Load one day
            arr = load_snapshots(path)
            arr = filter_session(arr)
            lbl = label_fixed(arr, da)
            feat = build_raw_features(arr)
            feat = normalize_features_zscore(feat)
            valid = lbl != -1
            
            feat_d = feat[valid]
            lbl_d = lbl[valid]
            del arr, feat, lbl, valid
            
            if len(feat_d) < 200:
                continue
            
            # Lazy Dataset
            day_ds = LOBDataset(feat_d, lbl_d, window_size=WINDOW_SIZE, stride=DOWNSAMPLE_STRIDE)
            
            # Weighted Sampler
            strided_labels = lbl_d[day_ds.starts]
            day_sampler = make_weighted_sampler(strided_labels)
            
            # Optimized DataLoader
            day_loader = DataLoader(
                day_ds,
                batch_size=BATCH_SIZE,
                sampler=day_sampler,
                num_workers=NUM_WORKERS_OPTIMIZED,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS,
                prefetch_factor=PREFETCH_FACTOR
            )
            
            # Pass description to train_epoch for the progress bar
            desc = f"Epoch {epoch} [{date}]"
            d_loss, d_acc = train_epoch(
                model, day_loader, optimizer, criterion, scaler, USE_AMP, desc=desc)
            
            d_total = len(day_ds)
            d_correct = d_acc * d_total
            
            ep_loss += d_loss * d_total
            ep_correct += d_correct
            ep_total += d_total
            
            del day_ds, day_loader, feat_d, lbl_d
        
        # End of epoch: validate
        if ep_total == 0:
            continue
        
        tr_loss = ep_loss / ep_total
        tr_acc = ep_correct / ep_total
        val_loss, val_acc, pcr = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t_ep
        
        print(f"{epoch:>5} {'all days':>14} {tr_loss:>8.4f} {tr_acc:>7.4f} "
              f"{val_acc:>8.4f} {pcr['DOWN']:>6.3f} {pcr['FLAT']:>6.3f} "
              f"{pcr['UP']:>6.3f} {lr_now:>8.2e} {elapsed:>5.1f}s")
        
        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_left = ES_PATIENCE
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_acc': val_acc,
                'da': da,
                'config': 'A',
            }
            torch.save(state, ckpt_path)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print(f"\nBest val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint : {ckpt_path}")
    
    # ── Step 6: Test set evaluation ─────────────────────────────────────────
    print(f"\nLoading best checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    
    print(f"Evaluating on {len(TEST_DATES)} test days...")
    test_paths = [
        os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin')
        for d in TEST_DATES
        if os.path.exists(os.path.join(DATA_DIR, f'{TICKER}_{d}_dataset.bin'))
    ]
    
    all_test_preds, all_test_labels = [], []
    test_loss_sum, test_total = 0.0, 0
    
    for path, date in zip(test_paths, TEST_DATES):
        arr = load_snapshots(path)
        arr = filter_session(arr)
        if len(arr) < 200:
            continue
        
        lbl = label_fixed(arr, da)
        feat = build_raw_features(arr)
        feat = normalize_features_zscore(feat)
        valid = lbl != -1
        
        feat_d = feat[valid]
        lbl_d = lbl[valid]
        del arr, feat, lbl, valid
        
        if len(feat_d) < 200:
            continue
        
        day_ds = LOBDataset(feat_d, lbl_d, window_size=WINDOW_SIZE, stride=DOWNSAMPLE_STRIDE)
        day_loader = DataLoader(
            day_ds, 
            batch_size=BATCH_SIZE * 2, 
            shuffle=False,
            num_workers=NUM_WORKERS_OPTIMIZED,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR
        )
        
        d_loss, d_acc, _ = eval_epoch(model, day_loader, criterion)
        
        model.eval()
        with torch.no_grad():
            for xb, yb in day_loader:
                logits = model(xb.to(DEVICE, non_blocking=True))
                all_test_preds.append(logits.argmax(1).cpu().numpy())
                all_test_labels.append(yb.numpy())
        
        test_loss_sum += d_loss * len(day_ds)
        test_total += len(day_ds)
        
        del day_ds, day_loader, feat_d, lbl_d
        print(f"  {date}: acc={d_acc:.4f} loss={d_loss:.4f} n={test_total:,}", flush=True)
    
    preds = np.concatenate(all_test_preds)
    labels = np.concatenate(all_test_labels)
    test_acc = (preds == labels).mean()
    test_loss = test_loss_sum / max(test_total, 1)
    
    test_pcr = {}
    for c, name in enumerate(['DOWN', 'FLAT', 'UP']):
        mask = labels == c
        if mask.sum() > 0:
            test_pcr[name] = (preds[mask] == c).mean()
        else:
            test_pcr[name] = 0.0
    
    print("\n" + "=" * 60)
    print(f"  TEST SET RESULTS (Config A, NVDA)")
    print("=" * 60)
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Loss : {test_loss:.4f}")
    print(f"  DOWN recall: {test_pcr['DOWN']:.4f}")
    print(f"  FLAT recall: {test_pcr['FLAT']:.4f}")
    print(f"  UP recall: {test_pcr['UP']:.4f}")
    print("=" * 60)
    
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