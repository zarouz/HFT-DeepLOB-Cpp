# 1. Verify everything looks right
python scripts/ml/preflight.py

# 2. Check label distributions (optional but fast)
python scripts/ml/tune_alpha.py data/datasets/clean/daily/nvda_20260105_dataset.bin

# 3. Smoke test feature builder on one day
python scripts/ml/feature_label_builder.py data/datasets/clean/daily/nvda_20260105_dataset.bin

# 4. Fit and save DoubleAlpha from all 10 training days
python scripts/ml/fit_alpha.py

# 5. Train
python scripts/ml/train_nvda.py 2>&1 | tee ~/HFT/models/train_run1.log