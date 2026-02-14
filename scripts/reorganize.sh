#!/usr/bin/env bash
# ============================================================
# reorganize.sh
# Run from the root of your HFT project: bash scripts/reorganize.sh
# ============================================================

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$SCRIPTS_DIR/ml"

echo "HFT Scripts Reorganisation"
echo "=================================="
echo "Project scripts dir : $SCRIPTS_DIR"
echo "ML destination      : $ML_DIR"
echo ""

# -- Create ml/ subdirectory
mkdir -p "$ML_DIR"
echo "[+] Created $ML_DIR"

# -------------------------------------------------------
# ML scripts → scripts/ml/
# -------------------------------------------------------
ML_SCRIPTS=(
    "config.py"
    "deeplob_model.py"
    "feature_label_builder.py"
    "train_nvda.py"
    "tune_alpha.py"
    "gpu_benchmark.py"
)

echo ""
echo "Moving ML scripts → scripts/ml/"
for f in "${ML_SCRIPTS[@]}"; do
    if [ -f "$SCRIPTS_DIR/$f" ]; then
        mv "$SCRIPTS_DIR/$f" "$ML_DIR/$f"
        echo "  ✓ $f"
    else
        echo "  ! $f not found (skipping)"
    fi
done

# -------------------------------------------------------
# Data-pipeline scripts stay in scripts/ (no action needed)
# -------------------------------------------------------
echo ""
echo "Data pipeline scripts remain in scripts/ (no changes):"
DATA_SCRIPTS=(
    "adaptive_labeler.py"
    "concatenate_datasets.py"
    "convert_amd_only.py"
    "convert_amd_parallel.py"
    "diagnose.py"
    "fix_amd_jan27.py"
    "generate_datasets_parallel.py"
    "preflight.py"
    "process_all_days_parallel.py"
    "regenerate_all_datasets.py"
    "regenerate_amd.py"
    "verify_all_days.py"
)
for f in "${DATA_SCRIPTS[@]}"; do
    if [ -f "$SCRIPTS_DIR/$f" ]; then
        echo "  · $f"
    else
        echo "  ! $f not found"
    fi
done

# -------------------------------------------------------
# Update sys.path in ML scripts so they can still find
# each other when run from any directory
# -------------------------------------------------------
echo ""
echo "Patching sys.path in ML scripts..."

# Prepend a path-fix block to each ML script that doesn't already have one
PATH_FIX='import sys, os
# Allow imports from scripts/ (data utils) and scripts/ml/ (ml modules)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)          # scripts/
_PROJECT = os.path.dirname(_SCRIPTS)       # HFT/
for _p in [_HERE, _SCRIPTS, _PROJECT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
'

for f in "${ML_SCRIPTS[@]}"; do
    TARGET="$ML_DIR/$f"
    if [ -f "$TARGET" ]; then
        # Only patch if not already patched
        if ! grep -q "_HERE = os.path.dirname" "$TARGET" 2>/dev/null; then
            # Insert after the module docstring (first non-comment, non-blank line
            # that starts with """) or at the top if no docstring
            TMPFILE=$(mktemp)
            python3 - "$TARGET" "$PATH_FIX" > "$TMPFILE" << 'PYEOF'
import sys

filepath = sys.argv[1]
path_fix = sys.argv[2]

with open(filepath) as fh:
    lines = fh.readlines()

# Find insertion point: after the closing triple-quote of a module docstring
# or after the last leading comment block (# lines), or at position 0
insert_at = 0
in_docstring = False
docstring_char = None

for i, line in enumerate(lines):
    stripped = line.strip()
    if i == 0 and not in_docstring:
        if stripped.startswith('"""') or stripped.startswith("'''"):
            docstring_char = stripped[:3]
            # single-line docstring?
            rest = stripped[3:]
            if rest.endswith(docstring_char) and len(rest) > 3:
                insert_at = i + 1
                break
            in_docstring = True
            continue
    if in_docstring:
        if docstring_char and docstring_char in line:
            insert_at = i + 1
            in_docstring = False
            break
    elif stripped.startswith('#') or stripped == '':
        insert_at = i + 1
    else:
        break

new_lines = lines[:insert_at] + ['\n', path_fix, '\n'] + lines[insert_at:]

with open(filepath, 'w') as fh:
    fh.writelines(new_lines)

print(f"  ✓ patched {filepath}")
PYEOF
            mv "$TMPFILE" "$TARGET"
        else
            echo "  · $f already patched"
        fi
    fi
done

# -------------------------------------------------------
# Final tree summary
# -------------------------------------------------------
echo ""
echo "Done. Final layout:"
echo ""
echo "scripts/"
echo "├── ml/"
for f in "${ML_SCRIPTS[@]}"; do
    echo "│   ├── $f"
done
echo "│   └── (future ML scripts go here)"
for f in "${DATA_SCRIPTS[@]}"; do
    echo "├── $f"
done
echo "└── reorganize.sh  (this script)"
echo ""
echo "Quick-start on M2 Pro:"
echo "  cd ~/HFT"
echo "  python scripts/ml/tune_alpha.py data/datasets/clean/daily/nvda_20260105_dataset.bin"
echo "  python scripts/ml/train_nvda.py"