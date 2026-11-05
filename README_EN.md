# D2

[中文](README.md) | [English](README_EN.md)

Discrete diffusion (masked diffusion) code for generating active IPv6 addresses.

A Transformer encoder is trained with a masked-diffusion objective to model the
interface identifier (IID) distribution conditioned on a `/64` prefix. Candidate
addresses are then generated per prefix, stratified by prefix density, for IPv6
address-space measurement experiments.

## Layout

```
.
├── 0_extract_test_prefixes.py    # Split /64 prefixes into low / mid / high density buckets
├── 1_clean_aliases.py            # Alias-prefix cleaning (CIDR interval merge + binary search)
├── 2_entropy_filter.py           # Shannon-entropy filter (keep structured, drop high-entropy random)
├── 3_train_fast6diff.py          # Discrete diffusion model training
├── 4_generate_paper_eval.py      # Iterative denoising generation and evaluation
├── compute_prefix_entropy.py     # IID entropy statistics per prefix set
└── PROJECT_OUTLINE.md            # Method and implementation notes (Chinese)
```

## Requirements

- Python 3.9+
- PyTorch 2.x (CUDA)
- numpy, tqdm

```bash
pip install -r requirements.txt
```

## Data

Data files are not included in this repository. Prepare them under `data/`:

| File | Description |
|------|-------------|
| `responsive-addresses.txt` | Raw responsive IPv6 addresses (one per line) |
| `aliased-prefixes.txt` | Alias-prefix blacklist (CIDR, one per line) |
| `clean-responsive-addresses.txt` | Output of `1_clean_aliases.py` |
| `gold-training-addresses.txt` | Output of `2_entropy_filter.py` |
| `test_prefixes_{low,mid,high}.txt` | Output of `0_extract_test_prefixes.py` (`prefix,seed_count`) |

## Usage

```bash
# Data preparation
python 0_extract_test_prefixes.py
python 1_clean_aliases.py
python 2_entropy_filter.py

# Training (checkpoints are written to models/)
python 3_train_fast6diff.py

# Generation / evaluation
python 4_generate_paper_eval.py --prefix_file data/test_prefixes_low.txt --density low \
    --sample_num <N> --k_factor <K>

# Fixed total budget
python 4_generate_paper_eval.py --prefix_file data/test_prefixes_low.txt --density low \
    --total_budget <TOTAL>
```

`4_generate_paper_eval.py` loads `models/fast6diff_epoch_20.tar` by default; change
`MODEL_CKPT` in the script to use another checkpoint.

## Notes

- `data/`, `models/`, and `outputs/` are excluded from version control due to size.
- Training and inference hyperparameters are configured inside the scripts.
