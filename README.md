# D2

[中文](README.md) | [English](README_EN.md)

IPv6 活跃地址的离散扩散（Masked Diffusion）生成实验代码。

本项目用 Transformer Encoder + 离散扩散目标学习 `/64` 前缀条件下的接口标识符（IID）分布，
并按前缀密度分层生成候选地址，用于 IPv6 地址空间测量的实验研究。

## 目录结构

```
.
├── 0_extract_test_prefixes.py    # 按种子数将 /64 前缀分为低 / 中 / 高密度三层
├── 1_clean_aliases.py            # 别名前缀清洗（CIDR 区间合并 + 二分查找）
├── 2_entropy_filter.py           # 香农熵过滤（保留结构化地址，丢弃高熵随机地址）
├── 3_train_fast6diff.py          # 离散扩散模型训练
├── 4_generate_paper_eval.py      # 迭代去噪生成与评估
├── compute_prefix_entropy.py     # 各前缀集 IID 熵统计
└── PROJECT_OUTLINE.md            # 方法与实现说明
```

## 环境

- Python 3.9+
- PyTorch 2.x（CUDA）
- numpy、tqdm

```bash
pip install -r requirements.txt
```

## 数据准备

数据文件不上传至仓库，需自行准备并放入 `data/`：

| 文件名 | 说明 |
|--------|------|
| `responsive-addresses.txt` | 原始活跃 IPv6 地址（每行一个） |
| `aliased-prefixes.txt` | 别名前缀黑名单（CIDR，每行一个） |
| `clean-responsive-addresses.txt` | `1_clean_aliases.py` 输出 |
| `gold-training-addresses.txt` | `2_entropy_filter.py` 输出 |
| `test_prefixes_{low,mid,high}.txt` | `0_extract_test_prefixes.py` 输出（`prefix,seed_count`） |

## 运行

```bash
# 数据准备
python 0_extract_test_prefixes.py
python 1_clean_aliases.py
python 2_entropy_filter.py

# 训练（权重保存到 models/）
python 3_train_fast6diff.py

# 生成评估
python 4_generate_paper_eval.py --prefix_file data/test_prefixes_low.txt --density low \
    --sample_num <N> --k_factor <K>

# 固定总预算模式
python 4_generate_paper_eval.py --prefix_file data/test_prefixes_low.txt --density low \
    --total_budget <TOTAL>
```

`4_generate_paper_eval.py` 默认读取 `models/fast6diff_epoch_20.tar`，如需使用其他
checkpoint，修改脚本中的 `MODEL_CKPT` 即可。

## 说明

- `data/`、`models/`、`outputs/` 因体积原因不纳入版本库。
- 训练与推理的超参数在脚本内配置。
