# Fast6Diff — IPv6 离散扩散扫描生成引擎 项目大纲

## 一、项目目标

利用**离散扩散模型（Discrete Diffusion Model）**学习全球活跃 IPv6 地址的结构化规律（`P(IID | /64 Prefix)`），并按前缀密度分层生成高命中率的候选地址，用于 IPv6 网络扫描与测量研究。

---

## 二、数据规模概览

| 数据集 | 文件名 | 数量 |
|--------|--------|------|
| 原始活跃 IPv6 地址 | `responsive-addresses.txt` | **40,322,856** 条（首行注释不计） |
| 别名前缀黑名单 | `aliased-prefixes.txt` | **24,549,132** 条（首行注释不计） |
| 清洗后干净地址 | `clean-responsive-addresses.txt` | **40,262,613** 条 |
| 最终黄金训练集 | `gold-training-addresses.txt` | **23,892,611** 条 |
| 低密度测试前缀 | `test_prefixes_low.txt` | **16,752** 个（去表头） |
| 中密度测试前缀 | `test_prefixes_mid.txt` | **3,315** 个（去表头） |
| 高密度测试前缀 | `test_prefixes_high.txt` | **766** 个（去表头） |

> 数据流通：4032 万原始地址 → 别名清洗 (-6 万) → 4026 万干净地址 → 熵过滤 (-1637 万高熵随机地址) → **2389 万黄金训练集（占干净地址的 59.3%）**
 - 按熵值剔除高熵的地址，保留低熵3.0以下的地址两千多万作为训练数据
---

## 三、完整 Pipeline（5 个脚本，按编号顺序执行）

### 第 0 步：`0_extract_test_prefixes.py` — 前缀密度分层提取

**做了什么：** 从黄金训练集中扫描所有 `/64` 前缀的地址分布密度，按种子地址数量分为三个密度层级，为后续分层实验做准备。

**技术原理：**

- 使用 `socket.inet_pton()`（底层 C API）将 IPv6 地址转为 16 字节二进制，提取前 8 字节（即 `/64` 前缀）的十六进制表示作为分组键
- 用 `collections.Counter` 统计每个前缀下有多少个活跃地址（种子数）
- 按密度阈值分为三层：

| 密度层级 | 种子数范围 | 含义 | 前缀数量 |
|----------|-----------|------|----------|
| **低密度 (Low)** | 100 ~ 299 | 稀疏前缀，地址分布分散 | **16,752** |
| **中密度 (Mid)** | 300 ~ 999 | 中等密度 | **3,315** |
| **高密度 (High)** | ≥ 1000 | 密集前缀，地址集中 | **766** |

- 输出密度分布直方图（按种子数区间统计前缀数量），便于论文分析
- 结果落盘为 CSV 格式（`prefix,seed_count`），按种子数降序排列

> 低密度前缀占绝大多数（~80%），符合 IPv6 地址空间稀疏分布的预期。

---

### 第 1 步：`1_clean_aliases.py` — 别名前缀清洗

**做了什么：** 剔除被"别名前缀"（Aliased Prefixes）污染的地址——即同一物理主机通过不同 IP 响应、但落在同一别名网段内的情况。这类地址会让模型误以为该网段活跃地址很多，实际却是重复主机。

**技术原理：**

- **CIDR 区间合并算法：** 将 2454 万个别名网段（CIDR 格式，如 `2001:250:345a::/48`）用 `ipaddress.IPv6Network` 转为整数区间 `[start_ip, end_ip]`，排序后合并重叠/相邻区间，极大压缩黑名单体积
- **二分查找加速：** 预提取所有合并后区间的左边界列表 `starts`，对每个待检测 IP 用 `bisect.bisect_right()` 做 O(log N) 查找，避免逐一遍历 2454 万条规则
- 清洗结果：从 40,322,856 条活跃地址中剔除 **60,243** 个别名 IP，保留 **40,262,613** 条干净地址

**输出：** `clean-responsive-addresses.txt`

---

### 第 2 步：`2_entropy_filter.py` — 香农熵过滤

**做了什么：** 用**信息熵（Shannon Entropy）**区分"有规律结构的地址"和"完全随机的地址"，只保留前者用于训练。这是整条 Pipeline 中**唯一不可逆的数据丢弃环节**。

**核心原理 — 香农熵：**

$$H = -\sum_{i} p(x_i) \log_2 p(x_i)$$

- 对每个 IPv6 地址的**后 64 位（IID，接口标识符，16 个十六进制字符）**计算香农熵
- **低熵 (< 1.5)：** 字符高度重复，说明是手工配置/简单规律生成（如 `0000:0000:0000:0001`）
- **中熵 (1.5 ~ 3.0)：** 混合规律，如 SLAAC/EUI-64 等自动配置协议生成
- **高熵 (> 3.0)：** 接近完全随机（最大理论熵 = 4.0），可能是隐私扩展地址或随机生成地址，**无法被模型学习**

**过滤结果：**

| 熵分级 | 数量 | 占比 | 处理方式 |
|--------|------|------|----------|
| Low (< 1.5) | — | — | ✅ 保留 |
| Medium (1.5 ~ 3.0) | — | — | ✅ 保留 |
| High (> 3.0) | ~1637 万 | ~40.7% | ❌ 丢弃 |

最终保留 **23,892,611** 条黄金训练数据（占干净地址的 59.3%）。

**性能优化技巧：**

- **查表法（空间换时间）：** 预计算 1~16 次字符出现概率 × log₂ 的值存入 `ENTROPY_TABLE` 字典，循环中直接查表，消除 `math.log2()` 浮点运算开销
- **底层 C API：** 用 `socket.inet_pton()` + `binascii.hexlify()` 代替 Python `ipaddress` 库，解析速度提升数十倍
- **大缓冲区 I/O：** 单次打开文件流式处理，边读边写

**学术意义：** 用信息论方法定量证明 IPv6 地址空间中有约 **59%** 的活跃地址是结构化的、可被模型学习的。

**输出：** `gold-training-addresses.txt`

---

### 第 3 步：`3_train_fast6diff.py` — 离散扩散模型训练

**做了什么：** 训练一个 Transformer 模型学习 IPv6 地址的**条件概率分布** P(IID | /64 Prefix)，即给定前缀，预测后缀。

**模型架构：**

| 组件 | 规格 |
|------|------|
| 基础架构 | Transformer Encoder（纯编码器，无解码器） |
| 层数 | 6 层 |
| 隐藏维度 | 256 |
| FFN 维度 | 1024（4× 隐藏维度） |
| 注意力头数 | 8 |
| 词表大小 | 17（0~f 十六进制字符 + `[MASK]` token = 16） |
| 序列长度 | 32（完整 IPv6 地址的 hex 表示，含 16 位前缀 + 16 位 IID） |
| 最大时间步 | 16（对应 IID 的全部 16 个位置） |
| 激活函数 | GELU |
| Dropout | 0.1 |
| 参数量 | 约 500 万 |
| 模型文件大小 | ~55 MB / checkpoint |

**核心原理 — 离散扩散（Masked Diffusion）：**

1. **前向扩散（加噪）：** 对每条训练样本，随机选择 `t` 个 IID 位置（t ∈ [1, 16]），将其替换为 `[MASK]` token。t 越大，噪声越多，任务越难。
2. **条件注入：** 将时间步 `t` 通过 `TimeEmbedding` 注入 Transformer，使模型感知当前噪声程度。
3. **反向去噪（训练目标）：** 模型接收部分被 mask 的地址 + 时间步 `t`，预测被 mask 位置的**原始十六进制字符**。
4. **损失函数：** **仅对被 mask 位置**计算 CrossEntropy Loss（非 mask 位置不参与 loss）。

这与 BERT 的 MLM（Masked Language Modeling）预训练思路一致，但增加了**时间步嵌入**使模型学会在不同噪声程度下去噪，是离散扩散模型的核心。

**关键技术细节：**

- **内存优化：** 2389 万条数据用 `uint8` 存储，仅占约 **760 MB**（float32 则需要约 3 GB）
- **混合精度训练：** PyTorch AMP + `bfloat16`，在 RTX 4090 上高效运行
- **大 Batch Size：** 8192，配合平方根法则调整学习率（LR = 4e-4）
- **`torch.compile()`** 编译模型计算图，融合算子加速训练
- **断点续训：** 每轮保存完整 checkpoint（模型权重 + optimizer 状态），支持从中断处恢复
- **梯度裁剪：** `max_norm=1.0` 防止梯度爆炸
- **DataLoader 优化：** `num_workers=8`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`
- **训练周期：** 20 个 Epoch

---

### 第 4 步：`4_generate_paper_eval.py` — 推理生成与论文评估

**做了什么：** 用训练好的模型，对每个测试 `/64` 前缀生成候选 IID，组合成完整 IPv6 地址，用于命中率评估。

**核心原理 — 迭代去噪生成（Iterative Unmasking）：**

1. **初始化：** 将 /64 前缀的 hex tokens（前 16 位）保留不变，IID 部分（后 16 位）全部设为 `[MASK]` token
2. **迭代去噪（T_infer 步）：**
   - 每步模型对所有 mask 位置输出 logits
   - Softmax + Temperature 缩放后，`torch.multinomial()` 按概率采样
   - **Confidence-based Remasking（低置信度回退）：** 每步保留置信度最低的 `ratio` 比例位置继续 mask，高置信度的则"锁定"
   - `ratio` 随步数线性递减：`(T_infer - 1 - step) / T_infer`
   - 前缀位置始终锁定，永不参与 mask
3. **Temperature 控制：** 低温 → 概率分布更尖锐 → 生成确定性高（保守）；高温 → 分布更平坦 → 多样性高（冒险）
4. **去重：** 同一前缀的生成结果用 `set()` 去重

**密度自适应推理：**

- 每个密度层级使用独立的推理参数（迭代步数与采样温度），由脚本内配置决定
- **停滞容忍：** 连续多少轮产出低于阈值则放弃该前缀（早停），防止在"难产"前缀上浪费算力
- **产出阈值：** 每 batch 新增候选数低于此值视为"停滞"

**预算控制策略：**

- **K 因子模式（`--k_factor`）：** 每个前缀按 `种子数 × K` 作为生成预算
- **Total Budget 模式（`--total_budget`）：** 固定总预算，按各前缀种子数比例自动分配，用于公平横向对比
- 采样数量按密度分层抽样（`--sample_num`）

**输出：** `outputs/paper_experiments/{density}/candidates_T{t}_Temp{temp}_K{k}.txt`

---

### 第 5 步：超参数调优（实验细节从略）

项目在低 / 中 / 高三个密度层级上分别进行了推理步数 T 与采样温度 Temperature 的搜索，
并在固定预算下与基线方法做了横向对比。具体网格范围、各区域最优参数组合、
K 因子与总预算取值不在本仓库公开。

---

## 四、核心技术体系总结

```
┌─────────────────────────────────────────────────────────┐
│                   Fast6Diff 技术栈                        │
├─────────────────┬───────────────────────────────────────┤
│ 数据层          │ IPv6 地址解析 (socket C API)            │
│                 │ CIDR 区间合并 + 二分查找过滤            │
│                 │ 香农熵 结构化/随机地址分类               │
│                 │ 按前缀密度三层分层采样                   │
├─────────────────┼───────────────────────────────────────┤
│ 模型层          │ Transformer Encoder × 6               │
│                 │ 离散扩散 (Masked Diffusion)             │
│                 │ Time Embedding (时间步条件注入)         │
│                 │ Token Embedding + Position Embedding   │
├─────────────────┼───────────────────────────────────────┤
│ 训练层          │ AMP bfloat16 混合精度                   │
│                 │ uint8 内存压缩 (760MB 存 2389 万条)     │
│                 │ torch.compile 图编译加速                │
│                 │ 断点续训 (Checkpoint + Optimizer State) │
│                 │ AdamW + Weight Decay + Gradient Clip   │
├─────────────────┼───────────────────────────────────────┤
│ 推理层          │ 迭代去噪 + Temperature 采样             │
│                 │ Confidence-based Remasking (低置信回退) │
│                 │ 密度自适应超参数                         │
│                 │ 动态预算分配 (K-Factor / Total Budget)  │
│                 │ 批量生成 + 去重 + 停滞早停               │
└─────────────────┴───────────────────────────────────────┘
```

---

## 五、项目创新点

1. **首次将离散扩散模型应用于 IPv6 地址生成：** 传统方法（如 6GAN、6Graph、6VecLM 等）多用 GAN 或 n-gram 语言模型。本项目使用 Masked Diffusion + Transformer，训练更稳定（无需对抗训练的 mode collapse 问题），且生成质量更高。

2. **香农熵驱动的数据筛选：** 用信息论方法定量证明 IPv6 地址空间中约 59% 的活跃地址是结构化的、可学习的，系统性过滤掉不可学习的随机噪声（隐私扩展地址等）。

3. **密度分层评估框架：** 按前缀密度（低/中/高）分别调优超参数和评估，更真实地反映模型在不同稀疏程度场景下的表现，而非用一个全局参数掩盖差异性。

4. **Confidence-based Remasking：** 推理时不一次性去噪，而是边去噪边把低置信度预测重新 mask，在后续步中利用更多上下文信息重新预测——类似"反复推敲"的迭代精炼过程。

5. **工程级全链路优化：** 从数据加载（uint8 压缩、C API 解析、查表法）、训练（AMP、`torch.compile`、persistent workers、梯度裁剪）、到推理（8192 批量生成、去重集合、停滞早停），全链路极致性能优化。

---

## 六、文件目录结构

```
IPv6_Diffusion_Project/code/
│
├── 0_extract_test_prefixes.py    # 前缀密度分层提取（按种子数分低/中/高）
├── 1_clean_aliases.py            # 别名前缀清洗（CIDR 区间合并 + 二分查找）
├── 2_entropy_filter.py           # 香农熵过滤（保留结构化地址，丢弃随机地址）
├── 3_train_fast6diff.py          # 离散扩散模型训练（Transformer + Masked Diffusion）
├── 4_generate_paper_eval.py      # 推理生成与论文评估（迭代去噪 + 预算控制）
│
├── data/
│   ├── responsive-addresses.txt          # 原始活跃地址 (~4032 万条，含注释头)
│   ├── aliased-prefixes.txt              # 别名黑名单 (~2454 万条，含注释头)
│   ├── clean-responsive-addresses.txt    # 清洗后地址 (~4026 万条)
│   ├── gold-training-addresses.txt       # 黄金训练集 (~2389 万条)
│   ├── test_prefixes_low.txt             # 低密度测试前缀 (16,752 个)
│   ├── test_prefixes_mid.txt             # 中密度测试前缀 (3,315 个)
│   └── test_prefixes_high.txt            # 高密度测试前缀 (766 个)
│
├── models/
│   └── fast6diff_epoch_1~20.tar          # 模型权重 checkpoint (×20, 各 ~55 MB)
│
├── outputs/
│   ├── paper_experiments/                # 论文实验输出（K 因子模式）
│   │   ├── low/
│   │   ├── mid/
│   │   └── high/
│   └── 固定预算/                          # 固定预算对比实验（Total Budget 模式）
│
└── PROJECT_OUTLINE.md                    # 本文件
```

---

## 七、典型命令行用法

```bash
# ---- 数据准备阶段 ----
python 0_extract_test_prefixes.py
python 1_clean_aliases.py
python 2_entropy_filter.py

# ---- 模型训练 ----
python 3_train_fast6diff.py

# ---- 推理评估阶段 ----

# K 因子模式（每个前缀生成 seed_count × K 个候选）
python 4_generate_paper_eval.py --prefix_file data/test_prefixes_low.txt  --density low  --sample_num <N> --k_factor <K>
python 4_generate_paper_eval.py --prefix_file data/test_prefixes_mid.txt  --density mid  --sample_num <N> --k_factor <K>
python 4_generate_paper_eval.py --prefix_file data/test_prefixes_high.txt --density high --sample_num <N> --k_factor <K>

# Total Budget 模式（固定总预算，按种子数比例自动分配）
python 4_generate_paper_eval.py --prefix_file data/test_prefixes_low.txt --density low --total_budget <TOTAL>
