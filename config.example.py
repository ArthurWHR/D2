# =============================================================================
# D2 configuration template / 配置模板
#
# Copy this file to `config.py` and fill in every value marked None.
# `config.py` is ignored by git, so your settings stay local.
#
# 复制本文件为 `config.py` 并填写所有 None 项。`config.py` 已被 git 忽略，
# 你的实际取值不会进入仓库。
# =============================================================================

# ---- Paths / 路径 -----------------------------------------------------------
DATA_DIR = "data"
MODEL_SAVE_DIR = "models"
OUTPUT_DIR = "outputs"

# ---- Density split: seed-address count per /64 prefix ------------------------
# 密度分层：每个 /64 前缀下的种子地址数
MIN_SEEDS = None        # lower bound of the low-density bucket / 低密度下界
LOW_MID_BOUND = None    # low / mid boundary / 低-中分界
MID_HIGH_BOUND = None   # mid / high boundary / 中-高分界

# ---- Entropy filter (bits per IID) / 熵过滤阈值 -----------------------------
ENTROPY_LOW = None      # below: highly structured / 低于此值：高度结构化
ENTROPY_HIGH = None     # above: discarded as random / 高于此值：视为随机丢弃
ENTROPY_BIN_WIDTH = None  # histogram bin width for reporting / 统计直方图组距

# ---- Model / 模型结构 -------------------------------------------------------
HIDDEN_DIM = None
NUM_LAYERS = None
NUM_HEADS = None
DROPOUT_RATE = None
FFN_MULTIPLIER = None   # feed-forward width = HIDDEN_DIM * FFN_MULTIPLIER

# ---- Training / 训练 --------------------------------------------------------
BATCH_SIZE = None
EPOCHS = None
LEARNING_RATE = None
WEIGHT_DECAY = None
GRAD_CLIP = None
NUM_WORKERS = None
PREFETCH_FACTOR = None
SEED = None
DATA_LIMIT = None       # None = use the full training set / None 表示全量
TORCH_COMPILE = True    # set False if torch.compile is unavailable / 不可用时设为 False

# ---- Inference / 推理 -------------------------------------------------------
T_INFER = None          # list of denoising step counts / 推理步数列表
TEMPERATURE = {"low": None, "mid": None, "high": None}
MAX_STAGNANT_BATCHES = {"low": None, "mid": None, "high": None}
YIELD_THRESHOLD = {"low": None, "mid": None, "high": None}
INFER_BATCH_SIZE = None
INFER_MIN_BATCH = None
MIN_TARGET_PER_PREFIX = None
MODEL_CKPT = None       # e.g. "models/<your_checkpoint>.tar"
