import os
import glob
import re
import socket
import binascii

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from common import cfg, require

# ==========================================
# [配置区] 具体取值见 config.py
# ==========================================
TRAIN_DATA_FILE = os.path.join(cfg.DATA_DIR, "gold-training-addresses.txt")
MODEL_SAVE_DIR = cfg.MODEL_SAVE_DIR
DATA_LIMIT = cfg.DATA_LIMIT

BATCH_SIZE = require("BATCH_SIZE", cfg.BATCH_SIZE)
EPOCHS = require("EPOCHS", cfg.EPOCHS)
LEARNING_RATE = require("LEARNING_RATE", cfg.LEARNING_RATE)
WEIGHT_DECAY = require("WEIGHT_DECAY", cfg.WEIGHT_DECAY)
GRAD_CLIP = require("GRAD_CLIP", cfg.GRAD_CLIP)
NUM_WORKERS = require("NUM_WORKERS", cfg.NUM_WORKERS)
PREFETCH_FACTOR = require("PREFETCH_FACTOR", cfg.PREFETCH_FACTOR)

HIDDEN_DIM = require("HIDDEN_DIM", cfg.HIDDEN_DIM)
NUM_LAYERS = require("NUM_LAYERS", cfg.NUM_LAYERS)
NUM_HEADS = require("NUM_HEADS", cfg.NUM_HEADS)
DROPOUT_RATE = require("DROPOUT_RATE", cfg.DROPOUT_RATE)
FFN_MULTIPLIER = require("FFN_MULTIPLIER", cfg.FFN_MULTIPLIER)

# 由 IPv6 地址格式决定：IID 固定为 16 个十六进制字符
IID_HEX_LEN = 16
MAX_TIMESTEP = IID_HEX_LEN
MAX_SEQ_LEN = 2 * IID_HEX_LEN
VOCAB_SIZE = 17
MASK_TOKEN = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg.SEED is not None:
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)


# ==========================================
# [Dataset & Model]
# ==========================================
class Fast6DiffDataset(Dataset):
    def __init__(self, data_file, limit_size=None):
        print("[*] 正在加载黄金训练集到内存 (Python List 解析阶段会有内存峰值，请稍候)...")
        self.data = []
        count = 0
        with open(data_file, 'r', encoding='ascii') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                packed = socket.inet_pton(socket.AF_INET6, line)
                hex_str = binascii.hexlify(packed).decode('ascii')
                tokens = [int(char, 16) for char in hex_str]
                self.data.append(tokens)
                count += 1
                if limit_size and count >= limit_size:
                    break

        print("[*] 数据读取完毕，正在转换为常驻内存的 PyTorch Tensor (uint8 省流模式)...")
        temp_array = np.array(self.data, dtype=np.uint8)
        # 保持 uint8 常驻内存，不提前转 .long()
        self.data = torch.from_numpy(temp_array)

        if limit_size:
            print(f"[*] 🚦 测试模式激活：已截取前 {len(self.data)} 条数据进行训练！")
        else:
            print(f"[*] ✅ 成功加载全量 {len(self.data)} 条黄金数据！内存压缩完毕。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 在被 DataLoader 抓取时，再转为模型需要的 .long()
        tokens = sample.clone().long()
        labels = sample.clone().long()

        t = torch.randint(1, MAX_TIMESTEP + 1, (1,)).item()
        mask_indices = torch.randperm(IID_HEX_LEN)[:t] + IID_HEX_LEN

        mask_indicator = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        mask_indicator[mask_indices] = True
        tokens[mask_indicator] = MASK_TOKEN

        return tokens, labels, mask_indicator, torch.tensor(t, dtype=torch.long)


class Fast6DiffModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, HIDDEN_DIM)
        self.time_emb = nn.Embedding(MAX_TIMESTEP + 1, HIDDEN_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * FFN_MULTIPLIER,
            dropout=DROPOUT_RATE, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.output_head = nn.Linear(HIDDEN_DIM, 16)

    def forward(self, tokens, t):
        batch_size, seq_len = tokens.size()
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = x + self.time_emb(t).unsqueeze(1)
        return self.output_head(self.transformer(x))


# ==========================================
# [辅助函数] 智能提取模型参数 (防 compile 污染)
# ==========================================
def get_clean_state_dict(model):
    """剥离 torch.compile 带来的 _orig_mod 前缀，保证推理脚本 100% 能加载"""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


# ==========================================
# 离散扩散训练循环
# ==========================================
def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("=" * 50)
    print("[*] 环境自检报告:")
    print(f"    - PyTorch 版本 : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"    - GPU 显卡型号 : {torch.cuda.get_device_name(0)}")
        print(f"    - 显卡总显存   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("    - 警告: 未检测到 GPU，将使用 CPU 训练！")
    print("=" * 50)

    dataset = Fast6DiffDataset(TRAIN_DATA_FILE, limit_size=DATA_LIMIT)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
    )

    model = Fast6DiffModel().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[*] 模型架构初始化完毕！Trainable Params: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    existing_checkpoints = glob.glob(os.path.join(MODEL_SAVE_DIR, "fast6diff_epoch_*.tar"))

    # 严格遵循 [先加载权重 -> 再 Compile] 的顺序
    if existing_checkpoints:
        latest_ckpt_path = max(
            existing_checkpoints,
            key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1])
        )
        latest_epoch = int(re.findall(r'\d+', os.path.basename(latest_ckpt_path))[-1])

        print(f"\n[!] 检测到历史训练记录: {latest_ckpt_path}")
        user_input = input(f"[?] 是否要从 Epoch {latest_epoch + 1} 继续训练？(Y/n): ").strip().lower()

        if user_input != 'n':
            print("[*] 正在恢复完整的模型权重与优化器状态...")
            checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[*] 成功恢复！将从 Epoch {start_epoch} 开始训练。")
        else:
            print("[*] 放弃历史权重，从头开始训练。")

    # 加载完权重后，再执行 compile()，防止 Key 冲突
    if getattr(cfg, "TORCH_COMPILE", True):
        print("[*] 正在利用 PyTorch 2.x 编译模型内核，首个 Epoch 前几步会稍慢，请耐心等待...")
        model = torch.compile(model)

    print(f"\n[*] 训练配置 -> 设备: {DEVICE} | 训练集: {len(dataset)} | 预计 Epochs: {start_epoch} -> {EPOCHS}")

    if start_epoch > EPOCHS:
        print("\n[!] 设定的总 Epochs 已经跑完啦！")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps_done = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

        try:
            for tokens, labels, mask_indicator, t in pbar:
                tokens = tokens.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                mask_indicator = mask_indicator.to(DEVICE, non_blocking=True)
                t = t.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    logits = model(tokens, t)
                    masked_logits = logits[mask_indicator]
                    masked_labels = labels[mask_indicator]
                    loss = criterion(masked_logits, masked_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()

                total_loss += loss.item()
                steps_done += 1
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        except KeyboardInterrupt:
            print(f"\n\n[!] 训练被手动中断！正在紧急保存当前 Epoch {epoch} 临时状态...")
            checkpoint_status = {
                'epoch': epoch - 1,
                'model_state_dict': get_clean_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint_status, os.path.join(MODEL_SAVE_DIR, f"fast6diff_epoch_{epoch}_interrupted.tar"))
            print("[*] 紧急保存成功。安全退出。")
            return

        avg_loss = total_loss / steps_done
        print(f"[+] Epoch {epoch} 顺利完成, 平均 Loss: {avg_loss:.4f}")

        checkpoint_status = {
            'epoch': epoch,
            'model_state_dict': get_clean_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint_status, os.path.join(MODEL_SAVE_DIR, f"fast6diff_epoch_{epoch}.tar"))

    print("\n✅ 训练完成！")


if __name__ == "__main__":
    train()
