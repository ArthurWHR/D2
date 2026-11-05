import os
import torch
import torch.nn as nn
import socket
import argparse
import random
from tqdm import tqdm

# ==========================================
# [修复环境变量] 防止底层多线程冲突导致生成降速
# ==========================================
os.environ["OMP_NUM_THREADS"] = "8"

# ==========================================
# [命令行参数解析]
# ==========================================
parser = argparse.ArgumentParser(description="Fast6Diff 离散扩散极速生成引擎")
parser.add_argument("--prefix_file", type=str, required=True, help="输入的前缀文件")
parser.add_argument("--density", type=str, choices=["low", "mid", "high"], required=True, help="密度类型")
parser.add_argument("--k_factor", type=float, default=2.0, help="生成预算的倍数 K")
parser.add_argument("--total_budget", type=int, default=None, help="总生成预算，自动按 seed_count 比例分配到各前缀（指定后覆盖 --k_factor）")
parser.add_argument("--sample_num", type=int, default=500, help="随机抽取多少个前缀")
args = parser.parse_args()

MODEL_CKPT = "models/fast6diff_epoch_20.tar" # 务必确保权重文件名正确
OUTPUT_DIR = os.path.join("outputs/paper_experiments", args.density)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 基于密度的智适应参数矩阵
# ==========================================

T_INFER_LIST = [16] 

if args.density == "low":
    TEMPERATURE_LIST = [0.4]
    MAX_STAGNANT_BATCHES = 20          
    YIELD_THRESHOLD = 2                 
elif args.density == "mid":
    TEMPERATURE_LIST = [0.6]
    MAX_STAGNANT_BATCHES = 15           
    YIELD_THRESHOLD = 5                 
elif args.density == "high":
    TEMPERATURE_LIST = [0.4]
    MAX_STAGNANT_BATCHES = 10           
    YIELD_THRESHOLD = 20                

BATCH_SIZE = 8192                      

# 模型架构超参数 
HIDDEN_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8
VOCAB_SIZE = 17
MASK_TOKEN = 16
MAX_SEQ_LEN = 32
MAX_TIMESTEP = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# [模型定义与工具函数] (保持不变)
# ==========================================
class Fast6DiffModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, HIDDEN_DIM)
        self.time_emb = nn.Embedding(MAX_TIMESTEP + 1, HIDDEN_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS, dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.0, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.output_head = nn.Linear(HIDDEN_DIM, 16)

    def forward(self, tokens, t):
        batch_size, seq_len = tokens.size()
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_emb(tokens) + self.pos_emb(positions) + self.time_emb(t).unsqueeze(1)
        return self.output_head(self.transformer(x))

def tokens_to_ipv6(tokens_list):
    hex_chars = "0123456789abcdef"
    hex_str = "".join([hex_chars[t] for t in tokens_list])
    packed = bytes.fromhex(hex_str)
    return socket.inet_ntop(socket.AF_INET6, packed)

@torch.no_grad()
def generate_batch(model, prefixes, T_infer, temp):
    batch_size = prefixes.size(0)
    prefix_mask = torch.zeros((batch_size, 32), dtype=torch.bool, device=DEVICE)
    prefix_mask[:, :16] = True 
    
    masked_iid = torch.full((batch_size, 16), MASK_TOKEN, dtype=torch.long, device=DEVICE)
    tokens = torch.cat([prefixes, masked_iid], dim=1) 
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for step in range(T_infer):
            is_mask = (tokens == MASK_TOKEN)
            t = is_mask.sum(dim=1) 
            if t.max() == 0: break
                
            logits = model(tokens, t) 
            probs = torch.softmax(logits.float() / temp, dim=-1)
            
            sampled_preds = torch.multinomial(probs.view(-1, 16), 1).view(batch_size, 32)
            confidence = probs.gather(2, sampled_preds.unsqueeze(2)).squeeze(2) 
            
            confidence[~is_mask] = 1e9
            confidence[prefix_mask] = 1e9
            tokens = torch.where(is_mask, sampled_preds, tokens)
            
            ratio = (T_infer - 1 - step) / T_infer
            num_keep_masked = int(16 * ratio)
            
            if num_keep_masked > 0:
                _, indices_to_mask = torch.topk(confidence, num_keep_masked, dim=1, largest=False)
                mask_recovery = torch.zeros((batch_size, 32), dtype=torch.bool, device=DEVICE)
                mask_recovery.scatter_(1, indices_to_mask, True)
                mask_recovery[prefix_mask] = False
                tokens[mask_recovery] = MASK_TOKEN

    return tokens

# ==========================================
# [主控流程]
# ==========================================
def main():
    print(f"[*] 初始化引擎 | 模式: {args.density.upper()} | 前缀文件: {args.prefix_file}")
    
    model = Fast6DiffModel().to(DEVICE)
    checkpoint = torch.load(MODEL_CKPT, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    print("[*] 正在利用 torch.compile 编译推理图，榨干算力...")
    model = torch.compile(model)

    all_available_prefixes = []
    with open(args.prefix_file, 'r', encoding='ascii') as f:
        lines = f.readlines()[1:] # 跳过 CSV 标头 "prefix,seed_count"
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 🌟 [核心修改 2] 解析时同时剥离出种子数 (Count)
            parts = line.split(",")
            prefix_str = parts[0].strip()
            seed_count = int(parts[1].strip())
            
            if len(prefix_str) == 16:
                prefix_tokens = [int(c, 16) for c in prefix_str]
                # 把前缀特征和对应的真实种子数绑成元组存起来
                all_available_prefixes.append((prefix_tokens, seed_count))
                
    random.seed(42) 
    sample_size = min(args.sample_num, len(all_available_prefixes))
    sampled_prefixes = random.sample(all_available_prefixes, sample_size)
    
    # 🌟 如果指定了 total_budget，自动计算等效 k_factor，后续逻辑完全不变
    use_total_budget = args.total_budget is not None
    if use_total_budget:
        total_seeds = sum(count for _, count in sampled_prefixes)
        args.k_factor = args.total_budget / total_seeds  # 覆盖 k_factor
        print(f"[*] 抽样完成: {sample_size} 个前缀。总预算 = {args.total_budget:,}，等效 K = {args.k_factor:.4f}")
    else:
        print(f"[*] 抽样完成: {sample_size} 个前缀。使用的预算放大系数 K = {args.k_factor}")

    # 仅提取 prefix_tokens 给 tensor，seed_count 留着计算用
    all_prefixes_tensor = torch.tensor([p[0] for p in sampled_prefixes], dtype=torch.long, device=DEVICE)

    for t_infer in T_INFER_LIST:
        for temp in TEMPERATURE_LIST:
            print(f"\n" + "="*60)
            print(f"[🚀 正在生成] Density={args.density.upper()} | T={t_infer} | Temp={temp} | K={args.k_factor}")
            
            # 🌟 [核心修改 3] 文件名体现出 K 因子
            budget_label = f"Budget{args.total_budget}" if use_total_budget else f"K{args.k_factor}"
            output_file = os.path.join(OUTPUT_DIR, f"candidates_T{t_infer}_Temp{temp}_{budget_label}.txt")
            all_candidates_count = 0
            
            # 🌟 [核心修改 4] 动态计算整个进度条的目标总量
            total_target_for_all = sum(int(count * args.k_factor) for _, count in sampled_prefixes)
            pbar = tqdm(total=total_target_for_all, desc="总进度")
            
            with open(output_file, "w", encoding="ascii") as f_out:
                for idx in range(sample_size):
                    # 🌟 [核心修改 5] 获取该前缀专属的 target_num
                    _, seed_count = sampled_prefixes[idx]
                    target_num = int(seed_count * args.k_factor)
                    
                    # 极端防呆机制：防止部分极端前缀 K 倍后数量仍太小（通常大于 100，所以不会出现此情况，但防一手）
                    target_num = max(10, target_num) 
                    
                    prefix_candidates = set()
                    stagnant_batches = 0
                    
                    while len(prefix_candidates) < target_num:
                        remain = target_num - len(prefix_candidates)
                        current_batch = min(BATCH_SIZE, max(1024, remain * 2))
                        
                        prefix_batch = all_prefixes_tensor[idx].unsqueeze(0).expand(current_batch, -1)
                        
                        final_tokens = generate_batch(model, prefix_batch, t_infer, temp)
                        old_size = len(prefix_candidates)
                        unique_tokens = torch.unique(final_tokens, dim=0)
                        
                        for row in unique_tokens.cpu().numpy():
                            ip_str = tokens_to_ipv6(row)
                            if ip_str not in prefix_candidates:
                                prefix_candidates.add(ip_str)
                                if len(prefix_candidates) >= target_num:
                                    break
                                    
                        added = len(prefix_candidates) - old_size
                        
                        if added < YIELD_THRESHOLD:
                            stagnant_batches += 1
                        else:
                            stagnant_batches = 0
                            
                        if added > 0:
                            all_candidates_count += added
                            pbar.update(added)
                                    
                        if stagnant_batches >= MAX_STAGNANT_BATCHES:  
                            leftover = target_num - len(prefix_candidates)
                            pbar.update(leftover) 
                            break
                            
                    if prefix_candidates:
                        f_out.write("\n".join(prefix_candidates) + "\n")

            pbar.close()
            print(f"[*] ✅ 落盘至: {output_file} | 实际产出总数: {all_candidates_count}")

if __name__ == "__main__":
    main()

