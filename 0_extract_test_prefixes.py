import os
import socket
import binascii
from collections import Counter

from tqdm import tqdm

from common import cfg, require

# ==========================================
# [Config] 路径与密度分界（取值见 config.py）
# ==========================================
GOLD_TRAIN_FILE = os.path.join(cfg.DATA_DIR, "gold-training-addresses.txt")
OUT_DIR = cfg.DATA_DIR

MIN_SEEDS = require("MIN_SEEDS", cfg.MIN_SEEDS)
LOW_MID_BOUND = require("LOW_MID_BOUND", cfg.LOW_MID_BOUND)
MID_HIGH_BOUND = require("MID_HIGH_BOUND", cfg.MID_HIGH_BOUND)

if not (MIN_SEEDS < LOW_MID_BOUND < MID_HIGH_BOUND):
    raise SystemExit(
        "config.py: density bounds must satisfy "
        "MIN_SEEDS < LOW_MID_BOUND < MID_HIGH_BOUND"
    )


def extract_prefixes():
    print("[*] 正在流式扫描黄金数据集，提取 /64 前缀分布...")
    prefix_counter = Counter()

    with open(GOLD_TRAIN_FILE, "r", encoding="ascii") as f:
        for line in tqdm(f, desc="极速解析 Prefix"):
            line = line.strip()
            if not line:
                continue
            try:
                packed = socket.inet_pton(socket.AF_INET6, line)
                # 安全提取前 8 字节并转为十六进制字符串
                prefix_hex = binascii.hexlify(packed[:8]).decode("ascii")
                prefix_counter[prefix_hex] += 1
            except Exception:
                continue

    print(f"\n[*] 全网共发现 {len(prefix_counter)} 个独立的 /64 前缀。")

    # 密度分布直方图（分界点由 config.py 提供）
    all_counts = list(prefix_counter.values())
    ranges = [
        (1, 10),
        (10, MIN_SEEDS),
        (MIN_SEEDS, LOW_MID_BOUND),
        (LOW_MID_BOUND, MID_HIGH_BOUND),
        (MID_HIGH_BOUND, float("inf")),
    ]

    print("\n[*] Density Histogram")

    for low, high in ranges:
        if high == float("inf"):
            cnt = sum(1 for x in all_counts if x >= low)
            print(f"{low:5d}+      : {cnt}")
        else:
            cnt = sum(1 for x in all_counts if low <= x < high)
            print(f"{low:5d}-{int(high) - 1:5d} : {cnt}")

    # 准备三个全量分层桶（全盘保留）
    bucket_low = []
    bucket_mid = []
    bucket_high = []

    for prefix, count in prefix_counter.items():
        if MIN_SEEDS <= count < LOW_MID_BOUND:
            bucket_low.append((prefix, count))
        elif LOW_MID_BOUND <= count < MID_HIGH_BOUND:
            bucket_mid.append((prefix, count))
        elif count >= MID_HIGH_BOUND:
            bucket_high.append((prefix, count))

    print("[*] 【全量分桶统计】")
    print(f"    - low  ({MIN_SEEDS}~{LOW_MID_BOUND - 1}) : {len(bucket_low)} 个可用前缀")
    print(f"    - mid  ({LOW_MID_BOUND}~{MID_HIGH_BOUND - 1}): {len(bucket_mid)} 个可用前缀")
    print(f"    - high ({MID_HIGH_BOUND}+)   : {len(bucket_high)} 个可用前缀")

    os.makedirs(OUT_DIR, exist_ok=True)

    def save_bucket(data, filename):
        filepath = os.path.join(OUT_DIR, filename)
        with open(filepath, "w", encoding="ascii") as f:
            f.write("prefix,seed_count\n")
            for prefix, count in sorted(data, key=lambda x: x[1], reverse=True):
                f.write(f"{prefix},{count}\n")
        print(f"✅ 成功归档 {len(data)} 个前缀至 -> {filepath}")

    print("\n[*] 开始落盘分层测试集：")
    save_bucket(bucket_low, "test_prefixes_low.txt")
    save_bucket(bucket_mid, "test_prefixes_mid.txt")
    save_bucket(bucket_high, "test_prefixes_high.txt")


if __name__ == "__main__":
    extract_prefixes()
