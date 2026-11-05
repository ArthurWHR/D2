import os
import socket
import binascii
import math

from common import cfg, require

BASE = cfg.DATA_DIR
GOLD = os.path.join(BASE, "gold-training-addresses.txt")
PREFIX_FILES = {
    "low": os.path.join(BASE, "test_prefixes_low.txt"),
    "mid": os.path.join(BASE, "test_prefixes_mid.txt"),
    "high": os.path.join(BASE, "test_prefixes_high.txt"),
}

ENTROPY_LOW = require("ENTROPY_LOW", cfg.ENTROPY_LOW)
ENTROPY_HIGH = require("ENTROPY_HIGH", cfg.ENTROPY_HIGH)
BIN_W = require("ENTROPY_BIN_WIDTH", cfg.ENTROPY_BIN_WIDTH)

# IID 固定 16 个十六进制字符，香农熵查表（与 2_entropy_filter.py 一致）
ENTROPY_TABLE = {
    count: (count / 16.0) * math.log2(count / 16.0)
    for count in range(1, 17)
}


def iid_entropy(packed):
    iid = binascii.hexlify(packed)[16:]  # 后 64 位 = 16 个 hex 字符
    ent = 0.0
    for b in set(iid):
        ent -= ENTROPY_TABLE[iid.count(b)]
    return ent


# ---- 1. 载入三个前缀集 -> {prefix_hex: label} ----
prefix2label = {}
total_prefixes = 0
for label, path in PREFIX_FILES.items():
    with open(path, "r", encoding="ascii") as f:
        next(f)  # 跳过 header
        for line in f:
            line = line.strip()
            if not line:
                continue
            prefix_hex = line.split(",")[0].strip()
            prefix2label[prefix_hex] = label
            total_prefixes += 1
print(f"[*] 载入前缀: low/mid/high 共 {total_prefixes} 个 /64 前缀")

# ---- 2. 流式统计 ----
# 每 label 累加 sum / count / min / max / 平方和 + 熵直方图用于中位数与分布
stats = {
    "low":  {"n": 0, "s": 0.0, "s2": 0.0, "mn": 4.0, "mx": 0.0},
    "mid":  {"n": 0, "s": 0.0, "s2": 0.0, "mn": 4.0, "mx": 0.0},
    "high": {"n": 0, "s": 0.0, "s2": 0.0, "mn": 4.0, "mx": 0.0},
    "unmatched": {"n": 0, "s": 0.0, "s2": 0.0, "mn": 4.0, "mx": 0.0},
}

NBINS = int(4.0 / BIN_W) + 2
hist = {k: [0] * NBINS for k in ["low", "mid", "high", "unmatched"]}


def hist_bin(e):
    return min(int(e / BIN_W), NBINS - 1)


n_total = 0
with open(GOLD, "r", encoding="ascii") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            packed = socket.inet_pton(socket.AF_INET6, line)
        except Exception:
            continue
        prefix_hex = binascii.hexlify(packed[:8]).decode("ascii")
        label = prefix2label.get(prefix_hex, "unmatched")
        e = iid_entropy(packed)
        st = stats[label]
        st["n"] += 1
        st["s"] += e
        st["s2"] += e * e
        if e < st["mn"]:
            st["mn"] = e
        if e > st["mx"]:
            st["mx"] = e
        hist[label][hist_bin(e)] += 1
        n_total += 1

print(f"[*] 有效地址总数: {n_total}")
print()


def median_from_hist(h):
    total = sum(h)
    if total == 0:
        return float("nan")
    half = total / 2.0
    cum = 0
    for i, c in enumerate(h):
        cum += c
        if cum >= half:
            return (i + 0.5) * BIN_W


def report(name, st, h):
    n = st["n"]
    if n == 0:
        print(f"{name:>10}: 0 个地址")
        return
    mean = st["s"] / n
    var = max(0.0, st["s2"] / n - mean * mean)
    std = math.sqrt(var)
    med = median_from_hist(h)
    lo = sum(h[:int(ENTROPY_LOW / BIN_W)])
    md = sum(h[int(ENTROPY_LOW / BIN_W):int(ENTROPY_HIGH / BIN_W) + 1])
    hi = n - lo - md
    print(f"{name:>10}: n={n:>9,}  "
          f"mean={mean:.4f}  median={med:.4f}  std={std:.4f}  "
          f"min={st['mn']:.4f}  max={st['mx']:.4f}")
    print(f"           分布: <{ENTROPY_LOW} -> {lo:,} ({lo / n * 100:.2f}%)   "
          f"{ENTROPY_LOW}~{ENTROPY_HIGH} -> {md:,} ({md / n * 100:.2f}%)   "
          f">{ENTROPY_HIGH} -> {hi:,} ({hi / n * 100:.2f}%)")


print("=" * 100)
print("各 /64 前缀集下的地址 IID 香农熵统计（训练集已熵过滤，理论应全部 <= ENTROPY_HIGH）")
print("=" * 100)
report("low", stats["low"], hist["low"])
report("mid", stats["mid"], hist["mid"])
report("high", stats["high"], hist["high"])
report("unmatched(未落入三个前缀集)", stats["unmatched"], hist["unmatched"])
print("=" * 100)
