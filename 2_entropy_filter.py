import os
import socket
import binascii
import math

from tqdm import tqdm

from common import cfg, require

# =====================================================================
# 【核心优化 1：空间换时间 (查表法)】
# IID 长度固定为 16 个字符。所以一个字符出现的次数只可能是 1 到 16。
# 我们直接在程序启动时，预先把所有可能的概率和 log2 计算好，存进字典。
# 循环中只需要直接取值，彻底消灭 math.log2() 浮点运算开销！
# =====================================================================
ENTROPY_TABLE = {
    count: (count / 16.0) * math.log2(count / 16.0)
    for count in range(1, 17)
}

INPUT_CLEAN = os.path.join(cfg.DATA_DIR, "clean-responsive-addresses.txt")
OUTPUT_GOLD = os.path.join(cfg.DATA_DIR, "gold-training-addresses.txt")

ENTROPY_LOW = require("ENTROPY_LOW", cfg.ENTROPY_LOW)
ENTROPY_HIGH = require("ENTROPY_HIGH", cfg.ENTROPY_HIGH)


def filter_by_entropy_fast(clean_ips_file, output_gold_file):
    print(f"[*] 正在分析与过滤数据: {clean_ips_file}")

    # 快速统计总行数（rb模式更快）
    print("[*] 正在统计活跃 IP 总数...")
    total_lines = sum(1 for _ in open(clean_ips_file, 'rb'))

    low_label = f"Low (<{ENTROPY_LOW})"
    mid_label = f"Medium ({ENTROPY_LOW}-{ENTROPY_HIGH})"
    high_label = f"High (>{ENTROPY_HIGH}, Random)"
    entropy_buckets = {low_label: 0, mid_label: 0, high_label: 0}
    gold_count = 0
    discard_count = 0

    # 使用大缓冲区读写，提升 I/O 性能
    with open(clean_ips_file, 'r', encoding='ascii') as f_in, \
         open(output_gold_file, 'w', encoding='ascii') as f_out:

        for line in tqdm(f_in, total=total_lines, desc="极速信息熵计算"):
            line = line.strip()
            if not line:
                continue

            try:
                # =========================================================
                # 【核心优化 2：弃用 ipaddress 库，改用底层 C API】
                # socket.inet_pton 是用 C 语言写的底层网络 API，速度比 ipaddress 快几十倍
                # 将 IP 转换为 16 字节的二进制 packed 格式
                # =========================================================
                packed = socket.inet_pton(socket.AF_INET6, line)

                # 转为十六进制的 bytes (32 长度), 并直接切片取后 16 位 (IID)
                iid_bytes = binascii.hexlify(packed)[16:]

                # =========================================================
                # 【核心优化 3：极速计算香农熵】
                # set() 提取唯一字符，然后直接通过 count 查表累加
                # =========================================================
                ent = 0.0
                for b in set(iid_bytes):
                    ent -= ENTROPY_TABLE[iid_bytes.count(b)]

                # 分桶统计
                if ent < ENTROPY_LOW:
                    entropy_buckets[low_label] += 1
                elif ent <= ENTROPY_HIGH:
                    entropy_buckets[mid_label] += 1
                else:
                    entropy_buckets[high_label] += 1

                # 过滤逻辑（保留 <= ENTROPY_HIGH 的黄金数据）
                if ent <= ENTROPY_HIGH:
                    f_out.write(line + '\n')
                    gold_count += 1
                else:
                    discard_count += 1

            except Exception:
                continue

    print("\n" + "=" * 50)
    print("✅ 模式极速过滤完成！【论文图表数据】统计报告：")
    print(f"   [分布] {low_label:<28}: {entropy_buckets[low_label]}")
    print(f"   [分布] {mid_label:<28}: {entropy_buckets[mid_label]}")
    print(f"   [分布] {high_label:<28}: {entropy_buckets[high_label]}")
    print("-" * 50)
    print(f"   -> 主动抛弃的高熵随机地址数: {discard_count}")
    print(f"   -> 提炼出的【黄金训练集】数: {gold_count}")
    print(f"   -> 黄金数据已保存至: {output_gold_file}")
    print("=" * 50)


if __name__ == "__main__":
    filter_by_entropy_fast(INPUT_CLEAN, OUTPUT_GOLD)
