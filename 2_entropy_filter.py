import socket
import binascii
import math
from tqdm import tqdm

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

def filter_by_entropy_fast(clean_ips_file, output_gold_file):
    print(f"[*] 正在分析与过滤数据: {clean_ips_file}")
    
    # 快速统计总行数（rb模式更快）
    print("[*] 正在统计活跃 IP 总数...")
    total_lines = sum(1 for _ in open(clean_ips_file, 'rb'))
    
    entropy_buckets = {"Low (<1.5)": 0, "Medium (1.5-3.0)": 0, "High (>3.0, Random)": 0}
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
                # 例如: b'20010db8000000000000000000000001'[16:] -> b'0000000000000001'
                iid_bytes = binascii.hexlify(packed)[16:]
                
                # =========================================================
                # 【核心优化 3：极速计算香农熵】
                # set() 提取唯一字符，然后直接通过 count 查表累加，没有复杂对象创建
                # =========================================================
                ent = 0.0
                for b in set(iid_bytes):
                    ent -= ENTROPY_TABLE[iid_bytes.count(b)]
                
                # 分桶统计
                if ent < 1.5:
                    entropy_buckets["Low (<1.5)"] += 1
                elif ent <= 3.0:
                    entropy_buckets["Medium (1.5-3.0)"] += 1
                else:
                    entropy_buckets["High (>3.0, Random)"] += 1
                
                # 过滤逻辑 (保留 <= 3.0 的黄金数据)
                if ent <= 3.0:
                    f_out.write(line + '\n')
                    gold_count += 1
                else:
                    discard_count += 1
                    
            except Exception:
                continue

    print("\n" + "=" * 50)
    print("✅ 模式极速过滤完成！【论文图表数据】统计报告：")
    print(f"   [分布] Low Entropy (<1.5, 高度结构化)  : {entropy_buckets['Low (<1.5)']}")
    print(f"   [分布] Medium Entropy(1.5-3.0, 混合规律): {entropy_buckets['Medium (1.5-3.0)']}")
    print(f"   [分布] High Entropy (>3.0, 随机盲盒)   : {entropy_buckets['High (>3.0, Random)']}")
    print("-" * 50)
    print(f"   -> 主动抛弃的高熵随机地址数: {discard_count}")
    print(f"   -> 提炼出的【黄金训练集】数: {gold_count}")
    print(f"   -> 黄金数据已保存至: {output_gold_file}")
    print("=" * 50)

if __name__ == "__main__":
    INPUT_CLEAN = "data/clean-responsive-addresses.txt"
    OUTPUT_GOLD = "data/gold-training-addresses.txt"
    
    filter_by_entropy_fast(INPUT_CLEAN, OUTPUT_GOLD)