import os
import ipaddress
import bisect

from tqdm import tqdm

from common import cfg


def load_and_merge_aliases(alias_file):
    """读取别名前缀文件，转换为整数区间，并合并重叠区间"""
    print(f"[*] 正在加载别名前缀: {alias_file}")
    intervals = []

    with open(alias_file, "r") as f:
        for line in tqdm(f, desc="解析网段"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                network = ipaddress.IPv6Network(line, strict=False)
                start_ip = int(network.network_address)
                end_ip = int(network.broadcast_address)
                intervals.append((start_ip, end_ip))
            except Exception:
                pass

    print(f"[*] 共解析出 {len(intervals)} 个别名网段。正在合并重叠区间...")
    intervals.sort(key=lambda x: x[0])

    merged = []
    for current in tqdm(intervals, desc="合并区间"):
        if not merged:
            merged.append(current)
        else:
            last = merged[-1]
            if current[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

    print(f"[*] 合并完成！最终得到 {len(merged)} 个独立别名区间。")
    return merged


def is_aliased(ip_int, merged_intervals, starts):
    """使用二分查找判断一个 IP 是否落在别名区间内"""
    idx = bisect.bisect_right(starts, ip_int)
    if idx == 0:
        return False
    last_interval = merged_intervals[idx - 1]
    if last_interval[0] <= ip_int <= last_interval[1]:
        return True
    return False


def clean_dataset(raw_ips_file, alias_file, output_file):
    merged_aliases = load_and_merge_aliases(alias_file)

    # 在循环外部只提取一次左边界列表
    print("[*] 正在提取二分查找索引 (这可能需要几秒钟)...")
    starts = [iv[0] for iv in merged_aliases]

    print(f"[*] 正在清洗活跃 IP 列表: {raw_ips_file}")
    clean_ips_count = 0
    aliased_ips_count = 0

    print("[*] 正在统计活跃 IP 总数...")
    total_lines = sum(1 for _ in open(raw_ips_file, "r"))

    with open(raw_ips_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in tqdm(f_in, total=total_lines, desc="极速清洗中"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                ip_int = int(ipaddress.IPv6Address(line))
                if is_aliased(ip_int, merged_aliases, starts):
                    aliased_ips_count += 1
                else:
                    f_out.write(line + "\n")
                    clean_ips_count += 1
            except Exception:
                continue

    print("\n" + "=" * 40)
    print("✅ 清洗完成！统计报告：")
    print(f"   - 原始活跃 IP 总数: {total_lines}")
    print(f"   - 剔除的别名 IP 数量: {aliased_ips_count}")
    print(f"   - 保留的干净 IP 数量: {clean_ips_count}")
    print(f"   - 干净数据已保存至: {output_file}")
    print("=" * 40)


if __name__ == "__main__":
    RAW_IPS = os.path.join(cfg.DATA_DIR, "responsive-addresses.txt")
    ALIAS_PREFIXES = os.path.join(cfg.DATA_DIR, "aliased-prefixes.txt")
    OUTPUT_CLEAN_IPS = os.path.join(cfg.DATA_DIR, "clean-responsive-addresses.txt")

    clean_dataset(RAW_IPS, ALIAS_PREFIXES, OUTPUT_CLEAN_IPS)
