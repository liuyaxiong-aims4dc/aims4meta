#!/usr/bin/env python3
"""
L4: 生成 SIRIUS 输入文件（MS格式）

新策略：
- 处理整个原始数据MSP文件（不仅仅是L1-L3未鉴定的部分）
- 关联CSV中的同位素分布信息
- 生成SIRIUS .ms格式文件
- 用户自行使用SIRIUS GUI进行分析

输入：
  - 原始样品MSP文件
  - 原始样品CSV文件（包含同位素分布信息）
  
输出：
  - SIRIUS .ms格式文件（包含MS1同位素模式和MS2谱图）
  - 统计摘要文件
"""

###############################################################################
# 配置参数区（全部由总控脚本注入，无默认值）
###############################################################################

import os
import sys
import re
import argparse
import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 输入: MSP 文件路径
SAMPLE_MSP = os.environ.get('L4_SAMPLE_MSP')

# 输入: 同位素JSON文件路径（由CSV关联脚本生成）
ISOTOPE_JSON = os.environ.get('L4_ISOTOPE_JSON', '')

# 输出目录
OUTPUT_DIR = os.environ.get('L4_OUTPUT_DIR', "")

# 离子模式
ION_MODE = os.environ.get('L4_ION_MODE', 'POS')

# m/z 阈值
MZ_THRESHOLD = os.environ.get('L4_MZ_THRESHOLD', '850')

# 验证必需参数
_missing = []
if not SAMPLE_MSP:
    _missing.append('L4_SAMPLE_MSP')

if _missing:
    raise ValueError(f"错误：以下环境变量未设置，必须由总控脚本提供：{', '.join(_missing)}")

# 类型转换
MZ_THRESHOLD = int(MZ_THRESHOLD)

# 13C-12C 质量差，用于计算同位素峰 m/z 间距
ISOTOPE_MASS_DIFF = 1.003355

###############################################################################


def parse_arguments():
    parser = argparse.ArgumentParser(description='L4: 生成SIRIUS输入文件')
    parser.add_argument('--sample_msp', default=SAMPLE_MSP,
                        help='输入MSP文件（原始样品数据）')
    parser.add_argument('--isotope_json', default=ISOTOPE_JSON,
                        help='同位素分布JSON文件（由CSV关联脚本生成）')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                        help='输出目录（留空自动推断）')
    parser.add_argument('--ion_mode', default=ION_MODE, choices=['POS', 'NEG'],
                        help='离子模式 (POS/NEG)')
    parser.add_argument('--mz_threshold', type=int, default=MZ_THRESHOLD,
                        help='m/z上限')
    return parser.parse_args()


# ============================================================
# MSP 预处理
# ============================================================


def preprocess_msp(input_file, output_file, mz_threshold, ion_mode='POS'):
    """
    预处理MSP文件：
    - 保留原始加合物类型（SIRIUS 原生支持 [M+H]+, [M+Na]+, [M+K]+ 等）
    - 无加合物的样品添加默认 [M+H]+ / [M-H]-（正常不会出现此情况）
    - 过滤多电荷离子
    - 过滤 m/z > 阈值的化合物
    """
    default_adduct = "[M+H]+" if ion_mode == 'POS' else "[M-H]-"

    compounds_total = 0
    compounds_kept = 0
    compounds_adduct_added = 0
    compounds_skipped_mz = 0
    compounds_skipped_charge = 0
    adduct_stats = {}

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        current_lines = []
        precursor_type_value = None
        precursor_type_idx = -1
        charge_line_idx = -1
        charge_value = None
        precursor_mz = None
        precursor_mz_idx = -1

        def process_compound():
            nonlocal compounds_total, compounds_kept
            nonlocal compounds_adduct_added
            nonlocal compounds_skipped_mz, compounds_skipped_charge

            compounds_total += 1

            if precursor_mz is not None and precursor_mz >= mz_threshold:
                compounds_skipped_mz += 1
                return

            if precursor_mz is None:
                compounds_skipped_mz += 1
                return

            if charge_value in ["2+", "3+", "4+", "5+", "6+", "7+"]:
                compounds_skipped_charge += 1
                return

            final_adduct = precursor_type_value
            if precursor_type_idx < 0:
                if charge_line_idx >= 0:
                    current_lines[charge_line_idx] = f"Precursor_type: {default_adduct}"
                    final_adduct = default_adduct
                    compounds_adduct_added += 1
                else:
                    insert_idx = precursor_mz_idx + 1 if precursor_mz_idx >= 0 else len(current_lines)
                    current_lines.insert(insert_idx, f"Precursor_type: {default_adduct}")
                    final_adduct = default_adduct
                    compounds_adduct_added += 1

            if final_adduct:
                adduct_stats[final_adduct] = adduct_stats.get(final_adduct, 0) + 1

            for line in current_lines:
                outfile.write(line + "\n")

            compounds_kept += 1

        for line in infile:
            stripped = line.rstrip()

            if not stripped:
                if current_lines:
                    current_lines.append("")
                    process_compound()
                current_lines = []
                precursor_type_value = None
                precursor_type_idx = -1
                charge_line_idx = -1
                charge_value = None
                precursor_mz = None
                precursor_mz_idx = -1
                continue

            current_lines.append(stripped)

            upper = stripped.upper()
            if upper.startswith('PRECURSORTYPE:') or upper.startswith('ADDUCT:') or upper.startswith('PRECURSOR_TYPE:'):
                precursor_type_idx = len(current_lines) - 1
                parts = stripped.split(':', 1)
                if len(parts) > 1:
                    precursor_type_value = parts[1].strip()

            charge_match = re.search(r'^Charge\s*:\s*([0-9]+)([-+])',
                                     stripped, re.IGNORECASE)
            if charge_match:
                charge_line_idx = len(current_lines) - 1
                charge_value = f"{charge_match.group(1)}{charge_match.group(2)}"

            mz_match = re.search(r'^PrecursorMZ\s*:\s*([0-9.]+)',
                                 stripped, re.IGNORECASE)
            if mz_match:
                precursor_mz_idx = len(current_lines) - 1
                try:
                    precursor_mz = float(mz_match.group(1))
                except ValueError:
                    pass

        if current_lines:
            process_compound()

    print(f"  预处理: {compounds_total} → {compounds_kept} 条 (m/z过滤: {compounds_skipped_mz}, 多电荷过滤: {compounds_skipped_charge})")

    return output_file, compounds_kept


def load_isotope_from_json(json_file):
    """
    从JSON文件加载同位素分布信息
    
    JSON文件由"辅助功能/同位素信息提取"脚本生成
    该脚本复用了"原始数据CSV关联"的匹配逻辑，统一管理CSV关联
    
    返回: {compound_id: [abundance_list]} 映射
    """
    if not json_file or not os.path.exists(json_file):
        print(f"  同位素JSON: 文件不存在 ({json_file})")
        return {}
    
    isotope_map = {}
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            isotope_map = json.load(f)
        
        # 转换列表中的值为float
        for key in isotope_map:
            isotope_map[key] = [float(x) for x in isotope_map[key]]
        
        print(f"  同位素JSON: 加载 {len(isotope_map)} 条有效分布")
    except Exception as e:
        print(f"  同位素JSON: 加载失败 ({e})")
        import traceback
        traceback.print_exc()
    
    return isotope_map


def extract_compound_id(name_field, pattern=r'\(([^)]+)\)'):
    """
    从NAME字段中提取compound_id
    
    例如: "Unknown (10.10_380.1323m/z)" → "10.10_380.1323m/z"
    """
    if not name_field:
        return ''
    
    match = re.search(pattern, name_field)
    if match:
        return match.group(1)
    
    return name_field


def find_isotope_abundances(compound_id, isotope_map, ion_mode='POS'):
    """
    精确匹配 compound_id 到同位素映射
    
    与其他层级的CSV关联逻辑一致：
    - MSP的NAME格式: "Unknown (1.09_437.9778m/z)"
    - CSV的Compound格式: "4.63_678.4991n" 或 "4.63_678.4991"
    - 从MSP NAME提取括号内的ID: "1.09_437.9778m/z"
    - 需要将"m/z"后缀转换为"n"或"p"后缀进行匹配
    
    匹配策略：
    1. 精确匹配（直接匹配）
    2. 后缀转换匹配（m/z → n/p）
    """
    if not compound_id or not isotope_map:
        return []

    # 策略1: 精确匹配
    if compound_id in isotope_map:
        return isotope_map[compound_id]

    # 策略2: 后缀转换匹配
    # MSP使用"m/z"后缀，CSV使用"n"(负模式)或"p"(正模式)后缀
    if compound_id.endswith('m/z'):
        base_id = compound_id[:-3]  # 去掉"m/z"
        # 根据离子模式尝试对应的后缀
        if ion_mode == 'POS':
            # 正模式：尝试p后缀
            test_id = base_id + 'p'
            if test_id in isotope_map:
                return isotope_map[test_id]
            # 也尝试n后缀（有些数据可能标记错误）
            test_id = base_id + 'n'
            if test_id in isotope_map:
                return isotope_map[test_id]
        else:
            # 负模式：尝试n后缀
            test_id = base_id + 'n'
            if test_id in isotope_map:
                return isotope_map[test_id]
            # 也尝试p后缀
            test_id = base_id + 'p'
            if test_id in isotope_map:
                return isotope_map[test_id]
        
        # 尝试无后缀
        if base_id in isotope_map:
            return isotope_map[base_id]

    return []


# ============================================================
# MS1 同位素峰生成
# ============================================================

def generate_ms1_peaks(precursor_mz, isotope_abundances, base_intensity=10000.0):
    """
    根据前体离子 m/z 和同位素相对丰度，生成 MS1 峰列表
    
    SIRIUS .ms 格式的 >ms1 段需要 m/z intensity 对
    同位素峰间距为 ISOTOPE_MASS_DIFF (1.003355 Da，13C-12C 质量差)
    
    输入: precursor_mz=315.085, isotope_abundances=[100, 15.9, 0.706]
    输出: [(315.085, 10000.0), (316.088, 1590.0), (317.092, 70.6)]
    """
    if not isotope_abundances or precursor_mz <= 0:
        return []
    
    peaks = []
    for i, rel_abundance in enumerate(isotope_abundances):
        mz = precursor_mz + i * ISOTOPE_MASS_DIFF
        intensity = rel_abundance / 100.0 * base_intensity
        peaks.append((mz, intensity))
    
    return peaks


# ============================================================
# MSP 转 SIRIUS .ms 格式
# ============================================================

def convert_msp_to_ms_with_ms1(processed_msp, isotope_map, output_ms, ion_mode='POS'):
    """
    将预处理后的 MSP 转为 SIRIUS .ms 格式，并注入 MS1 同位素模式
    
    SIRIUS .ms 格式:
        >compound <name>
        >parentmass <precursor_mz>
        >ionization <adduct>
        
        >ms1
        315.085 10000
        316.088 1590
        317.092 70.6
        
        >ms2
        78.042 6133
        226.064 1265
        ...
    
    返回: (output_ms_path, n_with_ms1, n_total)
    """
    n_total = 0
    n_with_ms1 = 0
    debug_count = 0
    
    with open(processed_msp, 'r', encoding='utf-8') as infile, \
         open(output_ms, 'w', encoding='utf-8') as outfile:
        
        current_name = ''
        current_mz = 0.0
        current_adduct = ''
        current_comment = ''
        current_peaks = []
        in_peaks = False
        
        def write_compound():
            nonlocal n_total, n_with_ms1, debug_count
            if not current_name:
                return
            
            n_total += 1
            
            # 写入化合物信息（SIRIUS MS格式）
            outfile.write(f">compound {current_name}\n")
            outfile.write(f">parentmass {current_mz}\n")
            outfile.write(f">ionization {current_adduct}\n")
            outfile.write("\n")
            
            # 提取compound_id并匹配同位素
            compound_id = extract_compound_id(current_name, r'\(([^)]+)\)')
            if not compound_id:
                compound_id = current_comment
            
            abundances = find_isotope_abundances(compound_id, isotope_map, ion_mode)
            
            # 调试输出
            if debug_count < 5:
                print(f"  [DEBUG] 化合物 {n_total}: {compound_id} → {'匹配成功' if abundances else '未匹配'}")
                if abundances:
                    print(f"          同位素分布: {abundances}")
                debug_count += 1
            
            # 如果找到同位素信息，写入MS1字段
            if abundances:
                ms1_peaks = generate_ms1_peaks(current_mz, abundances)
                if ms1_peaks:
                    outfile.write(">ms1\n")
                    for mz, intensity in ms1_peaks:
                        outfile.write(f"{mz:.6f} {intensity:.1f}\n")
                    outfile.write("\n")
                    n_with_ms1 += 1
            
            # 写入MS2谱图
            if current_peaks:
                outfile.write(">ms2\n")
                for mz, intensity in current_peaks:
                    outfile.write(f"{mz} {intensity}\n")
                outfile.write("\n")
        
        for line in infile:
            stripped = line.rstrip()
            
            if not stripped:
                if current_name:
                    write_compound()
                current_name = ''
                current_mz = 0.0
                current_adduct = ''
                current_comment = ''
                current_peaks = []
                in_peaks = False
                continue
            
            upper = stripped.upper()
            
            if upper.startswith('NAME:'):
                current_name = stripped.split(':', 1)[1].strip()
                in_peaks = False
                continue
            
            mz_match = re.match(r'^PRECURSORMZ\s*:\s*([0-9.]+)', stripped, re.IGNORECASE)
            if mz_match:
                try:
                    current_mz = float(mz_match.group(1))
                except ValueError:
                    pass
                in_peaks = False
                continue
            
            if re.match(r'^PRECURSOR_?TYPE\s*:', stripped, re.IGNORECASE):
                current_adduct = stripped.split(':', 1)[1].strip()
                in_peaks = False
                continue
            
            if upper.startswith('COMMENT:'):
                current_comment = stripped.split(':', 1)[1].strip()
                in_peaks = False
                continue
            
            if upper.startswith('NUM PEAKS:'):
                in_peaks = True
                continue
            
            if in_peaks:
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        peak_mz = parts[0]
                        peak_int = parts[1]
                        float(peak_mz)
                        float(peak_int)
                        current_peaks.append((peak_mz, peak_int))
                    except ValueError:
                        pass
        
        if current_name:
            write_compound()
    
    print(f"  MS1注入: {n_with_ms1}/{n_total} 化合物成功关联到CSV同位素分布")
    return output_msp, n_with_ms1, n_total

def main():
    args = parse_arguments()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.sample_msp).parent / 'L4_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"L4: 生成 SIRIUS 输入文件 | {args.sample_msp} | {args.ion_mode}")
    print("=" * 70)

    if not os.path.exists(args.sample_msp):
        print(f"错误: 输入文件不存在: {args.sample_msp}")
        return False

    # ===== [1/2] 加载同位素分布信息 =====
    print("\n[1/2] 加载同位素分布信息...")
    isotope_map = load_isotope_from_json(args.isotope_json)

    # ===== [2/2] 生成SIRIUS MS文件 =====
    print("\n[2/2] 生成SIRIUS MS文件...")
    output_ms = str(output_dir / 'sirius_input.ms')
    output_ms, n_with_ms1, n_total = convert_msp_to_ms_with_ms1(
        args.sample_msp, isotope_map, output_ms)

    _write_summary(output_dir, args, n_total, n_with_ms1, isotope_map)

    print("\n" + "=" * 70)
    print(f"L4 完成 | 化合物: {n_total} | 含MS1同位素: {n_with_ms1}")
    print(f"输出文件: {output_ms}")
    print("=" * 70)
    print("\n提示: 请使用 SIRIUS GUI 打开生成的 .ms 文件进行分析")
    print("      SIRIUS GUI 可以提供交互式的分子式预测和结构鉴定")

    return True


def _write_summary(output_dir, args, n_total, n_with_ms1, isotope_map):
    """写统计摘要"""
    summary_file = Path(output_dir) / 'L4_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("L4 SIRIUS 输入文件生成统计\n")
        f.write("=" * 60 + "\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入MSP: {args.sample_msp}\n")
        f.write(f"同位素JSON: {args.isotope_json if args.isotope_json else '无'}\n")
        f.write(f"离子模式: {args.ion_mode}\n")
        f.write(f"m/z阈值: {args.mz_threshold}\n\n")
        f.write(f"处理化合物数: {n_total}\n")
        f.write(f"含MS1同位素模式: {n_with_ms1}\n")
        f.write(f"同位素分布匹配率: {n_with_ms1/n_total*100:.1f}%\n" if n_total > 0 else "同位素分布匹配率: N/A\n")
        f.write("\n")
        f.write("输出文件:\n")
        f.write(f"  - sirius_input.ms (SIRIUS输入文件)\n")
        f.write("\n")
        f.write("说明: 请使用 SIRIUS GUI 打开 sirius_input.ms 文件\n")
        f.write("      进行交互式的分子式预测和结构鉴定分析\n")
        f.write("=" * 60 + "\n")


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
