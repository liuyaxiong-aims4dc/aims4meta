#!/usr/bin/env python3
"""
多层鉴定总控脚本（方法导向架构）

统一管理所有参数配置，按顺序调用各层级子脚本：
L1_DreaMS向量化 → L1_matchMS鉴定 → L1_DreaMS鉴定 → L1_整合 → L2_matchMS鉴定 → L2_DreaMS鉴定 → L2_整合 → L3_SIRIUS结构鉴定 → L4_DreaMS分子网络

架构说明：
- L1: 实验数据库鉴定（真实谱图：MSDIAL + SpectraVerse + PFAS实验谱图，多库遍历，跨库合并Top-K）
- L2: 模拟数据库鉴定（模拟谱图：FIORA + CFM-ID预测谱图）
- L1/L2均采用方法导向：matchms方法 + DreaMS方法
- matchms方法：使用matchms库的ModifiedCosine鉴定
- DreaMS方法：使用DreaMS模型的Embedding cosine鉴定
- L3：SIRIUS结构鉴定（分子式预测+结构推导）
- L4：DreaMS分子网络（纯样品间Embedding互连，不依赖数据库）

数据库推荐配置：
- L1实验库：通用代谢物用MSDIAL+SpectraVerse，PFAS检测加载PFAS实验库
- L2模拟库：正离子用FIORA+CFM-ID，负离子仅用FIORA（CFM-ID负离子碎片太少）

支持的数据格式：
- mzML格式（标准质谱数据格式）

使用方式：
    # 基本用法
    python 总控脚本.py --sample 样品.msp --ion_mode POS
"""

###############################################################################
# ==================== 参数配置区（总控脚本统一管理） ====================
###############################################################################
# 所有参数在此区域集中配置，子脚本不再硬编码任何参数
###############################################################################

# ============================================================
# 零、运行控制开关（优先级最高）
# ============================================================

# ----- 层级运行开关（Y=启用 / N=禁用） -----
RUN_L1 = "Y"          # 是否运行L1实验数据库鉴定
RUN_L2 = "Y"          # 是否运行L2模拟数据库鉴定
RUN_L3 = "Y"          # 是否运行L3 SIRIUS结构鉴定
RUN_L4 = "Y"          # 是否运行L4 DreaMS分子网络

# ----- L1 数据库开关 -----
L1_USE_MSDIAL = "Y"          # 使用MSDIAL通用实验库
L1_USE_SPECTRAVERSE = "Y"    # 使用SpectraVerse通用实验库

# ----- L2 数据库开关（大类前缀匹配：开关名对应字典键前缀） -----
L2_USE_TCM = "Y"              # 覆盖 TCM_* 所有条目（如 TCM_FIORA、TCM_CFMID）
L2_USE_DRUG = "Y"             # 覆盖 DRUG_* 所有条目（DrugBank、FDA等）
L2_USE_LIPID = "N"            # 覆盖 LIPID 条目
L2_USE_PFAS = "N"             # 覆盖 PFAS_* 所有条目

# ----- L3 在线数据库配置 -----
# 固定使用 BIO 在线库（CSI:FingerID 内置），不启用本地库
# 理由：本地库体积庞大、维护成本高；BIO 覆盖全面且版本由 SIRIUS 统一管理
L3_ONLINE_DATABASES = ["BIO"]

# ----- L4无数据库依赖（纯样品间Embedding互连） -----

# ============================================================
# 一、基础配置
# ============================================================

# ----- 样品数据 -----
SAMPLE = "/stor1/微力临时文件共享/20260508kangfengshi/kangfengshi-NEG.msp"
ION_MODE = "NEG"  # 离子模式: "POS" 或 "NEG"，仅支持单选项

# ----- 输出设置 -----
OUTPUT_BASE_DIR = ""  # 输出根目录（留空则使用样品文件同级目录）

# ============================================================
# 二、L1 实验数据库鉴定参数
# ============================================================

# ----- L1 数据库配置 -----
# 数据库路径定义 - L1实验数据库（仅真实实验谱图）
L1_EXPERIMENTAL_DATABASES = {
    "MSDIAL": {
        "POS": "/stor3/AIMS4Meta/数据库/实验数据库/MSDIAL/MSMS-Public_experimentspectra-pos-VS19.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/实验数据库/MSDIAL/MSMS-Public_experimentspectra-neg-VS19.msp",
    },
    "SpectraVerse": {
        "POS": "/stor3/AIMS4Meta/数据库/实验数据库/spectraverse/spectraverse-1.0.1-pos.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/实验数据库/spectraverse/spectraverse-1.0.1-neg.msp",
    },
}

# ----- L1 鉴定参数 -----
L1_PRECURSOR_PPM = 10.0            # 母离子容差 (ppm)
L1_FRAGMENT_TOLERANCE = 0.05       # 碎片容差 (Da)
L1_MIN_MATCHED_FRAGMENTS = 2       # 最少鉴定碎片数（matchMS 和 DreaMS 共用）
L1_COSINE_THRESHOLD = 0.5          # 余弦相似度阈值（matchMS 和 DreaMS 共用）
L1_TOP_K = 5                       # Top-K 结果数

# ----- L1 DreaMS 向量化参数 -----
L1_BATCH_SIZE = 256              # DreaMS 批处理大小（A30可用更大batch）

# ============================================================

# ----- L2 数据库配置 -----
# L2模拟数据库（理论预测谱图）
# 注意：CFM-ID MSP需先运行 fix_cfmid_msp.py 修复换行问题，否则matchMS崩溃
L2_SIMULATED_DATABASES = {
    "TCM_FIORA": {
        "POS": "/stor3/AIMS4Meta/数据库/模拟数据库/TCM/coconut_csv_lite-04-2026_simulated_fiora_v1.0.1_POS.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/模拟数据库/TCM/coconut_csv_lite-04-2026_simulated_fiora_v1.0.1_NEG.msp",
    },
    "TCM_CFMID": {
        "POS": "/stor3/AIMS4Meta/数据库/模拟数据库/TCM/tcmbank_herb_combined_simulated_cfmid_v4_POS.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/模拟数据库/TCM/tcmbank_herb_combined_simulated_cfmid_v4_NEG.msp",
    },
    "TCM_COCONUT_CFMID": {
        "POS": "/stor3/AIMS4Meta/数据库/模拟数据库/TCM/coconut_csv_lite-04-2026_simulated_cfmid_v4_POS.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/模拟数据库/TCM/coconut_csv_lite-04-2026_simulated_cfmid_v4_NEG.msp",
    },
    "DRUG_FIORA_DRUGBANK": {
        "POS": "/stor3/AIMS4Meta/数据库/模拟数据库/DRUG/drugbank小分子药物_simulated_fiora_v1.0.1_POS.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/模拟数据库/DRUG/drugbank小分子药物_simulated_fiora_v1.0.1_NEG.msp",
    },
    "DRUG_FIORA_FDA": {
        "POS": "/stor3/AIMS4Meta/数据库/模拟数据库/DRUG/FDA批准上市小分子药物_POS_simulated_fiora_v1.0.1_POS.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/模拟数据库/DRUG/FDA批准上市小分子药物_NEG_simulated_fiora_v1.0.1_NEG.msp",
    },
    "LIPID": {
        "POS": "/stor3/AIMS4Meta/数据库/模拟数据库/LIPID/MSDIAL-TandemMassSpectralAtlas-VS69-Pos.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/模拟数据库/LIPID/MSDIAL-TandemMassSpectralAtlas-VS69-Neg.msp",
    },
    "PFAS_FIORA": {
        "POS": "/stor3/AIMS4Meta/数据库/模拟数据库/PFAS/PFAS_compounds_5755_simulated_fiora_v1.0.1_POS.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/模拟数据库/PFAS/PFAS_compounds_5755_simulated_fiora_v1.0.1_NEG.msp",
    },
}

# ----- L2 鉴定参数 -----
L2_PRECURSOR_PPM = L1_PRECURSOR_PPM
L2_FRAGMENT_TOLERANCE = L1_FRAGMENT_TOLERANCE
L2_MIN_MATCHED_FRAGMENTS = L1_MIN_MATCHED_FRAGMENTS
L2_COSINE_THRESHOLD = L1_COSINE_THRESHOLD
L2_TOP_K = L1_TOP_K

# ============================================================
# 三、L3 SIRIUS 结构鉴定参数
# ============================================================

# ----- 鉴定参数 -----
L3_SIRIUS_INSTRUMENT = "qtof"  # orbitrap 或 qtof
L3_MIN_PEAKS = 6             # 最小碎片峰数量（默认：6）
L3_MAX_MZ = 1500             # 最大m/z值（默认：1500）
L3_MIN_INTENSITY = 500       # 信号强度阈值：配套CSV 的 Maximum Abundance 低于该值则跳过（0 停用）
                             # HDMSE 开淌度后信号较低，默认500（例大黄样品约保留 20-25%化合物）。根据实际丰度分布调整

# ============================================================
# 四、L4 DreaMS 分子网络参数
# ============================================================

# ----- 分子网络参数 -----
L4_SIM_THRESHOLD = 0.8              # DreaMS Embedding 余弦相似度阈值（样品间互连）

# ============================================================
# 五、结果汇总参数
# ============================================================

SUMMARY_OUTPUT_FORMAT = "xlsx"     # 汇总输出格式: "xlsx" 或 "csv"
SUMMARY_INCLUDE_L1 = True          # 是否包含 L1 结果
SUMMARY_INCLUDE_L2 = True          # 是否包含 L2 结果
SUMMARY_INCLUDE_L3 = True          # 是否包含 L3 结果
SUMMARY_INCLUDE_L4 = False         # L4为分子网络，不纳入鉴定结果汇总

###############################################################################
# ==================== 导入依赖 ====================
###############################################################################

import json as json_module
import argparse
import os
import sys
import subprocess
import logging
import signal
from datetime import datetime
from pathlib import Path

###############################################################################
# ==================== 信号处理 ====================
###############################################################################

# 全局变量：存储当前运行的子进程
current_process = None

def signal_handler(signum, frame):
    """信号处理器：优雅地终止子进程"""
    global current_process
    logger = logging.getLogger(__name__)
    logger.warning(f"\n收到终止信号 ({signum}), 正在停止子进程...")

    if current_process and current_process.poll() is None:
        logger.warning(f"终止子进程 PID={current_process.pid}")
        current_process.terminate()
        try:
            current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("子进程未响应,强制终止")
            current_process.kill()

    sys.exit(1)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

###############################################################################
# ==================== 日志配置 ====================
###############################################################################

def setup_logging():
    """配置日志：终端显示关键信息，文件记录详细日志"""
    # 日志目录
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # 日志文件名：多层鉴定_YYYYMMDD_HHMMSS.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"多层鉴定_{timestamp}.log"

    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 文件处理器：记录所有详细信息
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 控制台处理器：只显示INFO及以上
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file

# 初始化日志
LOG_FILE = setup_logging()
logger = logging.getLogger(__name__)

###############################################################################
# ==================== 参数构建函数 ====================
###############################################################################

# L1直接使用实验数据库字典
L1_DATABASES = L1_EXPERIMENTAL_DATABASES

def _on(v):
    """开关判断：Y/y/yes/true/1 视为启用，其余（含 N、空、None）视为禁用"""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().upper() in ("Y", "YES", "TRUE", "1")

# L2 开关前缀映射：开关变量 → 字典键前缀
_L2_SWITCH_PREFIX = {
    "L2_USE_TCM":   "TCM",
    "L2_USE_DRUG":  "DRUG",
    "L2_USE_LIPID": "LIPID",
    "L2_USE_PFAS":  "PFAS",
}

def apply_database_switches():
    """根据开关过滤L1/L2数据库"""
    global L1_DATABASES, L2_SIMULATED_DATABASES

    # L1：精确键名匹配
    filtered_l1 = {}
    if _on(L1_USE_MSDIAL) and "MSDIAL" in L1_EXPERIMENTAL_DATABASES:
        filtered_l1["MSDIAL"] = L1_EXPERIMENTAL_DATABASES["MSDIAL"]
    if _on(L1_USE_SPECTRAVERSE) and "SpectraVerse" in L1_EXPERIMENTAL_DATABASES:
        filtered_l1["SpectraVerse"] = L1_EXPERIMENTAL_DATABASES["SpectraVerse"]
    L1_DATABASES = filtered_l1

    # L2：大类前缀匹配（开关控制所有以该前缀开头的条目）
    # 约定：条目键名为 "<PREFIX>" 或 "<PREFIX>_<suffix>"
    enabled_prefixes = {
        prefix for switch_name, prefix in _L2_SWITCH_PREFIX.items()
        if _on(globals().get(switch_name))
    }
    filtered_l2 = {}
    for db_name, db_cfg in L2_SIMULATED_DATABASES.items():
        prefix = db_name.split("_", 1)[0]
        if prefix in enabled_prefixes:
            filtered_l2[db_name] = db_cfg
    L2_SIMULATED_DATABASES = filtered_l2

def clear_library_caches(msp_path):
    """清除库文件的所有缓存（matchMS pickle和DreaMS embedding）"""
    import glob
    base = msp_path.replace('.msp', '')
    cache_patterns = [
        f"{base}_matchms_cache.pkl",
        f"{base}_spectra_cache.pkl",
        f"{base}_dreams_emb.npz"
    ]
    for pattern in cache_patterns:
        for cache_file in glob.glob(pattern):
            if os.path.exists(cache_file):
                cache_mtime = os.path.getmtime(cache_file)
                msp_mtime = os.path.getmtime(msp_path)
                if cache_mtime < msp_mtime:
                    os.remove(cache_file)
                    logger.info(f"已删除过期缓存: {cache_file}")

def get_l1_databases(ion_mode, selected_databases=None):
    """根据 L1_DATABASES 自动生成完整的数据库配置（含 embedding 缓存路径）"""
    ion_mode = ion_mode.upper()  # 统一大写，兼容 "neg"/"NEG" 等写法
    databases = {}
    for db_name, ion_mode_paths in L1_DATABASES.items():
        # 如果指定了数据库列表，则只使用指定的数据库
        if selected_databases and db_name not in selected_databases:
            continue

        msp_path = ion_mode_paths.get(ion_mode)
        if not msp_path:
            continue
        # 清除过期缓存
        if os.path.exists(msp_path):
            clear_library_caches(msp_path)
        emb_path = msp_path.replace('.msp', '_dreams_emb.npz')
        databases[db_name] = {"msp": msp_path, "emb": emb_path}
    return databases

def get_l2_databases(ion_mode):
    """根据 L2_SIMULATED_DATABASES 获取L2数据库配置"""
    ion_mode = ion_mode.upper()  # 统一大写
    databases = {}
    for db_name, ion_mode_paths in L2_SIMULATED_DATABASES.items():
        msp_path = ion_mode_paths.get(ion_mode)
        if not msp_path:
            continue
        if not os.path.exists(msp_path):
            logger.warning(f"[L2] 数据库文件不存在: {msp_path}")
            continue
        # 清除过期缓存
        clear_library_caches(msp_path)
        databases[db_name] = {
            "msp": msp_path,
            "emb": msp_path.replace('.msp', '_dreams_emb.npz')
        }
    return databases


def build_l1_sample_vec_env(args):
    """构建 L1 样品向量化环境变量"""
    return {
        'L1_MSP_FILE': args.sample,
        'L1_OUTPUT_DIR': os.path.join(args.output_dir, "L1_results"),
    }

def get_sample_input_file(args):
    """获取样品输入文件"""
    return args.sample

def build_l1_db_vec_env(args):
    """构建 L1 数据库向量化环境变量"""
    l1_databases = get_l1_databases(args.ion_mode, args.databases)
    db_paths = {db_name: db_info["msp"] for db_name, db_info in l1_databases.items()}
    
    return {
        'L1_DATABASES': json_module.dumps({args.ion_mode: db_paths}),
        'L1_ION_MODE': args.ion_mode,
    }

def build_l1_env(args):
    """构建 L1 公共环境变量"""
    sample_file = get_sample_input_file(args)

    # 自动查找同名CSV文件（用于同位素评分）
    sample_csv = None
    if sample_file.lower().endswith('.mzml'):
        auto_csv = sample_file.replace('.mzML', '.csv').replace('.mzml', '.csv')
        if os.path.exists(auto_csv):
            sample_csv = auto_csv
    elif sample_file.lower().endswith('.msp'):
        # MSP文件也尝试查找同名CSV
        auto_csv = sample_file.replace('.msp', '.csv').replace('.MSP', '.csv')
        if os.path.exists(auto_csv):
            sample_csv = auto_csv

    # 同位素CSV信息已在主流程打印，此处不再重复
    # sample_csv通过环境变量传递给L1/L2子脚本

    env = {
        'L1_SAMPLE_MSP': sample_file,
        'L1_SAMPLE_CSV': sample_csv or '',
        'L1_ION_MODE': args.ion_mode,
        'L1_MZ_TOLERANCE_PPM': str(int(L1_PRECURSOR_PPM)),
        'L1_FRAGMENT_TOLERANCE': str(L1_FRAGMENT_TOLERANCE),
        'L1_MIN_MATCHED_PEAKS': str(L1_MIN_MATCHED_FRAGMENTS),
        'L1_COSINE_THRESHOLD': str(L1_COSINE_THRESHOLD),
        'L1_TOP_K': str(L1_TOP_K),
        'L1_OUTPUT_DIR': os.path.join(args.output_dir, "L1_results"),
    }
    return env

def build_l1_mc_env(args, qi_csv_path=None):
    """构建 L1 matchMS 鉴定环境变量"""
    l1_databases = get_l1_databases(args.ion_mode, args.databases)
    mc_libraries = {db_name: db_paths["msp"] for db_name, db_paths in l1_databases.items()}
    env = build_l1_env(args)
    env['L1_MATCHMS_LIBRARIES'] = json_module.dumps(mc_libraries)
    if qi_csv_path:
        env['L1_QI_CSV'] = qi_csv_path
    # 传递sample_csv用于同位素相似度计算
    if 'L1_SAMPLE_CSV' in env and env['L1_SAMPLE_CSV']:
        pass  # 已在build_l1_env中设置
    return env

def build_l1_dreams_env(args):
    """构建 L1 DreaMS 鉴定环境变量"""
    sample_emb_path = os.path.join(args.output_dir, "L1_results", "embeddings.npz")
    l1_output_dir = os.path.join(args.output_dir, "L1_results")
    env = build_l1_env(args)
    env['L1_SAMPLE_EMB'] = sample_emb_path
    env['L1_LIBRARIES'] = json_module.dumps(get_l1_databases(args.ion_mode, args.databases))
    env['L1_OUTPUT_DIR'] = l1_output_dir
    env['L1_MIN_MATCHED_PEAKS'] = str(L1_MIN_MATCHED_FRAGMENTS)
    return env

def build_l2_match_env(args, sample_emb_path, simulated_msp=None, simulated_emb=None):
    """构建 L2 鉴定环境变量（matchMS/DreaMS 鉴定共用）"""
    env = {
        'L2_L1_UNIDENTIFIED_MSP': get_sample_input_file(args),  # L2 直接处理全部原始样品
        'L2_ION_MODE': args.ion_mode,
        'L2_OUTPUT_DIR': os.path.join(args.output_dir, "L2_results"),
        'L2_PRECURSOR_PPM': str(L2_PRECURSOR_PPM),
        'L2_FRAGMENT_TOLERANCE': str(L2_FRAGMENT_TOLERANCE),
        'L2_MIN_MATCHED_FRAGMENTS': str(L2_MIN_MATCHED_FRAGMENTS),
        'L2_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
        'L2_MATCHMS_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
        'L2_TOP_K_RESULTS': str(L2_TOP_K),
        'L2_SAMPLE_EMB': sample_emb_path,
    }
    if simulated_msp:
        env['L2_SIMULATED_MSP'] = simulated_msp
    if simulated_emb:
        env['L2_SIMULATED_EMB'] = simulated_emb
    return env
def build_l1_single_file_vec_env(msp_file):
    """构建单/多文件向量化环境变量（L2模拟库用）
    
    参数:
        msp_file: 单个 MSP 路径字符串，或多个路径的列表
    """
    if isinstance(msp_file, list):
        msp_value = json_module.dumps(msp_file)
    else:
        msp_value = msp_file
    env = {
        'L1_MSP_FILE': msp_value,
        'L1_BATCH_SIZE': str(L1_BATCH_SIZE),
        'L1_FORCE_COMPUTE': '0',
    }
    return env

def build_l4_analog_env(args, sample_emb_path):
    """构建 L4 DreaMS 分子网络环境变量
    
    L4为纯分子网络，不依赖任何数据库。
    """
    env = {
        'L4_OUTPUT_DIR': os.path.join(args.output_dir, "L4_results"),
        'L4_SAMPLE_EMB': sample_emb_path,
        'L4_SAMPLE_MSP': get_sample_input_file(args),
        'L4_SIM_THRESHOLD': str(L4_SIM_THRESHOLD),
    }
    return env

def build_summary_env(args):
    """构建结果汇总环境变量"""
    return {
        'SUMMARY_OUTPUT_DIR': args.output_dir,
        'SUMMARY_SAMPLE_MSP': get_sample_input_file(args),
        'SUMMARY_OUTPUT_FORMAT': SUMMARY_OUTPUT_FORMAT,
        'SUMMARY_INCLUDE_L1': str(SUMMARY_INCLUDE_L1),
        'SUMMARY_INCLUDE_L2': str(SUMMARY_INCLUDE_L2),
        'SUMMARY_INCLUDE_L3': str(SUMMARY_INCLUDE_L3),
    }


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多层鉴定总控脚本（方法导向）")

    # 基础参数
    parser.add_argument("--sample", default=SAMPLE, help="样品文件路径（支持MSP/mzML格式）")
    parser.add_argument("--qi_csv", default=None, help="Progenesis QI导出的CSV文件路径（用于关联峰面积等信息）")
    parser.add_argument("--ion_mode", default=ION_MODE, choices=["POS", "NEG"], help="离子模式")
    parser.add_argument("--output_dir", default=OUTPUT_BASE_DIR, help="输出根目录")
    
    # 数据库选择参数
    parser.add_argument("--databases", nargs="*", default=[], help="指定要使用的数据库（默认使用所有数据库）")

    # 控制参数
    parser.add_argument("--skip_l1_vec", action="store_true", help="跳过L1向量化")
    parser.add_argument("--skip_l1", action="store_true", help="跳过L1")
    parser.add_argument("--skip_l2", action="store_true", help="跳过L2")
    parser.add_argument("--skip_l4", action="store_true", help="跳过L4 DreaMS分子网络")
    parser.add_argument("--skip_l3", action="store_true", help="跳过L3 SIRIUS结构鉴定")
    # 快捷参数
    parser.add_argument("--only_l1_vec", action="store_true", help="仅运行L1向量化")
    parser.add_argument("--only_l1", action="store_true", help="仅运行L1")
    parser.add_argument("--only_l2", action="store_true", help="仅运行L2")
    parser.add_argument("--only_l4", action="store_true", help="仅运行L4 DreaMS分子网络")
    parser.add_argument("--only_l3", action="store_true", help="仅运行L3 SIRIUS结构鉴定")
    
    args = parser.parse_args()
    
    # 离子模式统一大写（兼容 "neg"/"pos" 等小写写法）
    args.ion_mode = args.ion_mode.upper()
    
    # 处理快捷参数
    if args.only_l1_vec:
        args.skip_l1 = args.skip_l2 = args.skip_l3 = args.skip_l4 = True
    elif args.only_l1:
        args.skip_l2 = args.skip_l3 = args.skip_l4 = True
    elif args.only_l2:
        args.skip_l1_vec = args.skip_l1 = args.skip_l3 = args.skip_l4 = True
    elif args.only_l3:
        args.skip_l1_vec = args.skip_l1 = args.skip_l2 = args.skip_l4 = True
    elif args.only_l4:
        args.skip_l1_vec = args.skip_l1 = args.skip_l2 = args.skip_l3 = True
    
    # 所有模式都运行最终汇总（即使只跑了一层，也要生成汇总Excel）
    # 唯独 only_l1_vec 纯向量化没有CSV结果，跳过汇总
    if args.only_l1_vec:
        args.skip_summary = True
    else:
        args.skip_summary = False
    
    return args


def run_script(script_path, args_list, env=None):
    """运行子脚本（直接继承终端，支持tqdm原位刷新）"""
    # 获取Python解释器路径
    if env:
        python_exe = f"/home/lyx/miniconda3/envs/{env}/bin/python"
    else:
        python_exe = sys.executable

    cmd = [python_exe, "-u", script_path] + args_list
    
    # 详细命令写入日志文件，终端不显示
    logger.debug(f"[CMD] {' '.join(cmd)}")
    sys.stdout.flush()
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(script_path)
        )
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"脚本执行失败，退出码: {process.returncode}")
            return False
        else:
            logger.debug("脚本执行完成")
            return True
    except Exception as e:
        logger.error(f"执行异常: {e}")
        return False


def run_script_with_env(script_path, env_vars, env=None):
    """运行子脚本（通过环境变量注入参数，无命令行参数）"""
    # 获取Python解释器路径
    if env:
        python_exe = f"/home/lyx/miniconda3/envs/{env}/bin/python"
    else:
        python_exe = sys.executable

    # 构建环境变量
    full_env = os.environ.copy()
    full_env.update(env_vars)
    
    cmd = [python_exe, "-u", script_path]
    
    # 详细信息写入日志文件，终端不显示
    logger.debug(f"[CMD] {' '.join(cmd)}")
    logger.debug(f"[ENV] 注入环境变量: {list(env_vars.keys())}")
    sys.stdout.flush()
    
    try:
        global current_process
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(script_path),
            env=full_env,
            stdout=None,  # 直接继承终端，支持tqdm原位进度条
            stderr=subprocess.STDOUT
        )
        current_process = process
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"脚本执行失败，退出码: {process.returncode}")
            return False
        else:
            logger.debug("脚本执行完成")
            return True
    except Exception as e:
        logger.error(f"执行异常: {e}")
        return False


def print_progress(step_num, total_steps, step_name, show_pct=True):
    """打印进度"""
    if show_pct:
        progress = step_num / total_steps * 100
        logger.info(f"[{step_num}/{total_steps}] {step_name} (进度: {progress:.1f}%)")
    else:
        logger.info(f"[{step_num}/{total_steps}] {step_name}")


def run_auxiliary_for_level(results_csv, level, args):
    """对指定层级的结果执行必要的辅助功能增强
    
    中间层（L1/L2）仅执行排序（影响unidentified MSP生成），
    其他辅助功能（Ontology/CCS/翻译/CSV关联）延迟到最终汇总后统一执行。
    """
    if not os.path.exists(results_csv):
        logger.info(f"{level}结果文件不存在，跳过辅助功能")
        return

    # 中间层仅执行排序（Top-K/去重影响 unidentified MSP 生成）
    logger.info(f"[{level}辅助功能] 结果排序...")
    sort_script = os.path.join(os.path.dirname(__file__), "辅助功能", "鉴定结果排序", "鉴定结果排序.py")
    top_k_value = L1_TOP_K if level == 'L1' else L2_TOP_K
    sort_args = ["--input", results_csv, "--output", results_csv, "--level", level, "--top_k", str(top_k_value)]
    run_script(sort_script, sort_args, "dreams")

    logger.info(f"[{level}辅助功能] 完成")


def main():
    """主函数"""
    args = parse_arguments()

    # 应用数据库开关（L1精确匹配 + L2大类前缀匹配）
    apply_database_switches()

    # 应用层级开关（命令行参数优先级更高）
    if not args.skip_l1:
        args.skip_l1 = not _on(RUN_L1)
    if not args.skip_l2:
        args.skip_l2 = not _on(RUN_L2)
    if not args.skip_l3:
        args.skip_l3 = not _on(RUN_L3)
    if not args.skip_l4:
        args.skip_l4 = not _on(RUN_L4)

    # 处理输入文件
    if not args.sample:
        logger.error("[ERROR] 请提供样品文件路径 (--sample)")
        return

    # 判断是否为单步骤模式（不显示进度百分比）
    single_step_mode = args.only_l1_vec or args.only_l1 or args.only_l2 or args.only_l3 or args.only_l4

    # 确定多层鉴定结果总目录
    if not args.output_dir:
        sample_dir = os.path.dirname(args.sample)
        sample_name = os.path.splitext(os.path.basename(args.sample))[0]
        args.output_dir = os.path.join(sample_dir, f"{sample_name}_多层鉴定结果")

    os.makedirs(args.output_dir, exist_ok=True)

    # 运行前清理：删除即将运行的层级的旧结果子目录（保留embeddings.npz样品缓存）
    def clean_level_dir(level_name):
        """清理指定层级的结果目录，但保留样品embedding缓存"""
        level_dir = os.path.join(args.output_dir, f"{level_name}_results")
        if not os.path.isdir(level_dir):
            return
        for item in os.listdir(level_dir):
            item_path = os.path.join(level_dir, item)
            # 保留样品embedding缓存（输入MSP不变则结果不变，重算需数分钟）
            if item == 'embeddings.npz':
                continue
            # 保留谱图缓存（sirius_project内部结构复杂，且不含旧格式残留）
            if item.startswith('sirius_project'):
                continue
            try:
                if os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                logger.warning(f"清理旧文件失败: {item_path} - {e}")
        logger.info(f"已清理 {level_name} 旧结果（保留embedding缓存）")

    levels_to_run = []
    if not args.skip_l1: levels_to_run.append('L1')  # L1_vec仅生成embedding，不触发L1结果清理
    if not args.skip_l2: levels_to_run.append('L2')
    if not args.skip_l3: levels_to_run.append('L3')
    if not args.skip_l4: levels_to_run.append('L4')
    for level in set(levels_to_run):
        clean_level_dir(level)
    # 清理根目录旧汇总文件
    import glob as _glob
    for old_file in _glob.glob(os.path.join(args.output_dir, '多层鉴定总结果.*')):
        try:
            os.remove(old_file)
        except Exception:
            pass
    for old_file in _glob.glob(os.path.join(args.output_dir, '*_多层鉴定总结果.*')):
        try:
            os.remove(old_file)
        except Exception:
            pass

    # 输出启动信息（终端和日志都显示）
    logger.info("=" * 60)
    logger.info("多层鉴定总控脚本（方法导向架构）")
    logger.info("=" * 60)
    logger.info(f"样品文件: {args.sample}")
    logger.info(f"离子模式: {args.ion_mode}")

    # 显示启用的数据库
    l1_db_names = list(L1_DATABASES.keys()) if L1_DATABASES else ["无"]
    l2_db_names = list(L2_SIMULATED_DATABASES.keys()) if L2_SIMULATED_DATABASES else ["无"]
    logger.info(f"L1数据库: {', '.join(l1_db_names)}")
    logger.info(f"L2数据库: {', '.join(l2_db_names)}")

    # 显示运行层级
    run_levels = []
    if not args.skip_l1: run_levels.append("L1")
    if not args.skip_l2: run_levels.append("L2")
    if not args.skip_l3: run_levels.append("L3")
    if not args.skip_l4: run_levels.append("L4")
    logger.info(f"运行层级: {' → '.join(run_levels) if run_levels else '无'}")

    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"日志文件: {LOG_FILE}")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    total_steps = 7
    current_step = 0

    # Embedding路径
    sample_emb_path = os.path.join(args.output_dir, "L1_results", "embeddings.npz")

    # ================================================
    # L1-预处理: 检查QI CSV文件（如果输入是MSP格式）
    # ================================================
    qi_csv_path = None
    if args.sample.lower().endswith('.msp'):
        qi_csv = args.sample.replace('.msp', '.csv')
        if os.path.exists(qi_csv):
            qi_csv_path = qi_csv
            logger.info(f"找到配套CSV文件: {qi_csv}")
        else:
            logger.info(f"未找到配套CSV文件，继续处理")

    # ================================================
    # L1-预处理: DreaMS 向量化
    # ================================================
    if not args.skip_l1_vec:
        current_step += 1
        print_progress(current_step, total_steps, "L1-预处理: 向量化", show_pct=not single_step_mode)

        sample_vec_script = os.path.join(os.path.dirname(__file__), "L1_实验数据库鉴定", "L1_DreaMS向量化.py")
        if not run_script_with_env(sample_vec_script, build_l1_sample_vec_env(args), "dreams"):
            logger.error("样品向量化失败，终止流程")
            return

        # 向量化L1数据库（如果有启用的数据库）
        l1_databases = get_l1_databases(args.ion_mode)
        if l1_databases:
            db_vec_script = os.path.join(os.path.dirname(__file__), "L1_实验数据库鉴定", "L1_DreaMS向量化.py")
            if not run_script_with_env(db_vec_script, build_l1_db_vec_env(args), "dreams"):
                logger.warning("数据库向量化失败，L1 双方法鉴定将在运行时自动计算")
        else:
            logger.info("未启用L1数据库，跳过数据库向量化")
    
    # ================================================
    # L1: 实验数据库鉴定（方法导向：matchMS → DreaMS → 整合）
    # 使用MSDIAL和SpectraVerse实验数据库，多库遍历，跨库合并Top-K
    # ================================================
    if not args.skip_l1:
        current_step += 1
        print_progress(current_step, total_steps, "L1: 实验数据库鉴定（matchMS + DreaMS）", show_pct=not single_step_mode)
        
        l1_output_dir = os.path.join(args.output_dir, "L1_results")
        l1_mc_results = os.path.join(l1_output_dir, "L1_matchMS_results.csv")
        l1_dreams_results = os.path.join(l1_output_dir, "L1_DreaMS_results.csv")
        
        mc_script = os.path.join(os.path.dirname(__file__), "L1_实验数据库鉴定", "L1_matchMS鉴定.py")
        if not run_script_with_env(mc_script, build_l1_mc_env(args, qi_csv_path), "matchms"):
            logger.error("L1 matchMS鉴定失败")
            return
        
        dreams_script = os.path.join(os.path.dirname(__file__), "L1_实验数据库鉴定", "L1_DreaMS鉴定.py")
        if not run_script_with_env(dreams_script, build_l1_dreams_env(args), "dreams"):
            logger.warning("L1 DreaMS鉴定失败，仅使用matchMS结果")
        
        l1_results_csv = os.path.join(l1_output_dir, "L1_results.csv")
        summary_script = os.path.join(os.path.dirname(__file__), "辅助功能", "各层鉴定结果汇总", "各层鉴定结果汇总.py")
        summary_args = ["--mode", "L1", "--mc_input", l1_mc_results, "--dreams_input", l1_dreams_results, "--output", l1_results_csv, "--sample_msp", args.sample]
        if not run_script(summary_script, summary_args, "dreams"):
            logger.error(" L1结果整合失败")
            return
        
        run_auxiliary_for_level(l1_results_csv, "L1", args)

        # 生成L1未鉴定MSP供L2使用
        logger.info("[L1] 生成未鉴定MSP...")
        l1_unidentified_msp = os.path.join(l1_output_dir, "L1_unidentified.msp")
        l1_unidentified_ok = False
        unidentified_args = ["--mode", "generate_unidentified", "--input", l1_results_csv, "--sample_msp", args.sample, "--output", l1_unidentified_msp, "--output_msp", l1_unidentified_msp]
        if run_script(summary_script, unidentified_args, "dreams"):
            l1_unidentified_ok = True
        else:
            logger.warning("生成未鉴定MSP失败，L2将处理全部样品")
            l1_unidentified_msp = args.sample
    
    # ================================================
    # L2: 模拟数据库鉴定（全量模拟库 + matchMS + DreaMS）
    # ================================================
    if not args.skip_l2:
        current_step += 1
        print_progress(current_step, total_steps, "L2: 模拟数据库鉴定（matchMS + DreaMS）", show_pct=not single_step_mode)

        # 确定L2输入MSP（提前定义，即使L2数据库为空也保留供L3回退）
        l2_input_msp = l1_unidentified_msp if 'l1_unidentified_msp' in locals() else args.sample

        # 检查是否有启用的L2数据库
        if not L2_SIMULATED_DATABASES:
            logger.warning("[L2] 未启用任何L2数据库，跳过L2鉴定")
        else:

            l2_output_dir = os.path.join(args.output_dir, "L2_results")
            l2_matchms_results = os.path.join(l2_output_dir, "L2_matchMS_results.csv")
            l2_dreams_results = os.path.join(l2_output_dir, "L2_DreaMS_results.csv")

            l2_sample_emb_path = sample_emb_path

            # 获取L2数据库配置
            l2_databases = get_l2_databases(args.ion_mode)

            if not l2_databases:
                logger.error(f"[L2] 未找到{args.ion_mode}模式的L2数据库")
                logger.error("[L2] 请检查L2数据库配置和文件是否存在")
            else:
                # 显示使用的数据库
                logger.info(f"[L2] 使用数据库: {', '.join(l2_databases.keys())}")

                # 向量化所有数据库（如果embedding不存在）
                vectorize_script = os.path.join(os.path.dirname(__file__), "L1_实验数据库鉴定", "L1_DreaMS向量化.py")
                vec_failed = False

                for db_name, db_info in l2_databases.items():
                    if not os.path.exists(db_info["emb"]):
                        logger.info(f"[L2] 生成{db_name}数据库embedding: {os.path.basename(db_info['msp'])}")
                        if not run_script_with_env(vectorize_script, build_l1_single_file_vec_env(db_info["msp"]), "dreams"):
                            logger.warning(f"[L2] {db_name}向量化失败")
                            vec_failed = True

                if vec_failed:
                    logger.warning("[L2] 部分数据库向量化失败，DreaMS鉴定可能不完整")

                # 各库独立匹配（与L1逻辑一致，不合并）
                l2_msp_libraries = {db_name: db_info["msp"] for db_name, db_info in l2_databases.items()}

                # 运行matchMS鉴定
                mc_script = os.path.join(os.path.dirname(__file__), "L2_模拟数据库鉴定", "L2_matchMS鉴定.py")

                l2_mc_env = {
                    'L1_SAMPLE_MSP': l2_input_msp,
                    'L1_ION_MODE': args.ion_mode,
                    'L1_MATCHMS_LIBRARIES': json_module.dumps(l2_msp_libraries),
                    'L1_MZ_TOLERANCE_PPM': str(L2_PRECURSOR_PPM),
                    'L1_FRAGMENT_TOLERANCE': str(L2_FRAGMENT_TOLERANCE),
                    'L1_MIN_MATCHED_PEAKS': str(L2_MIN_MATCHED_FRAGMENTS),
                    'L1_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
                    'L1_TOP_K': str(L2_TOP_K),
                    'L1_OUTPUT_DIR': l2_output_dir,
                }

                if not run_script_with_env(mc_script, l2_mc_env, "matchms"):
                    logger.error("[L2] matchMS鉴定失败")
                else:
                    # 重命名输出文件
                    l1_matchms_out = os.path.join(l2_output_dir, "L1_matchMS_results.csv")
                    if os.path.exists(l1_matchms_out):
                        os.rename(l1_matchms_out, l2_matchms_results)

                # 运行DreaMS鉴定
                dreams_script = os.path.join(os.path.dirname(__file__), "L2_模拟数据库鉴定", "L2_DreaMS鉴定.py")

                # 检查所有embedding是否存在
                all_emb_exist = all(os.path.exists(db_info["emb"]) for db_info in l2_databases.values())

                if all_emb_exist:
                    l2_dreams_env = {
                        'L1_SAMPLE_MSP': l2_input_msp,
                        'L1_SAMPLE_EMB': l2_sample_emb_path,
                        'L1_ION_MODE': args.ion_mode,
                        'L1_LIBRARIES': json_module.dumps(l2_databases),
                        'L1_MZ_TOLERANCE_PPM': str(L2_PRECURSOR_PPM),
                        'L1_FRAGMENT_TOLERANCE': str(L2_FRAGMENT_TOLERANCE),
                        'L1_MIN_MATCHED_PEAKS': str(L2_MIN_MATCHED_FRAGMENTS),
                        'L1_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
                        'L1_TOP_K': str(L2_TOP_K),
                        'L1_OUTPUT_DIR': l2_output_dir,
                    }

                    if not run_script_with_env(dreams_script, l2_dreams_env, "dreams"):
                        logger.warning("[L2] DreaMS鉴定失败，仅使用matchMS结果")
                    else:
                        l1_dreams_out = os.path.join(l2_output_dir, "L1_DreaMS_results.csv")
                        if os.path.exists(l1_dreams_out):
                            os.rename(l1_dreams_out, l2_dreams_results)
                else:
                    logger.warning("[L2] 部分数据库embedding缺失，跳过DreaMS鉴定")

                # 整合L2结果
                l2_results_csv = os.path.join(l2_output_dir, "L2_results.csv")
                summary_script = os.path.join(os.path.dirname(__file__), "辅助功能", "各层鉴定结果汇总", "各层鉴定结果汇总.py")
                summary_args = ["--mode", "L2", "--mc_input", l2_matchms_results, "--dreams_input", l2_dreams_results, "--output", l2_results_csv]
                if not run_script(summary_script, summary_args, "dreams"):
                    logger.error(" L2结果整合失败")
                else:
                    run_auxiliary_for_level(l2_results_csv, "L2", args)

                # 生成L2未鉴定MSP供L3/L4使用
                logger.info("[L2] 生成未鉴定MSP...")
                l2_unidentified_msp = os.path.join(l2_output_dir, "L2_unidentified.msp")
                unidentified_args = ["--mode", "generate_unidentified", "--input", l2_results_csv, "--sample_msp", l2_input_msp, "--output", l2_unidentified_msp, "--output_msp", l2_unidentified_msp]
                if not run_script(summary_script, unidentified_args, "dreams"):
                    logger.warning("生成L2未鉴定MSP失败，L3将处理全部样品")
                    l2_unidentified_msp = l2_input_msp

    # ================================================
    # L3: SIRIUS结构鉴定
    # ================================================
    if not args.skip_l3:
        current_step += 1
        print_progress(current_step, total_steps, "L3: SIRIUS结构鉴定", show_pct=not single_step_mode)

        l3_output_dir = os.path.join(args.output_dir, "L3_results")
        os.makedirs(l3_output_dir, exist_ok=True)

        # 确定L3输入MSP（L2未鉴定 → L2输入 → L1未鉴定 → 原始样品）
        l3_input_msp = args.sample
        if 'l2_unidentified_msp' in locals() and os.path.exists(l2_unidentified_msp) and os.path.getsize(l2_unidentified_msp) > 0:
            l3_input_msp = l2_unidentified_msp
        elif 'l2_input_msp' in locals() and os.path.exists(l2_input_msp) and os.path.getsize(l2_input_msp) > 0:
            l3_input_msp = l2_input_msp
            logger.info(f"[L3] L2未鉴定不可用，使用L2输入MSP")
        elif 'l1_unidentified_ok' in locals() and l1_unidentified_ok and os.path.exists(l1_unidentified_msp) and os.path.getsize(l1_unidentified_msp) > 0:
            l3_input_msp = l1_unidentified_msp
            logger.info(f"[L3] 使用L1未鉴定MSP作为输入（已过滤L1鉴定结果）")
        else:
            logger.warning(f"[L3] ⚠ 无可用的未鉴定MSP，使用原始样品（全量谱图），建议先运行L1/L2鉴定")

        # L3固定使用 BIO 在线数据库，不再加载本地库
        all_databases = list(L3_ONLINE_DATABASES)

        if not all_databases:
            logger.warning("[L3] 未配置任何数据库，SIRIUS结构搜索将无数据库可用")

        # L3参数校验（检查参数是否变化，自动决定续传/重置）
        param_check_script = os.path.join(os.path.dirname(__file__), "辅助功能", "L3参数续传校验", "L3参数续传校验.py")
        param_check_args = [
            "--mode", "check",
            "--output_dir", l3_output_dir,
            "--sample_msp", l3_input_msp,
            "--ion_mode", args.ion_mode,
            "--instrument", L3_SIRIUS_INSTRUMENT,
            "--mz_threshold", str(L3_MAX_MZ),
            "--min_peaks", str(L3_MIN_PEAKS),
            "--databases",
        ] + all_databases
        if qi_csv_path and L3_MIN_INTENSITY > 0:
            param_check_args += ["--sample_csv", qi_csv_path, "--min_intensity", str(L3_MIN_INTENSITY)]
        run_script(param_check_script, param_check_args, "dreams")

        # SIRIUS结构鉴定
        l3_script = os.path.join(os.path.dirname(__file__), "L3_SIRIUS结构鉴定", "L3_SIRIUS_CLI.py")
        cmd = [
            sys.executable, l3_script,
            "--sample_msp", l3_input_msp,
            "--output_dir", l3_output_dir,
            "--instrument", L3_SIRIUS_INSTRUMENT,
            "--ion_mode", args.ion_mode,
            "--min_peaks", str(L3_MIN_PEAKS),
            "--mz_threshold", str(L3_MAX_MZ),
        ]
        # 信号强度过滤（需配套 QI CSV）
        if qi_csv_path and L3_MIN_INTENSITY > 0:
            cmd += ["--sample_csv", qi_csv_path, "--min_intensity", str(L3_MIN_INTENSITY)]
        cmd += ["--databases"] + all_databases

        logger.info(f"[L3] 启动 SIRIUS CLI（详细参数见下方 banner）")

        try:
            result = subprocess.run(cmd, text=True)

            if result.returncode != 0:
                logger.error(f"[L3] 运行失败，返回码: {result.returncode}")
            else:
                logger.info(f"[L3] 运行成功")
                # 保存L3参数（供下次运行时比对续传）
                param_save_args = [
                    "--mode", "save",
                    "--output_dir", l3_output_dir,
                    "--sample_msp", l3_input_msp,
                    "--ion_mode", args.ion_mode,
                    "--instrument", L3_SIRIUS_INSTRUMENT,
                    "--mz_threshold", str(L3_MAX_MZ),
                    "--min_peaks", str(L3_MIN_PEAKS),
                    "--databases",
                ] + all_databases
                if qi_csv_path and L3_MIN_INTENSITY > 0:
                    param_save_args += ["--sample_csv", qi_csv_path, "--min_intensity", str(L3_MIN_INTENSITY)]
                run_script(param_check_script, param_save_args, "dreams")
        except Exception as e:
            logger.error(f"[L3] 运行异常: {e}")

        # 生成L3未鉴定MSP供L4使用
        l3_identified_csv = os.path.join(l3_output_dir, "L3_identified.csv")
        l3_unidentified_msp = None
        if os.path.exists(l3_identified_csv):
            logger.info("[L3] 生成未鉴定MSP...")
            l3_unidentified_msp = os.path.join(l3_output_dir, "L3_unidentified.msp")
            summary_script_path = os.path.join(os.path.dirname(__file__), "辅助功能", "各层鉴定结果汇总", "各层鉴定结果汇总.py")
            unidentified_args = ["--mode", "generate_unidentified", "--input", l3_identified_csv, "--sample_msp", l3_input_msp, "--output", l3_unidentified_msp, "--output_msp", l3_unidentified_msp]
            if not run_script(summary_script_path, unidentified_args, "dreams"):
                logger.warning("生成L3未鉴定MSP失败，L4将处理L2未鉴定样品")
                l3_unidentified_msp = None
    
    # ================================================
    # L4: DreaMS 分子网络（纯样品间 Embedding 互连）
    # ================================================
    if not args.skip_l4:
        current_step += 1
        print_progress(current_step, total_steps, "L4: DreaMS 分子网络", show_pct=not single_step_mode)

        l4_script = os.path.join(os.path.dirname(__file__), "L4_DreaMS分子网络", "L4_DreaMS分子网络.py")

        l4_env = build_l4_analog_env(args, sample_emb_path)

        if not run_script_with_env(l4_script, l4_env, "dreams"):
            logger.error(" L4 DreaMS分子网络失败")
            return

        l4_results_csv = os.path.join(args.output_dir, "L4_results", "L4_network_edges.csv")

    # ================================================
    # 最终汇总
    # ================================================
    final_output = os.path.join(args.output_dir, "多层鉴定总结果.csv")
    if not args.skip_summary:
        current_step += 1
        print_progress(current_step, total_steps, "最终结果汇总", show_pct=not single_step_mode)
        
        summary_script = os.path.join(os.path.dirname(__file__), "辅助功能", "各层鉴定结果汇总", "各层鉴定结果汇总.py")
        
        l1_results = os.path.join(args.output_dir, "L1_results", "L1_results.csv")
        l2_results = os.path.join(args.output_dir, "L2_results", "L2_results.csv")
        l3_results = os.path.join(args.output_dir, "L3_results", "L3_identified.csv")
        l4_results = os.path.join(args.output_dir, "L4_results", "L4_network_nodes.csv")

        summary_cmd = [
            sys.executable, summary_script,
            "--mode", "final",
            "--input", args.output_dir,
            "--output", final_output,
            "--l1_results", l1_results,
            "--l2_results", l2_results,
            "--l3_results", l3_results,
            "--l4_results", l4_results,
            "--sample_msp", args.sample,
            "--ion_mode", args.ion_mode,
        ]
        
        try:
            result = subprocess.run(summary_cmd, capture_output=True, text=True)
            # 打印汇总脚本的stdout
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    print(line)
            if result.returncode != 0:
                logger.warning(f"最终汇总失败(退出码{result.returncode}): {result.stderr or result.stdout}")
        except Exception as e:
            logger.warning(f"最终汇总执行异常: {e}")

    # ================================================
    # 辅助功能（汇总后统一执行，避免中间层重复）
    # ================================================
    if os.path.exists(final_output):
        aux_functions = ["CSV关联", "同位素相似度", "Ontology", "CCS", "翻译"]
        logger.info(f"[最终汇总] 统一执行辅助功能: {' + '.join(aux_functions)}...")

        # 1. CSV关联（峰面积等原始数据）
        qi_csv_path = getattr(args, 'qi_csv', None)
        if not qi_csv_path and hasattr(args, 'sample') and args.sample:
            sample_base = os.path.splitext(args.sample)[0]
            potential_csv = sample_base + '.csv'
            if os.path.exists(potential_csv):
                qi_csv_path = potential_csv
        if qi_csv_path and os.path.exists(qi_csv_path):
            csv_link_script = os.path.join(os.path.dirname(__file__), "辅助功能", "原始数据CSV关联", "原始数据CSV关联.py")
            if not run_script(csv_link_script, ["--input", final_output, "--sample_csv", qi_csv_path, "--output", final_output], "dreams"):
                logger.warning("CSV数据关联失败")

        # 2. 同位素相似度计算（需先有 CSV关联 提供的 Isotope Distribution 列）
        isotope_script = os.path.join(os.path.dirname(__file__), "辅助功能", "同位素相似度计算", "同位素相似度计算.py")
        run_script(isotope_script, ["--input", final_output, "--output", final_output], "dreams")

        # 3. Ontology分类获取（优先本轮 SIRIUS CANOPUS TSV，否则 SIRIUS 持久化缓存，最后实时 CANOPUS）
        ontology_script = os.path.join(os.path.dirname(__file__), "辅助功能", "ontology获取", "ontology获取.py")
        ontology_args = [
            "--input_csv", final_output, "--output_csv", final_output, "--field", "class",
            "--canopus_fallback", "--sample_msp", args.sample,
            "--ion_mode", args.ion_mode, "--instrument", L3_SIRIUS_INSTRUMENT,
        ]
        l3_sirius_results_dir = os.path.join(args.output_dir, "L3_results", "sirius_results")
        if os.path.isdir(l3_sirius_results_dir):
            ontology_args += ["--sirius_results_dir", l3_sirius_results_dir]
            logger.info(f"[Ontology] 检测到 L3 CANOPUS 输出目录，将优先从 TSV 填充 ontology：{l3_sirius_results_dir}")
        run_script(ontology_script, ontology_args, "dreams")

        # 3. CCS预测
        ccsbase2_script = os.path.join(os.path.dirname(__file__), "辅助功能", "CCS预测", "ccsbase2预测.py")
        run_script(ccsbase2_script, [final_output, final_output], "dreams")

        # 4. 翻译（名称 + Ontology）
        translate_script = os.path.join(os.path.dirname(__file__), "辅助功能", "小牛中文翻译", "小牛中文翻译.py")
        run_script(translate_script, [os.path.dirname(final_output), "--input", final_output, "--output", final_output], "dreams")

        # 5. 辅助功能完成后，从CSV重新生成Excel（含CCS/翻译等全部数据）
        try:
            import pandas as pd
            import re
            import importlib.util
            _summ_path = os.path.join(os.path.dirname(__file__), "辅助功能", "各层鉴定结果汇总", "各层鉴定结果汇总.py")
            _summ_spec = importlib.util.spec_from_file_location("各层鉴定结果汇总", _summ_path)
            _summ = importlib.util.module_from_spec(_summ_spec)
            _summ_spec.loader.exec_module(_summ)
            get_column_rename_map = _summ.get_column_rename_map
            get_excel_filename_from_msp = _summ.get_excel_filename_from_msp
            get_sheet_name_from_msp = _summ.get_sheet_name_from_msp
            format_excel_output = _summ.format_excel_output
            collapse_shared_query_columns = _summ.collapse_shared_query_columns
            df = pd.read_csv(final_output)
            
            # ---- 数据清理 ----
            # a) 删除全空列（含NaN + 空字符串 + 纯空白）
            def _is_effectively_empty(col):
                """判断列是否为空（NaN 或 空字符串 或 'nan' 等）"""
                return col.isna().all() or col.astype(str).str.strip().isin(['', 'nan', 'NaN']).all()
            empty_cols = [c for c in df.columns if _is_effectively_empty(df[c])]
            if empty_cols:
                df = df.drop(columns=empty_cols, errors='ignore')
            
            # b) 删除formula列（L3预测分子式）—— 最终汇总里与matched_formula永远冗余
            if 'formula' in df.columns and 'matched_formula' in df.columns:
                df = df.drop(columns=['formula'], errors='ignore')
            
            # c) CCS组合列已由 CCS预测 辅助脚本在CSV中生成，此处不再重复计算
            #    仅清理预测过程产生的原始数值列（保留已格式化好的 ccs_combined）
            drop_cols = ['predicted_ccs', 'predicted_ccs_deviation_pct', 'CCS (angstrom^2)', 'CCS_error',
                         'Isotope Distribution', 'measured_ccs',
                         'precursor_mz', 'Retention time (min)',
                         'zodiac_score', 'library_precursor_mz', 'precursor_ppm_diff']
            for col in drop_cols:
                if col in df.columns:
                    df = df.drop(columns=[col], errors='ignore')
            
            # d) 加合物类型去空格（SIRIUS 输出 "[M + H]+"，统一为 "[M+H]+"）
            for col in ['adduct', '加合物类型']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(' ', '', regex=False)
            
            rename_map = get_column_rename_map('final')
            rename_map.update(get_column_rename_map('L2'))
            df_cn = df.rename(columns=rename_map)
            # 同一 query_name 多候选时合并共享列（样品化合物、CCS 等只显示一次）
            df_cn = collapse_shared_query_columns(df_cn, query_col='样品化合物')
            wrap_columns_cn = [rename_map.get(col, col) for col in "matched_name,matched_ontology,matched_fragments".split(',') if col]
            xlsx_path = get_excel_filename_from_msp(args.sample, final_output, "final", ion_mode=args.ion_mode)
            sheet_name = get_sheet_name_from_msp(args.sample)
            format_excel_output(df_cn, xlsx_path, wrap_columns=','.join(wrap_columns_cn), sheet_name=sheet_name)
            logger.info(f"[最终汇总] Excel已更新（含辅助功能数据）: {xlsx_path}")
        except Exception as e:
            logger.warning(f"Excel重新生成失败: {e}")

    # ================================================
    # 完成
    # ================================================
    elapsed_time = datetime.now() - start_time
    
    if args.only_l1_vec:
        completion_msg = "L1 向量化完成"
    elif args.only_l1:
        completion_msg = "L1 数据库鉴定完成"
    elif args.only_l2:
        completion_msg = "L2 模拟库鉴定完成"
    elif args.only_l3:
        completion_msg = "L3 SIRIUS结构鉴定完成"
    elif args.only_l4:
        completion_msg = "L4 DreaMS分子网络完成"
    else:
        completion_msg = "多层鉴定流程完成"
    
    logger.info("\n" + "=" * 60)
    logger.info(f"{completion_msg}! 总耗时: {elapsed_time}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
