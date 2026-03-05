#!/usr/bin/env python3
"""
多层鉴定总控脚本（方法导向架构）

统一管理所有参数配置，按顺序调用各层级子脚本：
L1_DreaMS向量化 → L1_MC匹配 → L1_DreaMS匹配 → L1_整合 → L2_MC匹配 → L2_DreaMS匹配 → L2_整合 → L3_DreaMS类似物筛查 → L4a/L4b_SIRIUS从头鉴定

架构说明：
- L1/L2均采用方法导向：MC方法 + DreaMS方法
- MC方法：ModifiedCosine匹配
- DreaMS方法：Embedding cosine匹配
- L3：纯Embedding类似物筛查（无母离子/碎片约束）
- L4：SIRIUS从头鉴定，分L4a(类似物优先)和L4b(未知物)

使用方式：
    python 总控脚本.py --sample_msp 样品.msp --ion_mode POS
"""

###############################################################################
# ==================== 参数配置区（总控脚本统一管理） ====================
###############################################################################
# 所有参数在此区域集中配置，子脚本不再硬编码任何参数
###############################################################################

# ============================================================
# 一、基础配置
# ============================================================

# ----- 样品数据 -----
SAMPLE_MSP = "/stor3/20260204_LIHUA/POS/LIHUA_POS.msp"
SAMPLE_CSV = "/stor3/20260204_LIHUA/POS/LIHUA_POS.csv"
ION_MODE = "POS"  # 离子模式: "POS" 或 "NEG"

# ----- 输出设置 -----
OUTPUT_BASE_DIR = ""  # 输出根目录（留空则使用样品文件同级目录）

# ============================================================
# 二、L1 真实数据库匹配参数
# ============================================================

# ----- L1 数据库配置 -----
# 结构: {库名: {离子模式: MSP路径}}
L1_DATABASES = {
    "MSDIAL": {
        "POS": "/stor3/AIMS4Meta/数据库/MSDIAL/MSMS-Public_all-pos-VS19.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/MSDIAL/MSMS-Public_all-neg-VS19.msp",
    },
    "SpectraTraverse": {
        "POS": "/stor3/AIMS4Meta/数据库/spectraverse/spectraverse-1.0.1-pos.msp",
        "NEG": "/stor3/AIMS4Meta/数据库/spectraverse/spectraverse-1.0.1-neg.msp",
    },
}

# ----- L1 匹配参数 -----
L1_PRECURSOR_PPM = 10.0            # 母离子容差 (ppm)
L1_FRAGMENT_TOLERANCE = 0.05       # 碎片容差 (Da)
L1_MIN_MATCHED_FRAGMENTS = 2       # 最少匹配碎片数
L1_COSINE_THRESHOLD = 0.5          # 余弦相似度阈值
L1_TOP_K = 5                       # Top-K 结果数

# ----- L1 DreaMS 向量化参数 -----
L1_BATCH_SIZE = 64              # DreaMS 批处理大小

# ============================================================
# 三、L2 模拟数据库匹配参数
# ============================================================

# ----- 匹配参数 -----
L2_PRECURSOR_PPM = L1_PRECURSOR_PPM          # 母离子容差 (ppm)
L2_FRAGMENT_TOLERANCE = L1_FRAGMENT_TOLERANCE  # 碎片容差 (Da)
L2_MIN_MATCHED_FRAGMENTS = L1_MIN_MATCHED_FRAGMENTS  # 最少匹配碎片数
L2_COSINE_THRESHOLD = L1_COSINE_THRESHOLD  # 余弦相似度阈值
L2_TOP_K_RESULTS = L1_TOP_K        # Top-K 结果数

# ----- 候选库配置 -----
L2_CANDIDATE_LIBRARY = {
    "POS": "/stor3/AIMS4Meta/数据库/FDA_approved_drugs+drugbank/positive",
    "NEG": "/stor3/AIMS4Meta/数据库/FDA_approved_drugs+drugbank/negative"
}

# ----- 预测工具配置 -----
L2_CFMID_DIR = "/stor3/AIMS4Meta/源代码/cfm-id-code"
L2_CFMID_WORKERS = 5               # CFM-ID 并行工作进程数
L2_FIORA_DEVICE = "cuda:0"         # FIORA GPU 设备
L2_FORCE_REGENERATE = False        # 强制重新生成模拟库（忽略版本标签缓存）

# ============================================================
# 四、L3 DreaMS 类似物筛查参数
# ============================================================

L3_SIM_THRESHOLD = 0.4              # DreaMS Embedding 余弦相似度阈值（L3不限制母离子偏差，需更严格）
L3_FRAGMENT_TOLERANCE = L1_FRAGMENT_TOLERANCE  # 碎片匹配容差 (Da)
L3_MIN_MATCHED_PEAKS = L1_MIN_MATCHED_FRAGMENTS  # 最少匹配碎片数（低于此值的结果过滤掉）
L3_MIN_PRECURSOR_PPM = L2_PRECURSOR_PPM  # 母离子最小偏差 (ppm)（≤此值的排除，与L2精确匹配互斥）

# ----- L3 候选库选择 -----
L3_USE_MSDIAL = False               # 使用MSDIAL真实库（关闭）
L3_USE_SPECTRAVERSE = False         # 使用SpectraTraverse真实库（关闭）
L3_USE_SIMULATED = True             # 使用L2模拟库（FIORA+CFM-ID）

# ============================================================
# 五、L4 从头鉴定参数
# ============================================================

# ----- 仪器配置 -----
L4_INSTRUMENT_TYPE = "tof"         # 仪器类型: "tof" (Q-TOF, 10ppm) 或 "orbi" (Orbitrap, 5ppm)

# ----- 分析参数 -----
L4_MZ_THRESHOLD = 1500             # m/z 阈值（仅处理 m/z 小于此值的化合物）
L4_DATABASE_CHOICE = "ALL"         # 数据库选择: "BIO", "ALL", "PUBCHEM" 等
L4_EXPANSIVE_SEARCH = "APPROXIMATE" # 扩展搜索模式: "APPROXIMATE"（推荐）, "EXACT", "OFF"

# ----- SIRIUS 配置 -----
L4_SIRIUS_BIN = "/stor3/AIMS4Meta/源代码/SIRIUS/sirius-6.3.3-linux-x64/sirius/bin/sirius"
L4_SIRIUS_CORES = 8                # CPU 工作线程数（降低以避免 CSI:FingerID 云端 API 限流超时）
L4_SIRIUS_BUFFER = 0               # 化合物预加载缓冲（0=自动）

# ----- Formula 分析参数 -----
L4_FORMULA_CANDIDATES = 10         # 候选分子式数
L4_FORMULA_CANDIDATES_PER_ION = 1  # 每种离子态的候选数
L4_DETECTABLE_ELEMENTS = "SClBrSe"  # 可检测元素（移除 B,Fe,Zn,Mg 避免预测出异常分子式）
L4_ENFORCED_ELEMENTS = "CHNOP"     # 强制元素
L4_FIX_LIPIDS = "True"             # 脂质修正

# ============================================================
# 六、结果汇总参数
# ============================================================

SUMMARY_OUTPUT_FORMAT = "CSV"     # 汇总输出格式: "xlsx" 或 "csv"
SUMMARY_INCLUDE_L1 = True          # 是否包含 L1 结果
SUMMARY_INCLUDE_L2 = True          # 是否包含 L2 结果
SUMMARY_INCLUDE_L3 = True          # 是否包含 L3 结果
SUMMARY_INCLUDE_L4 = True          # 是否包含 L4 结果

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

def get_l1_databases(ion_mode):
    """根据 L1_DATABASES 自动生成完整的数据库配置（含 embedding 缓存路径）"""
    databases = {}
    for db_name, ion_mode_paths in L1_DATABASES.items():
        msp_path = ion_mode_paths.get(ion_mode)
        if not msp_path:
            continue
        emb_path = msp_path.replace('.msp', '_dreams_emb.npz')
        databases[db_name] = {"msp": msp_path, "emb": emb_path}
    return databases


def build_l1_sample_vec_env(args):
    """构建 L1 样品向量化环境变量"""
    return {
        'L1_MSP_FILE': args.sample_msp,
        'L1_OUTPUT_DIR': os.path.join(args.output_dir, "L1_results"),
    }

def build_l1_db_vec_env(args):
    """构建 L1 数据库向量化环境变量"""
    l1_databases = get_l1_databases(args.ion_mode)
    db_paths = {db_name: db_info["msp"] for db_name, db_info in l1_databases.items()}
    
    return {
        'L1_DATABASES': json_module.dumps({args.ion_mode: db_paths}),
        'L1_ION_MODE': args.ion_mode,
    }

def build_l1_env(args):
    """构建 L1 公共环境变量"""
    return {
        'L1_SAMPLE_MSP': args.sample_msp,
        'L1_ION_MODE': args.ion_mode,
        'L1_MZ_TOLERANCE_PPM': str(int(L1_PRECURSOR_PPM)),
        'L1_FRAGMENT_TOLERANCE': str(L1_FRAGMENT_TOLERANCE),
        'L1_MIN_MATCHED_PEAKS': str(L1_MIN_MATCHED_FRAGMENTS),
        'L1_COSINE_THRESHOLD': str(L1_COSINE_THRESHOLD),
        'L1_TOP_K': str(L1_TOP_K),
        'L1_OUTPUT_DIR': os.path.join(args.output_dir, "L1_results"),
    }

def build_l1_mc_env(args):
    """构建 L1 MC 匹配环境变量"""
    l1_databases = get_l1_databases(args.ion_mode)
    mc_libraries = {db_name: db_paths["msp"] for db_name, db_paths in l1_databases.items()}
    env = build_l1_env(args)
    env['L1_MC_LIBRARIES'] = json_module.dumps(mc_libraries)
    return env

def build_l1_dreams_env(args):
    """构建 L1 DreaMS 匹配环境变量"""
    sample_emb_path = os.path.join(args.output_dir, "L1_results", "embeddings.npz")
    l1_output_dir = os.path.join(args.output_dir, "L1_results")
    env = build_l1_env(args)
    env['L1_SAMPLE_EMB'] = sample_emb_path
    env['L1_LIBRARIES'] = json_module.dumps(get_l1_databases(args.ion_mode))
    env['L1_OUTPUT_DIR'] = l1_output_dir
    return env

def build_l2_fiora_env(args):
    """构建 L2 FIORA 预测环境变量"""
    return {
        'L2_ION_MODE': args.ion_mode,
        'L2_OUTPUT_DIR': os.path.join(args.output_dir, "L2_results"),
        'L2_CANDIDATE_LIBRARY': L2_CANDIDATE_LIBRARY[args.ion_mode],
        'L2_FIORA_DEVICE': L2_FIORA_DEVICE,
        'L2_FORCE_REGENERATE': '1' if L2_FORCE_REGENERATE else '0',
    }

def build_l2_cfmid_env(args):
    """构建 L2 CFM-ID 预测环境变量"""
    return {
        'L2_ION_MODE': args.ion_mode,
        'L2_OUTPUT_DIR': os.path.join(args.output_dir, "L2_results"),
        'L2_CANDIDATE_LIBRARY': L2_CANDIDATE_LIBRARY[args.ion_mode],
        'L2_CFMID_DIR': L2_CFMID_DIR,
        'L2_CFMID_WORKERS': str(L2_CFMID_WORKERS),
        'L2_FORCE_REGENERATE': '1' if L2_FORCE_REGENERATE else '0',
    }

def build_l2_match_env(args, sample_emb_path, simulated_msp=None, simulated_emb=None):
    """构建 L2 匹配环境变量（MC/DreaMS 匹配共用）"""
    env = {
        'L2_L1_UNIDENTIFIED_MSP': args.sample_msp,  # L2 直接处理全部原始样品
        'L2_ION_MODE': args.ion_mode,
        'L2_OUTPUT_DIR': os.path.join(args.output_dir, "L2_results"),
        'L2_CANDIDATE_LIBRARY': L2_CANDIDATE_LIBRARY[args.ion_mode],
        'L2_PRECURSOR_PPM': str(L2_PRECURSOR_PPM),
        'L2_FRAGMENT_TOLERANCE': str(L2_FRAGMENT_TOLERANCE),
        'L2_MIN_MATCHED_FRAGMENTS': str(L2_MIN_MATCHED_FRAGMENTS),
        'L2_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
        'L2_MC_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
        'L2_TOP_K_RESULTS': str(L2_TOP_K_RESULTS),
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
        'L1_FORCE_COMPUTE': '1' if L2_FORCE_REGENERATE else '0',
    }
    return env

def build_l3_analog_env(args, sample_emb_path,
                       simulated_msp=None, simulated_emb=None):
    """构建 L3 DreaMS 类似物筛查环境变量
    
    候选库选择（通过命令行参数覆盖配置区默认值）：
    - --l3_use_msdial: 使用MSDIAL真实库
    - --l3_use_spectraverse: 使用SpectraTraverse真实库  
    - --l3_use_simulated: 使用L2模拟库
    """
    # 获取候选库选择（命令行参数优先，否则用配置区默认值）
    use_msdial = args.l3_use_msdial if args.l3_use_msdial is not None else L3_USE_MSDIAL
    use_spectraverse = args.l3_use_spectraverse if args.l3_use_spectraverse is not None else L3_USE_SPECTRAVERSE
    use_simulated = args.l3_use_simulated if args.l3_use_simulated is not None else L3_USE_SIMULATED
    
    # 构建真实库列表
    real_db_libraries = {}
    if use_msdial or use_spectraverse:
        all_dbs = get_l1_databases(args.ion_mode)
        if use_msdial and 'MSDIAL' in all_dbs:
            real_db_libraries['MSDIAL'] = all_dbs['MSDIAL']
        if use_spectraverse and 'SpectraTraverse' in all_dbs:
            real_db_libraries['SpectraTraverse'] = all_dbs['SpectraTraverse']
    
    env = {
        'L3_OUTPUT_DIR': os.path.join(args.output_dir, "L3_results"),
        'L3_SAMPLE_EMB': sample_emb_path,
        'L3_SAMPLE_MSP': args.sample_msp,  # L3 也直接处理全部原始样品
        'L3_SIM_THRESHOLD': str(L3_SIM_THRESHOLD),
        'L3_FRAGMENT_TOLERANCE': str(L3_FRAGMENT_TOLERANCE),
        'L3_MIN_MATCHED_PEAKS': str(L3_MIN_MATCHED_PEAKS),
        'L3_MIN_PRECURSOR_PPM': str(L3_MIN_PRECURSOR_PPM),
        'L3_REAL_DB_LIBRARIES': json_module.dumps(real_db_libraries),
    }
    # 仅当启用模拟库且路径有效时才传递
    if use_simulated:
        if simulated_msp:
            env['L3_SIMULATED_MSP'] = simulated_msp
        if simulated_emb:
            env['L3_SIMULATED_EMB'] = simulated_emb
    return env

def build_l4_env(args):
    """构建 L4 环境变量（新策略：处理整个原始数据）"""
    return {
        'L4_SAMPLE_MSP': args.sample_msp,
        'L4_SAMPLE_CSV': args.sample_csv if args.sample_csv else '',
        'L4_OUTPUT_DIR': os.path.join(args.output_dir, "L4_results"),
        'L4_ION_MODE': args.ion_mode,
        'L4_MZ_THRESHOLD': str(L4_MZ_THRESHOLD),
    }

def build_summary_env(args):
    """构建结果汇总环境变量"""
    return {
        'SUMMARY_OUTPUT_DIR': args.output_dir,
        'SUMMARY_SAMPLE_MSP': args.sample_msp,
        'SUMMARY_OUTPUT_FORMAT': SUMMARY_OUTPUT_FORMAT,
        'SUMMARY_INCLUDE_L1': str(SUMMARY_INCLUDE_L1),
        'SUMMARY_INCLUDE_L2': str(SUMMARY_INCLUDE_L2),
        'SUMMARY_INCLUDE_L3': str(SUMMARY_INCLUDE_L3),
        'SUMMARY_INCLUDE_L4': str(SUMMARY_INCLUDE_L4),
    }


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多层鉴定总控脚本（方法导向）")
    
    # 基础参数
    parser.add_argument("--sample_msp", default=SAMPLE_MSP, help="样品MSP文件路径")
    parser.add_argument("--ion_mode", default=ION_MODE, choices=["POS", "NEG"], help="离子模式")
    parser.add_argument("--sample_csv", default=SAMPLE_CSV, help="原始样本CSV文件")
    parser.add_argument("--output_dir", default=OUTPUT_BASE_DIR, help="输出根目录")
    
    # 控制参数
    parser.add_argument("--skip_l1_vec", action="store_true", help="跳过L1向量化")
    parser.add_argument("--skip_l1", action="store_true", help="跳过L1")
    parser.add_argument("--skip_l2", action="store_true", help="跳过L2")
    parser.add_argument("--skip_l3", action="store_true", help="跳过L3 DreaMS类似物筛查")
    parser.add_argument("--skip_l4", action="store_true", help="跳过L4 SIRIUS从头鉴定")
    # 快捷参数
    parser.add_argument("--only_l1_vec", action="store_true", help="仅运行L1向量化")
    parser.add_argument("--only_l1", action="store_true", help="仅运行L1")
    parser.add_argument("--only_l2", action="store_true", help="仅运行L2")
    parser.add_argument("--only_l3", action="store_true", help="仅运行L3 DreaMS类似物筛查")
    parser.add_argument("--only_l4", action="store_true", help="仅运行L4 SIRIUS从头鉴定")
    
    # L3 候选库选择参数（覆盖配置区默认值）
    parser.add_argument("--l3_use_msdial", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, help="L3使用MSDIAL真实库 (true/false)")
    parser.add_argument("--l3_use_spectraverse", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, help="L3使用SpectraTraverse真实库 (true/false)")
    parser.add_argument("--l3_use_simulated", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, help="L3使用L2模拟库 (true/false)")
    
    args = parser.parse_args()
    
    # 处理快捷参数
    if args.only_l1_vec:
        args.skip_l1 = args.skip_l2 = args.skip_l3 = args.skip_l4 = True
        args.skip_summary = True
    elif args.only_l1:
        args.skip_l2 = args.skip_l3 = args.skip_l4 = True
        args.skip_summary = True
    elif args.only_l2:
        args.skip_l1_vec = args.skip_l1 = args.skip_l3 = args.skip_l4 = True
        args.skip_summary = True
    elif args.only_l3:
        args.skip_l1_vec = args.skip_l1 = args.skip_l2 = args.skip_l4 = True
        args.skip_summary = True
    elif args.only_l4:
        args.skip_l1_vec = args.skip_l1 = args.skip_l2 = args.skip_l3 = True
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
            env=full_env
        )
        current_process = process  # 记录当前进程,用于信号处理
        
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
    logger.info(f"\n{'='*60}")
    if show_pct:
        progress = step_num / total_steps * 100
        logger.info(f"[{step_num}/{total_steps}] {step_name} (进度: {progress:.1f}%)")
    else:
        logger.info(f"{step_name}")
    logger.info(f"{'='*60}")


def run_auxiliary_for_level(results_csv, level, args):
    """对指定层级的结果执行所有辅助功能增强"""
    if not os.path.exists(results_csv):
        logger.info(f"{level}结果文件不存在，跳过辅助功能")
        return
    
    logger.info(f"[{level}辅助功能] Ontology + CSV关联 + 同位素 + CCS + 翻译...")
    
    # 1. Ontology分类获取
    ontology_script = os.path.join(os.path.dirname(__file__), "辅助功能", "ontology获取", "ontology获取.py")
    ontology_args = ["--input_csv", results_csv, "--output_csv", results_csv, "--field", "class"]
    run_script(ontology_script, ontology_args, "dreams")
    
    # 2. 原始数据CSV关联
    csv_link_script = os.path.join(os.path.dirname(__file__), "辅助功能", "原始数据CSV关联", "原始数据CSV关联.py")
    csv_link_args = ["--input", results_csv, "--sample_csv", args.sample_csv, "--output", results_csv]
    run_script(csv_link_script, csv_link_args, "dreams")
    
    # 3. 同位素相似度计算
    isotope_script = os.path.join(os.path.dirname(__file__), "辅助功能", "同位素相似度计算", "同位素相似度计算.py")
    isotope_args = ["--input", results_csv, "--output", results_csv, "--formula_col", "matched_formula", "--isotope_dist_col", "isotope_distribution"]
    run_script(isotope_script, isotope_args, "dreams")
    
    # 4. CCS预测
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp(prefix="ccs_temp_")
    
    sigma_script = os.path.join(os.path.dirname(__file__), "辅助功能", "CCS预测", "sigmaCCS预测.py")
    sigma_output = os.path.join(temp_dir, "sigmaCCS.csv")
    run_script(sigma_script, ["--input", results_csv, "--output", sigma_output], "sigma")
    
    ccsbase_script = os.path.join(os.path.dirname(__file__), "辅助功能", "CCS预测", "ccsbase预测.py")
    ccsbase_output = os.path.join(temp_dir, "ccsbase.csv")
    run_script(ccsbase_script, ["--input", sigma_output, "--output", ccsbase_output])
    
    integrate_ccs_script = os.path.join(os.path.dirname(__file__), "辅助功能", "CCS预测", "CCS整合.py")
    run_script(integrate_ccs_script, ["--sigma_input", sigma_output, "--ccsbase_input", ccsbase_output, "--output", results_csv], "sigma")
    
    shutil.rmtree(temp_dir, ignore_errors=True)  # 静默清理
    
    # 5. 翻译
    translate_script = os.path.join(os.path.dirname(__file__), "辅助功能", "小牛中文翻译", "小牛中文翻译.py")
    run_script(translate_script, [os.path.dirname(results_csv), "--input", results_csv, "--output", results_csv], "dreams")
    
    logger.info(f"[{level}辅助功能] 完成")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 判断是否为单步骤模式（不显示进度百分比）
    single_step_mode = args.only_l1_vec or args.only_l1 or args.only_l2 or args.only_l3 or args.only_l4
    
    # 确定多层鉴定结果总目录
    if not args.output_dir:
        sample_dir = os.path.dirname(args.sample_msp)
        sample_name = os.path.splitext(os.path.basename(args.sample_msp))[0]
        args.output_dir = os.path.join(sample_dir, f"{sample_name}_多层鉴定结果")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 输出启动信息（终端和日志都显示）
    logger.info("=" * 60)
    logger.info("多层鉴定总控脚本（方法导向架构）")
    logger.info("=" * 60)
    logger.info(f"样品MSP: {args.sample_msp}")
    logger.info(f"离子模式: {args.ion_mode}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"日志文件: {LOG_FILE}")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    total_steps = 6
    current_step = 0

    # Embedding路径
    sample_emb_path = os.path.join(args.output_dir, "L1_results", "embeddings.npz")

    # 模拟库路径（在L2中生成，L3中使用）
    simulated_msp_path = None
    simulated_emb_path_l2 = None

    # ================================================
    # L4: 生成 SIRIUS 输入文件（优先执行，便于用户立即使用SIRIUS GUI）
    # ================================================
    if not args.skip_l4:
        current_step += 1
        print_progress(current_step, total_steps, "L4: 生成 SIRIUS 输入文件", show_pct=not single_step_mode)

        # L4前置：提取同位素信息（调用CSV关联脚本）
        isotope_json_path = os.path.join(args.output_dir, "L4_results", "isotope_distribution.json")
        if args.sample_csv and os.path.exists(args.sample_csv):
            csv_link_script = os.path.join(os.path.dirname(__file__), "辅助功能", "原始数据CSV关联", "原始数据CSV关联.py")
            isotope_args = [
                "--extract_isotope",
                "--sample_csv", args.sample_csv,
                "--output_json", isotope_json_path
            ]
            if not run_script(csv_link_script, isotope_args, "dreams"):
                logger.warning(" 同位素信息提取失败，L4将缺少同位素数据")
                # 创建空JSON避免L4报错
                os.makedirs(os.path.dirname(isotope_json_path), exist_ok=True)
                with open(isotope_json_path, 'w') as f:
                    json_module.dump({}, f)
        else:
            # 无CSV时创建空JSON
            os.makedirs(os.path.dirname(isotope_json_path), exist_ok=True)
            with open(isotope_json_path, 'w') as f:
                json_module.dump({}, f)
            logger.info("无样品CSV，跳过同位素提取")

        # 调用L4脚本生成MSP文件
        l4_script = os.path.join(os.path.dirname(__file__), "L4_从头鉴定", "L4_SIRIUS从头鉴定.py")

        # 构建L4环境变量（添加isotope_json路径）
        l4_env = build_l4_env(args)
        l4_env['L4_ISOTOPE_JSON'] = isotope_json_path

        if not run_script_with_env(l4_script, l4_env, "dreams"):
            logger.warning(" L4 SIRIUS输入文件生成失败")
        else:
            logger.info(" ✓ L4 完成：已生成SIRIUS输入文件，可立即使用SIRIUS GUI分析")
            logger.info(f"  SIRIUS输入目录: {os.path.join(args.output_dir, 'L4_results', 'sirius_input')}")

    # ================================================
    # L1-预处理: DreaMS 向量化
    # ================================================
    if not args.skip_l1_vec:
        current_step += 1
        print_progress(current_step, total_steps, "L1-预处理: 向量化", show_pct=not single_step_mode)
        
        sample_vec_script = os.path.join(os.path.dirname(__file__), "L1_真实数据库匹配", "L1_DreaMS向量化.py")
        if not run_script_with_env(sample_vec_script, build_l1_sample_vec_env(args), "dreams"):
            logger.error("样品向量化失败，终止流程")
            return
        
        db_vec_script = os.path.join(os.path.dirname(__file__), "L1_真实数据库匹配", "L1_DreaMS向量化.py")
        if not run_script_with_env(db_vec_script, build_l1_db_vec_env(args), "dreams"):
            logger.warning("数据库向量化失败，L1 双方法匹配将在运行时自动计算")
    
    # ================================================
    # L1: 真实数据库匹配（方法导向：MC → DreaMS → 整合）
    # ================================================
    if not args.skip_l1:
        current_step += 1
        print_progress(current_step, total_steps, "L1: 真实数据库匹配（MC + DreaMS）", show_pct=not single_step_mode)
        
        l1_output_dir = os.path.join(args.output_dir, "L1_results")
        l1_mc_results = os.path.join(l1_output_dir, "L1_MC_results.csv")
        l1_dreams_results = os.path.join(l1_output_dir, "L1_DreaMS_results.csv")
        
        mc_script = os.path.join(os.path.dirname(__file__), "L1_真实数据库匹配", "L1_MC匹配.py")
        if not run_script_with_env(mc_script, build_l1_mc_env(args), "dreams"):
            logger.error("L1 MC匹配失败")
            return
        
        dreams_script = os.path.join(os.path.dirname(__file__), "L1_真实数据库匹配", "L1_DreaMS匹配.py")
        if not run_script_with_env(dreams_script, build_l1_dreams_env(args), "dreams"):
            logger.warning("L1 DreaMS匹配失败，仅使用MC结果")
        
        l1_results_csv = os.path.join(l1_output_dir, "L1_results.csv")
        summary_script = os.path.join(os.path.dirname(__file__), "辅助功能", "各层鉴定结果汇总", "各层鉴定结果汇总.py")
        summary_args = ["--mode", "L1", "--mc_input", l1_mc_results, "--dreams_input", l1_dreams_results, "--output", l1_results_csv, "--skip_excel"]
        if not run_script(summary_script, summary_args, "dreams"):
            logger.error(" L1结果整合失败")
            return
        
        run_auxiliary_for_level(l1_results_csv, "L1", args)
        
        # 辅助功能完成后生成Excel
        if os.path.exists(summary_script):
            summary_cmd = ["/home/lyx/miniconda3/envs/dreams/bin/python", summary_script, "--mode", "L1", "--input", l1_results_csv, "--output", l1_results_csv]
            subprocess.run(summary_cmd, capture_output=True, text=True)
    
    # ================================================
    # L2: 模拟数据库匹配（全量模拟库 + MC + DreaMS + 跨库分子网络）
    # ================================================
    if not args.skip_l2:
        current_step += 1
        print_progress(current_step, total_steps, "L2: 模拟数据库匹配（MC + DreaMS + 分子网络）", show_pct=not single_step_mode)
        
        l2_output_dir = os.path.join(args.output_dir, "L2_results")
        l2_mc_results = os.path.join(l2_output_dir, "L2_MC_results.csv")
        l2_dreams_results = os.path.join(l2_output_dir, "L2_DreaMS_results.csv")
        
        l2_sample_emb_path = sample_emb_path
        
        fiora_script = os.path.join(os.path.dirname(__file__), "L2_模拟数据库匹配", "L2_FIORA_模拟数据库构建.py")
        fiora_ok = run_script_with_env(fiora_script, build_l2_fiora_env(args), "fiora")
        if not fiora_ok:
            logger.warning(" FIORA 预测失败，仅使用 CFM-ID 预测结果")
        
        cfmid_script = os.path.join(os.path.dirname(__file__), "L2_模拟数据库匹配", "L2_CFMID_模拟数据库构建.py")
        cfmid_ok = run_script_with_env(cfmid_script, build_l2_cfmid_env(args), "dreams")
        if not cfmid_ok:
            logger.warning(" CFM-ID 预测失败，仅使用 FIORA 预测结果")
        
        if not fiora_ok and not cfmid_ok:
            logger.error(" FIORA 和 CFM-ID 均预测失败，无法继续 L2")
            return
        
        fiora_msp = None
        cfmid_msp = None
        fiora_path_file = os.path.join(l2_output_dir, "fiora_msp_path.txt")
        cfmid_path_file = os.path.join(l2_output_dir, "cfmid_msp_path.txt")
        
        if fiora_ok and os.path.exists(fiora_path_file):
            fiora_msp = open(fiora_path_file).read().strip()
        if cfmid_ok and os.path.exists(cfmid_path_file):
            cfmid_msp = open(cfmid_path_file).read().strip()
        
        msp_files_for_vec = [msp for msp in (fiora_msp, cfmid_msp) if msp and os.path.exists(msp)]
        
        vectorize_script = os.path.join(os.path.dirname(__file__), "L1_真实数据库匹配", "L1_DreaMS向量化.py")
        
        if msp_files_for_vec:
            if len(msp_files_for_vec) > 1:
                first_dir = os.path.dirname(msp_files_for_vec[0])
                simulated_msp_path = os.path.join(first_dir, "simulated_merged.msp")
            else:
                simulated_msp_path = msp_files_for_vec[0]
            simulated_emb_path_l2 = simulated_msp_path.rsplit('.', 1)[0] + '_dreams_emb.npz'
            
            if not run_script_with_env(vectorize_script, build_l1_single_file_vec_env(msp_files_for_vec), "dreams"):
                simulated_emb_path_l2 = None
        else:
            logger.error(" L2模拟库生成失败,无可用MSP文件")
            return
        
        mc_script = os.path.join(os.path.dirname(__file__), "L2_模拟数据库匹配", "L2_MC匹配.py")
        
        l2_mc_env = {
            'L1_SAMPLE_MSP': args.sample_msp,
            'L1_ION_MODE': args.ion_mode,
            'L1_MC_LIBRARIES': json_module.dumps({'simulated': simulated_msp_path}),
            'L1_MZ_TOLERANCE_PPM': str(L2_PRECURSOR_PPM),
            'L1_FRAGMENT_TOLERANCE': str(L2_FRAGMENT_TOLERANCE),
            'L1_MIN_MATCHED_PEAKS': str(L2_MIN_MATCHED_FRAGMENTS),
            'L1_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
            'L1_TOP_K': str(L2_TOP_K_RESULTS),
            'L1_OUTPUT_DIR': l2_output_dir,
        }
        
        if not run_script_with_env(mc_script, l2_mc_env, "dreams"):
            logger.error(" L2 MC 匹配失败")
            return
        
        l1_mc_out = os.path.join(l2_output_dir, "L1_MC_results.csv")
        if os.path.exists(l1_mc_out):
            os.rename(l1_mc_out, l2_mc_results)
        
        dreams_script = os.path.join(os.path.dirname(__file__), "L2_模拟数据库匹配", "L2_DreaMS匹配.py")
        
        if simulated_emb_path_l2:
            l2_dreams_env = {
                'L1_SAMPLE_MSP': args.sample_msp,
                'L1_SAMPLE_EMB': l2_sample_emb_path,
                'L1_ION_MODE': args.ion_mode,
                'L1_LIBRARIES': json_module.dumps({'simulated': {'msp': simulated_msp_path, 'emb': simulated_emb_path_l2}}),
                'L1_MZ_TOLERANCE_PPM': str(L2_PRECURSOR_PPM),
                'L1_FRAGMENT_TOLERANCE': str(L2_FRAGMENT_TOLERANCE),
                'L1_MIN_MATCHED_PEAKS': str(L2_MIN_MATCHED_FRAGMENTS),
                'L1_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
                'L1_TOP_K': str(L2_TOP_K_RESULTS),
                'L1_OUTPUT_DIR': l2_output_dir,
            }
            
            if not run_script_with_env(dreams_script, l2_dreams_env, "dreams"):
                logger.warning(" L2 DreaMS 匹配失败，仅使用 MC 结果")
            else:
                l1_dreams_out = os.path.join(l2_output_dir, "L1_DreaMS_results.csv")
                if os.path.exists(l1_dreams_out):
                    os.rename(l1_dreams_out, l2_dreams_results)
        else:
            logger.warning(" 模拟库 embedding 不可用，跳过 L2 DreaMS 匹配")
        
        l2_results_csv = os.path.join(l2_output_dir, "L2_results.csv")
        summary_script = os.path.join(os.path.dirname(__file__), "辅助功能", "各层鉴定结果汇总", "各层鉴定结果汇总.py")
        summary_args = ["--mode", "L2", "--mc_input", l2_mc_results, "--dreams_input", l2_dreams_results, "--output", l2_results_csv, "--skip_excel"]
        if not run_script(summary_script, summary_args, "dreams"):
            logger.error(" L2结果整合失败")
            return
        
        run_auxiliary_for_level(l2_results_csv, "L2", args)
        
        # 辅助功能完成后生成Excel
        if os.path.exists(summary_script):
            summary_cmd = ["/home/lyx/miniconda3/envs/dreams/bin/python", summary_script, "--mode", "L2", "--input", l2_results_csv, "--output", l2_results_csv]
            subprocess.run(summary_cmd, capture_output=True, text=True)
    
    # ================================================
    # L3: DreaMS 类似物筛查（纯 Embedding 语义匹配）
    # ================================================
    if not args.skip_l3:
        current_step += 1
        print_progress(current_step, total_steps, "L3: DreaMS 类似物筛查", show_pct=not single_step_mode)

        l3_script = os.path.join(os.path.dirname(__file__), "L3_DreaMS类似物筛查", "L3_DreaMS类似物筛查.py")

        # 自动查找已有的模拟库 (当 --only_l3 时)
        # 优先使用合并的模拟库（simulated_merged.msp + simulated_merged_dreams_emb.npz）
        if not simulated_msp_path or not simulated_emb_path_l2:
            l2_results_dir = os.path.join(args.output_dir, "L2_results")
            cfmid_path_file = os.path.join(l2_results_dir, "cfmid_msp_path.txt")
            fiora_path_file = os.path.join(l2_results_dir, "fiora_msp_path.txt")

            # 从 CFM-ID 或 FIORA 路径推断合并文件位置
            base_sim_dir = None
            if os.path.exists(cfmid_path_file):
                with open(cfmid_path_file, 'r') as f:
                    base_sim_dir = os.path.dirname(f.read().strip())
            elif os.path.exists(fiora_path_file):
                with open(fiora_path_file, 'r') as f:
                    base_sim_dir = os.path.dirname(f.read().strip())

            if base_sim_dir:
                # 使用合并的模拟库文件
                merged_msp = os.path.join(base_sim_dir, "simulated_merged.msp")
                merged_emb = os.path.join(base_sim_dir, "simulated_merged_dreams_emb.npz")
                if os.path.exists(merged_msp) and os.path.exists(merged_emb):
                    simulated_msp_path = merged_msp
                    simulated_emb_path_l2 = merged_emb
                    logger.info(f"自动找到合并模拟库: {merged_msp}")
                else:
                    logger.warning(f"合并模拟库不存在: MSP={merged_msp}, EMB={merged_emb}")

        l3_env = build_l3_analog_env(
            args, sample_emb_path,
            simulated_msp=simulated_msp_path,
            simulated_emb=simulated_emb_path_l2,
        )
        
        if not run_script_with_env(l3_script, l3_env, "dreams"):
            logger.error(" L3 DreaMS类似物筛查失败")
            return
        
        l3_results_csv = os.path.join(args.output_dir, "L3_results", "L3_results.csv")
        if os.path.exists(l3_results_csv):
            ontology_script = os.path.join(os.path.dirname(__file__), "辅助功能", "ontology获取", "ontology获取.py")
            run_script(ontology_script, ["--input_csv", l3_results_csv, "--output_csv", l3_results_csv, "--field", "class"], "dreams")
            
            translate_script = os.path.join(os.path.dirname(__file__), "辅助功能", "小牛中文翻译", "小牛中文翻译.py")
            run_script(translate_script, [os.path.dirname(l3_results_csv), "--input", l3_results_csv, "--column", "matched_name", "--output", l3_results_csv], "dreams")
            
            l3_results_excel = l3_results_csv.replace('.csv', '.xlsx')
            run_script(l3_script, ["--regenerate_excel", l3_results_csv, l3_results_excel], "dreams")        # ================================================

    
    # ================================================
    # 最终汇总
    # ================================================
    if not args.skip_summary:
        current_step += 1
        print_progress(current_step, total_steps, "最终结果汇总", show_pct=not single_step_mode)
        
        summary_script = os.path.join(os.path.dirname(__file__), "辅助功能", "各层鉴定结果汇总", "各层鉴定结果汇总.py")
        
        l1_results = os.path.join(args.output_dir, "L1_results", "L1_results.csv")
        l2_results = os.path.join(args.output_dir, "L2_results", "L2_results.csv")
        l3_results = os.path.join(args.output_dir, "L3_results", "L3_results.csv")
        final_output = os.path.join(args.output_dir, "多层鉴定总结果.csv")
        
        summary_cmd = [
            sys.executable, summary_script,
            "--mode", "final",
            "--input", args.output_dir,
            "--output", final_output,
            "--l1_results", l1_results,
            "--l2_results", l2_results,
            "--l3_results", l3_results,
        ]
        
        try:
            result = subprocess.run(summary_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"最终汇总失败: {result.stderr}")
        except Exception as e:
            logger.warning(f"最终汇总执行异常: {e}")
    
    # ================================================
    # 完成
    # ================================================
    elapsed_time = datetime.now() - start_time
    
    if args.only_l1_vec:
        completion_msg = "L1 向量化完成"
    elif args.only_l1:
        completion_msg = "L1 数据库匹配完成"
    elif args.only_l2:
        completion_msg = "L2 模拟库匹配完成"
    elif args.only_l3:
        completion_msg = "L3 DreaMS类似物筛查完成"
    elif args.only_l4:
        completion_msg = "L4 SIRIUS从头鉴定完成"
    else:
        completion_msg = "多层鉴定流程完成"
    
    logger.info("\n" + "=" * 60)
    logger.info(f"{completion_msg}! 总耗时: {elapsed_time}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
