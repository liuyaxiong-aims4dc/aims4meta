#!/usr/bin/env python3
"""
多层鉴定总控脚本 —— 参数配置 + 流程调度，一站式管理。
上方修改参数，下方调度运行。
"""
import os, sys, subprocess, logging, signal, argparse, shutil, json, glob as _g
from datetime import datetime
from pathlib import Path

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       一、运行控制开关（Y/N）                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

RUN_L1 = "Y"
RUN_L2 = "Y"
RUN_L3 = "Y"
RUN_L4 = "Y"

L1_USE_MSDIAL = "Y"
L1_USE_SPECTRAVERSE = "Y"

L2_USE_TCM = "N"
L2_USE_DRUG = "N"
L2_USE_LIPID = "N"
L2_USE_PFAS = "N"

L3_ONLINE_DATABASES = ["BIO"]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       二、基础配置                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SAMPLE = "/stor1/微力临时文件共享/20260508kangfengshi/kangfengshi-NEG.msp"
ION_MODE = "NEG"
OUTPUT_BASE_DIR = ""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    三、L1 实验数据库鉴定参数                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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

L1_PRECURSOR_PPM = 10.0
L1_FRAGMENT_TOLERANCE = 0.05
L1_MIN_MATCHED_FRAGMENTS = 2
L1_COSINE_THRESHOLD = 0.5
L1_TOP_K = 5
L1_BATCH_SIZE = 256

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    四、L2 模拟数据库鉴定参数                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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

L2_PRECURSOR_PPM = L1_PRECURSOR_PPM
L2_FRAGMENT_TOLERANCE = L1_FRAGMENT_TOLERANCE
L2_MIN_MATCHED_FRAGMENTS = L1_MIN_MATCHED_FRAGMENTS
L2_COSINE_THRESHOLD = L1_COSINE_THRESHOLD
L2_TOP_K = L1_TOP_K

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    五、L3 SIRIUS 结构鉴定参数                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

L3_SIRIUS_INSTRUMENT = "qtof"
L3_MIN_PEAKS = 6
L3_MAX_MZ = 1500
L3_MIN_INTENSITY = 500

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    六、L4 DreaMS 分子网络参数                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

L4_SIM_THRESHOLD = 0.8

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    七、结果汇总参数                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SUMMARY_OUTPUT_FORMAT = "xlsx"
SUMMARY_INCLUDE_L1 = True
SUMMARY_INCLUDE_L2 = True
SUMMARY_INCLUDE_L3 = True
SUMMARY_INCLUDE_L4 = False

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                 八、数据库开关与辅助函数                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

L1_DATABASES = L1_EXPERIMENTAL_DATABASES

_L2_SWITCH_PREFIX = {
    "L2_USE_TCM": "TCM", "L2_USE_DRUG": "DRUG",
    "L2_USE_LIPID": "LIPID", "L2_USE_PFAS": "PFAS",
}

def _on(v):
    """开关判断：Y/y/yes/true/1/True 视为启用，其余禁用"""
    if isinstance(v, bool): return v
    if v is None: return False
    return str(v).strip().upper() in ("Y", "YES", "TRUE", "1")

def apply_database_switches():
    """根据上方 L1_USE_* / L2_USE_* 开关过滤 L1/L2 数据库列表"""
    global L1_DATABASES, L2_SIMULATED_DATABASES
    filtered_l1 = {}
    if _on(L1_USE_MSDIAL) and "MSDIAL" in L1_EXPERIMENTAL_DATABASES:
        filtered_l1["MSDIAL"] = L1_EXPERIMENTAL_DATABASES["MSDIAL"]
    if _on(L1_USE_SPECTRAVERSE) and "SpectraVerse" in L1_EXPERIMENTAL_DATABASES:
        filtered_l1["SpectraVerse"] = L1_EXPERIMENTAL_DATABASES["SpectraVerse"]
    L1_DATABASES = filtered_l1
    enabled = {p for sn, p in _L2_SWITCH_PREFIX.items() if _on(globals().get(sn))}
    filtered_l2 = {n:c for n,c in L2_SIMULATED_DATABASES.items() if n.split("_",1)[0] in enabled}
    L2_SIMULATED_DATABASES = filtered_l2

def clear_library_caches(msp_path):
    """清除过期缓存：matchms_cache.pkl / spectra_cache.pkl / dreams_emb.npz"""
    base = msp_path.replace('.msp', '')
    for pat in [f"{base}_matchms_cache.pkl", f"{base}_spectra_cache.pkl", f"{base}_dreams_emb.npz"]:
        for cf in _g.glob(pat):
            if os.path.exists(cf) and os.path.getmtime(cf) < os.path.getmtime(msp_path):
                os.remove(cf)

def get_l1_databases(ion_mode):
    """根据离子模式获取已启用的 L1 数据库配置（msp路径 + embedding缓存路径）"""
    ion_mode = ion_mode.upper()
    dbs = {}
    for nm, paths in L1_DATABASES.items():
        msp = paths.get(ion_mode)
        if msp and os.path.exists(msp):
            clear_library_caches(msp)
            dbs[nm] = {"msp": msp, "emb": msp.replace('.msp', '_dreams_emb.npz')}
    return dbs

def get_l2_databases(ion_mode):
    """根据离子模式获取已启用的 L2 数据库配置（msp路径 + embedding缓存路径）"""
    ion_mode = ion_mode.upper()
    dbs = {}
    for nm, paths in L2_SIMULATED_DATABASES.items():
        msp = paths.get(ion_mode)
        if msp:
            if os.path.exists(msp):
                clear_library_caches(msp)
                dbs[nm] = {"msp": msp, "emb": msp.replace('.msp', '_dreams_emb.npz')}
    return dbs

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              九、子脚本环境变量构建函数                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_l1_sample_vec_env(args):
    """L1 样品 DreaMS 向量化 —— 环境变量"""
    return {'L1_MSP_FILE': args.sample,
            'L1_OUTPUT_DIR': os.path.join(args.output_dir, "L1_results")}

def build_l1_db_vec_env(args):
    """L1 数据库 DreaMS 向量化 —— 环境变量"""
    dbs = get_l1_databases(args.ion_mode)
    return {'L1_DATABASES': json.dumps({args.ion_mode: {n: i["msp"] for n,i in dbs.items()}}),
            'L1_ION_MODE': args.ion_mode}

def build_l1_env(args):
    """L1 公共环境变量（matchMS 和 DreaMS 共用基础参数）"""
    sf = args.sample; sc = None
    if sf.lower().endswith('.mzml'):
        ac = sf.replace('.mzML','.csv').replace('.mzml','.csv')
        if os.path.exists(ac): sc = ac
    elif sf.lower().endswith('.msp'):
        ac = sf.replace('.msp','.csv').replace('.MSP','.csv')
        if os.path.exists(ac): sc = ac
    return {'L1_SAMPLE_MSP': sf, 'L1_SAMPLE_CSV': sc or '',
            'L1_ION_MODE': args.ion_mode,
            'L1_MZ_TOLERANCE_PPM': str(int(L1_PRECURSOR_PPM)),
            'L1_FRAGMENT_TOLERANCE': str(L1_FRAGMENT_TOLERANCE),
            'L1_MIN_MATCHED_PEAKS': str(L1_MIN_MATCHED_FRAGMENTS),
            'L1_COSINE_THRESHOLD': str(L1_COSINE_THRESHOLD),
            'L1_TOP_K': str(L1_TOP_K),
            'L1_OUTPUT_DIR': os.path.join(args.output_dir, "L1_results")}

def build_l1_mc_env(args, qi=None):
    """L1 matchMS 鉴定 —— 环境变量（含数据库列表 + 可选 QI CSV）"""
    dbs = get_l1_databases(args.ion_mode)
    env = build_l1_env(args)
    env['L1_MATCHMS_LIBRARIES'] = json.dumps({n: i["msp"] for n,i in dbs.items()})
    if qi: env['L1_QI_CSV'] = qi
    return env

def build_l1_dreams_env(args):
    """L1 DreaMS 鉴定 —— 环境变量（含样品 embedding + 库 embedding）"""
    env = build_l1_env(args)
    env['L1_SAMPLE_EMB'] = os.path.join(args.output_dir, "L1_results", "embeddings.npz")
    env['L1_LIBRARIES'] = json.dumps(get_l1_databases(args.ion_mode))
    return env

def build_l1_single_file_vec_env(msp_file):
    """单/多文件 DreaMS 向量化 —— 环境变量（L2 模拟库用，强制不重算）"""
    return {'L1_MSP_FILE': json.dumps(msp_file) if isinstance(msp_file, list) else msp_file,
            'L1_BATCH_SIZE': str(L1_BATCH_SIZE), 'L1_FORCE_COMPUTE': '0'}

def build_l2_mc_env_full(args, l2_in, dbs, out):
    """L2 matchMS 鉴定 —— 完整环境变量（总控内联调用）"""
    return {'L1_SAMPLE_MSP': l2_in, 'L1_ION_MODE': args.ion_mode,
            'L1_MATCHMS_LIBRARIES': json.dumps({n: i["msp"] for n,i in dbs.items()}),
            'L1_MZ_TOLERANCE_PPM': str(L2_PRECURSOR_PPM),
            'L1_FRAGMENT_TOLERANCE': str(L2_FRAGMENT_TOLERANCE),
            'L1_MIN_MATCHED_PEAKS': str(L2_MIN_MATCHED_FRAGMENTS),
            'L1_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
            'L1_TOP_K': str(L2_TOP_K), 'L1_OUTPUT_DIR': out}

def build_l2_dreams_env_full(args, l2_in, emb, dbs, out):
    """L2 DreaMS 鉴定 —— 完整环境变量（总控内联调用）"""
    return {'L1_SAMPLE_MSP': l2_in, 'L1_SAMPLE_EMB': emb,
            'L1_ION_MODE': args.ion_mode,
            'L1_LIBRARIES': json.dumps(dbs),
            'L1_MZ_TOLERANCE_PPM': str(L2_PRECURSOR_PPM),
            'L1_FRAGMENT_TOLERANCE': str(L2_FRAGMENT_TOLERANCE),
            'L1_MIN_MATCHED_PEAKS': str(L2_MIN_MATCHED_FRAGMENTS),
            'L1_COSINE_THRESHOLD': str(L2_COSINE_THRESHOLD),
            'L1_TOP_K': str(L2_TOP_K), 'L1_OUTPUT_DIR': out}

def build_l4_analog_env(args, emb):
    """L4 DreaMS 分子网络 —— 环境变量（样品间 embedding 互连）"""
    return {'L4_OUTPUT_DIR': os.path.join(args.output_dir, "L4_results"),
            'L4_SAMPLE_EMB': emb, 'L4_SAMPLE_MSP': args.sample,
            'L4_SIM_THRESHOLD': str(L4_SIM_THRESHOLD)}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                 十、调度引擎（纯调度，不处理数据）                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

current_process = None
def _on_signal(signum, frame):
    global current_process
    L = logging.getLogger(__name__); L.warning(f"收到信号{signum}, 停止子进程...")
    if current_process and current_process.poll() is None:
        current_process.terminate()
        try: current_process.wait(timeout=5)
        except subprocess.TimeoutExpired: current_process.kill()
    sys.exit(1)
signal.signal(signal.SIGINT, _on_signal); signal.signal(signal.SIGTERM, _on_signal)

def _setup_logging():
    d = Path(__file__).parent.parent.parent / "logs"; d.mkdir(exist_ok=True)
    f = d / f"多层鉴定_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    r = logging.getLogger(); r.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f, encoding='utf-8'); fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    r.addHandler(fh); r.addHandler(ch); return f

LOG_FILE = _setup_logging()
logger = logging.getLogger(__name__)

def parse_arguments():
    p = argparse.ArgumentParser(description="多层鉴定总控脚本")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--qi_csv", default=None)
    p.add_argument("--ion_mode", default=ION_MODE, choices=["POS","NEG"])
    p.add_argument("--output_dir", default=OUTPUT_BASE_DIR)
    p.add_argument("--databases", nargs="*", default=[])
    for x in ["skip_l1_vec","skip_l1","skip_l2","skip_l3","skip_l4"]: p.add_argument(f"--{x}", action="store_true")
    for x in ["only_l1_vec","only_l1","only_l2","only_l3","only_l4"]: p.add_argument(f"--{x}", action="store_true")
    a = p.parse_args(); a.ion_mode = a.ion_mode.upper()
    if a.only_l1_vec: a.skip_l1 = a.skip_l2 = a.skip_l3 = a.skip_l4 = True
    elif a.only_l1: a.skip_l2 = a.skip_l3 = a.skip_l4 = True
    elif a.only_l2: a.skip_l1_vec = a.skip_l1 = a.skip_l3 = a.skip_l4 = True
    elif a.only_l3: a.skip_l1_vec = a.skip_l1 = a.skip_l2 = a.skip_l4 = True
    elif a.only_l4: a.skip_l1_vec = a.skip_l1 = a.skip_l2 = a.skip_l3 = True
    a.skip_summary = a.only_l1_vec; return a

# ---- 工具函数 ----
# SD/SP: 脚本目录/脚本路径 简写
SD = lambda: os.path.dirname(__file__)
SP = lambda p: os.path.join(SD(), p)
# PY: Python 解释器路径（conda 环境隔离）
PY = lambda e=None: f"/home/lyx/miniconda3/envs/{e}/bin/python" if e else sys.executable
# FC: 从 MSP 路径推导配套 CSV 文件（Progenesis QI 导出）
FC = lambda m: (c:=m.replace('.msp','.csv').replace('.MSP','.csv'), c if os.path.exists(c) else None)[1]

def _run(script, args_list, env=None):
    """调用子脚本（命令行参数方式）—— 阻塞等待完成，支持 Ctrl+C 中断"""
    cmd = [PY(env), "-u", script] + args_list
    logger.debug(f"[CMD] {' '.join(cmd)}")
    try:
        global current_process
        p = subprocess.Popen(cmd, cwd=os.path.dirname(script))
        current_process = p; p.wait()
        if p.returncode != 0: logger.error(f"脚本失败({p.returncode})"); return False
        return True
    except Exception as e: logger.error(f"异常: {e}"); return False

def _rune(script, env_vars, env=None):
    """调用子脚本（环境变量方式）—— 注入 env_vars 后阻塞等待完成"""
    cmd = [PY(env), "-u", script]
    fe = os.environ.copy(); fe.update(env_vars)
    logger.debug(f"[CMD] {' '.join(cmd)}")
    try:
        global current_process
        p = subprocess.Popen(cmd, cwd=os.path.dirname(script), env=fe,
                            stdout=None, stderr=subprocess.STDOUT)
        current_process = p; p.wait()
        if p.returncode != 0: logger.error(f"脚本失败({p.returncode})"); return False
        return True
    except Exception as e: logger.error(f"异常: {e}"); return False

def _pg(step, total, name, pct=True):
    """打印进度： [当前步/总步数] 步骤名"""
    logger.info(f"[{step}/{total}] {name}" + (f" ({step/total*100:.1f}%)" if pct else ""))

def _gen_umsp(sc, rcsv, msp, out):
    """调用汇总脚本生成未鉴定 MSP（从已鉴定 CSV 反筛剩余化合物）"""
    return _run(sc, ["--mode","generate_unidentified","--input",rcsv,
        "--sample_msp",msp,"--output",out,"--output_msp",out], "dreams")

def _clean(args):
    """清理旧结果：删除各层 CSV/缓存，保留 embedding.npz 和 sirius_project 续传文件"""
    for lv in {l for l,f in [('L1',args.skip_l1),('L2',args.skip_l2),
               ('L3',args.skip_l3),('L4',args.skip_l4)] if not f}:
        d = os.path.join(args.output_dir, f"{lv}_results")
        if not os.path.isdir(d): continue
        for it in os.listdir(d):
            ip = os.path.join(d, it)
            if it in ('embeddings.npz',) or it.startswith('sirius_project') or it == 'sirius_params.json': continue
            try: shutil.rmtree(ip) if os.path.isdir(ip) else os.remove(ip)
            except Exception as e: logger.warning(f"清理失败: {ip}")
        logger.info(f"已清理 {lv} 旧结果")
    for p in ['多层鉴定总结果.*', '*_多层鉴定总结果.*']:
        for f in _g.glob(os.path.join(args.output_dir, p)):
            try: os.remove(f)
            except Exception: pass

def _sort(rcsv, lv):
    """调用排序脚本按综合得分重排鉴定结果"""
    if not os.path.exists(rcsv): return
    _run(SP("辅助功能/鉴定结果排序/鉴定结果排序.py"),
         ["--input",rcsv,"--output",rcsv,"--level",lv,
          "--top_k",str(L1_TOP_K if lv=='L1' else L2_TOP_K)], "dreams")

# ---- 各步骤调度 ----
def _l1_vec(args):
    """步骤1: L1 预处理 —— 样品 + 数据库 DreaMS 向量化"""
    sv = SP("L1_实验数据库鉴定/L1_DreaMS向量化.py")
    if not _rune(sv, build_l1_sample_vec_env(args), "dreams"): return False
    dbs = get_l1_databases(args.ion_mode)
    if dbs: _rune(sv, build_l1_db_vec_env(args), "dreams")
    else: logger.info("未启用L1数据库")
    return True

def _l1_id(args, qi):
    """步骤2: L1 实验数据库鉴定 —— matchMS → DreaMS → 整合 → 排序 → 生成未鉴定MSP"""
    out = os.path.join(args.output_dir, "L1_results")
    mc, dr = os.path.join(out,"L1_matchMS_results.csv"), os.path.join(out,"L1_DreaMS_results.csv")
    if not _rune(SP("L1_实验数据库鉴定/L1_matchMS鉴定.py"), build_l1_mc_env(args, qi), "matchms"):
        logger.error("L1 matchMS失败"); return {}
    _rune(SP("L1_实验数据库鉴定/L1_DreaMS鉴定.py"), build_l1_dreams_env(args), "dreams")
    r = os.path.join(out, "L1_results.csv")
    if not _run(SP("辅助功能/各层鉴定结果汇总/各层鉴定结果汇总.py"),
        ["--mode","L1","--mc_input",mc,"--dreams_input",dr,"--output",r,"--sample_msp",args.sample], "dreams"):
        return {}
    _sort(r, "L1"); u = os.path.join(out, "L1_unidentified.msp")
    return {"u": u, "ok": _gen_umsp(SP("辅助功能/各层鉴定结果汇总/各层鉴定结果汇总.py"), r, args.sample, u)}

def _l2_id(args, l1u, emb):
    """步骤3: L2 模拟数据库鉴定 —— 向量化库 → matchMS → DreaMS → 整合 → 排序 → 生成未鉴定MSP"""
    out = os.path.join(args.output_dir, "L2_results")
    if not L2_SIMULATED_DATABASES: return {"u": l1u, "in": l1u}
    dbs = get_l2_databases(args.ion_mode)
    if not dbs: logger.error(f"[L2] 无{args.ion_mode}数据库"); return {}
    logger.info(f"[L2] 数据库: {', '.join(dbs.keys())}")
    vs = SP("L1_实验数据库鉴定/L1_DreaMS向量化.py")
    for nm, info in dbs.items():
        if not os.path.exists(info["emb"]): _rune(vs, build_l1_single_file_vec_env(info["msp"]), "dreams")
    li = l1u
    _rune(SP("L2_模拟数据库鉴定/L2_matchMS鉴定.py"), build_l2_mc_env_full(args, li, dbs, out), "matchms")
    o = os.path.join(out, "L1_matchMS_results.csv")
    if os.path.exists(o): os.rename(o, os.path.join(out, "L2_matchMS_results.csv"))
    mc2, dr2 = os.path.join(out,"L2_matchMS_results.csv"), os.path.join(out,"L2_DreaMS_results.csv")
    if all(os.path.exists(info["emb"]) for info in dbs.values()):
        _rune(SP("L2_模拟数据库鉴定/L2_DreaMS鉴定.py"),
              build_l2_dreams_env_full(args, li, emb, dbs, out), "dreams")
        o = os.path.join(out, "L1_DreaMS_results.csv")
        if os.path.exists(o): os.rename(o, dr2)
    r = os.path.join(out, "L2_results.csv")
    _run(SP("辅助功能/各层鉴定结果汇总/各层鉴定结果汇总.py"),
         ["--mode","L2","--mc_input",mc2,"--dreams_input",dr2,"--output",r], "dreams")
    _sort(r, "L2"); u = os.path.join(out, "L2_unidentified.msp")
    if not _gen_umsp(SP("辅助功能/各层鉴定结果汇总/各层鉴定结果汇总.py"), r, li, u):
        logger.warning("L2 未鉴定失败"); u = li
    return {"u": u, "in": li}

def _l3_sirius(args, l2u, l1ok, l1u, qi):
    """步骤4: L3 SIRIUS 结构鉴定 —— 参数校验 → SIRIUS CLI → 保存续传状态 → 生成未鉴定MSP"""
    out = os.path.join(args.output_dir, "L3_results"); os.makedirs(out, exist_ok=True)
    l3i = args.sample
    if l2u and os.path.exists(l2u) and os.path.getsize(l2u)>0: l3i = l2u
    elif l1ok and os.path.exists(l1u) and os.path.getsize(l1u)>0: l3i = l1u; logger.info("[L3] 使用 L1 未鉴定")
    else: logger.warning("[L3] 无可用的未鉴定 MSP，使用原始样品")
    dbs = list(L3_ONLINE_DATABASES)
    ck = SP("辅助功能/L3参数续传校验/L3参数续传校验.py")
    ca = ["--mode","check","--output_dir",out,"--sample_msp",l3i,
          "--ion_mode",args.ion_mode,"--instrument",L3_SIRIUS_INSTRUMENT,
          "--mz_threshold",str(L3_MAX_MZ),"--min_peaks",str(L3_MIN_PEAKS),"--databases"]+dbs
    if qi and L3_MIN_INTENSITY>0: ca += ["--sample_csv",qi,"--min_intensity",str(L3_MIN_INTENSITY)]
    _run(ck, ca, "dreams")
    cmd = [sys.executable, SP("L3_SIRIUS结构鉴定/L3_SIRIUS_CLI.py"),
           "--sample_msp",l3i,"--output_dir",out,"--instrument",L3_SIRIUS_INSTRUMENT,
           "--ion_mode",args.ion_mode,"--min_peaks",str(L3_MIN_PEAKS),"--mz_threshold",str(L3_MAX_MZ)]
    if qi and L3_MIN_INTENSITY>0: cmd += ["--sample_csv",qi,"--min_intensity",str(L3_MIN_INTENSITY)]
    cmd += ["--databases"]+dbs
    logger.info("[L3] 启动 SIRIUS")
    try:
        global current_process
        p = subprocess.Popen(cmd, text=True)
        current_process = p; r = p.wait()
        if r != 0: logger.error(f"[L3] 失败({r})")
        else:
            logger.info("[L3] 成功")
            sa = ["--mode","save","--output_dir",out,"--sample_msp",l3i,
                  "--ion_mode",args.ion_mode,"--instrument",L3_SIRIUS_INSTRUMENT,
                  "--mz_threshold",str(L3_MAX_MZ),"--min_peaks",str(L3_MIN_PEAKS),"--databases"]+dbs
            if qi and L3_MIN_INTENSITY>0: sa += ["--sample_csv",qi,"--min_intensity",str(L3_MIN_INTENSITY)]
            _run(ck, sa, "dreams")
    except Exception as e: logger.error(f"[L3] 异常: {e}")
    idf = os.path.join(out, "L3_identified.csv")
    if os.path.exists(idf):
        _run(SP("辅助功能/各层鉴定结果汇总/各层鉴定结果汇总.py"),
            ["--mode","generate_unidentified","--input",idf,"--sample_msp",l3i,
             "--output",os.path.join(out,"L3_unidentified.msp"),
             "--output_msp",os.path.join(out,"L3_unidentified.msp")], "dreams")

def _l4_net(args, emb):
    """步骤5: L4 DreaMS 分子网络 —— 样品间 embedding 余弦相似度建网"""
    _rune(SP("L4_DreaMS分子网络/L4_DreaMS分子网络.py"), build_l4_analog_env(args, emb), "dreams")

def _final(args):
    """步骤6: 最终汇总 —— 合并L1+L2+L3 → CSV关联 → 同位素 → Ontology → CCS → 翻译 → Excel"""
    fc = os.path.join(args.output_dir, "多层鉴定总结果.csv")
    sm = SP("辅助功能/各层鉴定结果汇总/各层鉴定结果汇总.py")
    try:
        r = subprocess.run([sys.executable,sm,"--mode","final","--input",args.output_dir,
            "--output",fc,"--l1_results",os.path.join(args.output_dir,"L1_results","L1_results.csv"),
            "--l2_results",os.path.join(args.output_dir,"L2_results","L2_results.csv"),
            "--l3_results",os.path.join(args.output_dir,"L3_results","L3_identified.csv"),
            "--l4_results",os.path.join(args.output_dir,"L4_results","L4_network_nodes.csv"),
            "--sample_msp",args.sample,"--ion_mode",args.ion_mode], capture_output=True, text=True)
        if r.returncode != 0: logger.warning(f"汇总失败({r.returncode})")
    except Exception as e: logger.warning(f"汇总异常: {e}")
    if not os.path.exists(fc): return
    logger.info("[汇总] 辅助功能: CSV关联→同位素→Ontology→CCS→翻译")
    qi = getattr(args,'qi_csv',None) or FC(args.sample)
    if qi and os.path.exists(qi):
        _run(SP("辅助功能/原始数据CSV关联/原始数据CSV关联.py"), ["--input",fc,"--sample_csv",qi,"--output",fc], "dreams")
    _run(SP("辅助功能/同位素相似度计算/同位素相似度计算.py"), ["--input",fc,"--output",fc], "dreams")
    oa = ["--input_csv",fc,"--output_csv",fc,"--field","class","--canopus_fallback",
          "--sample_msp",args.sample,"--ion_mode",args.ion_mode,"--instrument",L3_SIRIUS_INSTRUMENT]
    sd = os.path.join(args.output_dir,"L3_results","sirius_results")
    if os.path.isdir(sd): oa += ["--sirius_results_dir",sd]
    _run(SP("辅助功能/ontology获取/ontology获取.py"), oa, "dreams")
    _run(SP("辅助功能/CCS预测/ccsbase2预测.py"), [fc,fc], "dreams")
    _run(SP("辅助功能/小牛中文翻译/小牛中文翻译.py"), [os.path.dirname(fc),"--input",fc,"--output",fc], "dreams")
    logger.info("[汇总] 生成 Excel...")
    _run(sm, ["--mode","final_excel","--input",fc,"--output",fc,
        "--sample_msp",args.sample,"--ion_mode",args.ion_mode], "dreams")

# ---- 主函数（纯编排，顺序调度 7 步） ----
def main():
    """
    多层鉴定主流程：
      参数解析 → 开关应用 → 清理旧结果 →
      L1向量化 → L1鉴定 → L2鉴定 → L3 SIRIUS → L4分子网络 → 最终汇总
    """
    args = parse_arguments(); apply_database_switches()
    if not args.skip_l1: args.skip_l1 = not _on(RUN_L1)
    if not args.skip_l2: args.skip_l2 = not _on(RUN_L2)
    if not args.skip_l3: args.skip_l3 = not _on(RUN_L3)
    if not args.skip_l4: args.skip_l4 = not _on(RUN_L4)
    if not args.sample: logger.error("[ERROR] 请提供 --sample"); return
    ss = args.only_l1_vec or args.only_l1 or args.only_l2 or args.only_l3 or args.only_l4
    if not args.output_dir:
        d = os.path.dirname(args.sample); n = os.path.splitext(os.path.basename(args.sample))[0]
        args.output_dir = os.path.join(d, f"{n}_多层鉴定结果")
    os.makedirs(args.output_dir, exist_ok=True); _clean(args)
    logger.info("="*60)
    logger.info(f"多层鉴定总控 | {args.sample} | {args.ion_mode}")
    l1n = list(L1_DATABASES.keys()) if L1_DATABASES else ["无"]
    l2n = list(L2_SIMULATED_DATABASES.keys()) if L2_SIMULATED_DATABASES else ["无"]
    logger.info(f"L1: {', '.join(l1n)} | L2: {', '.join(l2n)}")
    ls = [lv for lv,f in [("L1",args.skip_l1),("L2",args.skip_l2),
          ("L3",args.skip_l3),("L4",args.skip_l4)] if not f]
    logger.info(f"运行: {'→'.join(ls) if ls else '无'} → 汇总")
    logger.info(f"输出: {args.output_dir}"); logger.info("="*60)
    qi = FC(args.sample) if args.sample.lower().endswith('.msp') else None
    if qi: logger.info(f"配套CSV: {qi}")
    emb = os.path.join(args.output_dir, "L1_results", "embeddings.npz")
    start, step, total = datetime.now(), 0, 7
    if not args.skip_l1_vec:
        step += 1; _pg(step, total, "L1 向量化", not ss)
        if not _l1_vec(args): return
    l1 = {}
    if not args.skip_l1:
        step += 1; _pg(step, total, "L1 实验库鉴定", not ss)
        l1 = _l1_id(args, qi)
    l2 = {}
    if not args.skip_l2:
        step += 1; _pg(step, total, "L2 模拟库鉴定", not ss)
        l2 = _l2_id(args, l1.get("u", args.sample), emb)
    if not args.skip_l3:
        step += 1; _pg(step, total, "L3 SIRIUS", not ss)
        _l3_sirius(args, l2.get("u", args.sample), l1.get("ok", False), l1.get("u", args.sample), qi)
    if not args.skip_l4:
        step += 1; _pg(step, total, "L4 分子网络", not ss)
        _l4_net(args, emb)
    if not args.skip_summary:
        step += 1; _pg(step, total, "最终汇总", not ss)
        _final(args)
    logger.info("\n"+"="*60)
    logger.info(f"完成! 耗时: {datetime.now()-start}")
    logger.info(f"输出: {args.output_dir}"); logger.info("="*60)

if __name__ == "__main__":
    main()
