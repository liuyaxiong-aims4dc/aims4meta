#!/usr/bin/env python3
"""
L1-预处理: DreaMS 向量化工具

支持三种模式：
1. 数据库模式（L1数据库）：通过 L1_DATABASES 环境变量传入数据库配置
2. 单文件模式（L2模拟库）：通过 L1_MSP_FILE 环境变量传入单个 MSP 文件路径
   - 支持传入多个文件（JSON列表），自动合并后向量化
3. 样品模式（L1样品）：通过 L1_MSP_FILE + L1_OUTPUT_DIR 指定输入和输出路径

输出：
- 数据库/单文件模式：与输入MSP同目录的 *_dreams_emb.npz 文件
- 样品模式：{OUTPUT_DIR}/embeddings.npz

运行环境：dreams
"""

###############################################################################
# 配置参数区（全部由总控脚本注入，无默认值）
###############################################################################
import os
import json

# 模式判断：
# - 单文件模式：L1_MSP_FILE 指定单个 MSP 文件路径，或 JSON 列表（多个文件自动合并）
# - 数据库模式：L1_DATABASES 指定数据库配置 JSON

MSP_FILE = os.environ.get('L1_MSP_FILE')  # 单文件模式（支持JSON列表）
DATABASES_JSON = os.environ.get('L1_DATABASES')  # 数据库模式（L1）
ION_MODE = os.environ.get('L1_ION_MODE')  # 离子模式（数据库模式必需）
OUTPUT_DIR = os.environ.get('L1_OUTPUT_DIR')  # 自定义输出目录（样品模式）
BATCH_SIZE = os.environ.get('L1_BATCH_SIZE')  # 批处理大小
FORCE_COMPUTE = os.environ.get('L1_FORCE_COMPUTE', '0')  # 强制重新计算

# 验证参数
if not MSP_FILE and not DATABASES_JSON:
    raise ValueError("错误：必须设置 L1_MSP_FILE 或 L1_DATABASES 环境变量！")

L1_DATABASE_MSP = json.loads(DATABASES_JSON) if DATABASES_JSON else None

# 类型转换
BATCH_SIZE = int(BATCH_SIZE) if BATCH_SIZE else 64
FORCE_COMPUTE = FORCE_COMPUTE.lower() in ('1', 'true', 'yes')

###############################################################################

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# DreaMS PYTHONPATH
DREAMS_SRC = "/stor3/AIMS4Meta/源代码/DreaMS"
if DREAMS_SRC not in sys.path:
    sys.path.insert(0, DREAMS_SRC)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def get_databases_for_mode(ion_mode):
    """获取指定离子模式的数据库 MSP 路径"""
    return L1_DATABASE_MSP.get(ion_mode, {})


def count_spectra_in_msp(msp_path):
    """快速计算MSP文件中的谱图数量（通过统计Name字段）"""
    count = 0
    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().lower().startswith('name:'):
                count += 1
    return count


def extract_ontology_from_msp(msp_path):
    """从MSP文件中提取名称到Ontology的映射（matchms不解析Ontology字段）"""
    ontology_map = {}
    current_name = None
    current_ontology = ""
    
    with open(msp_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_lower = line.strip().lower()
            
            # 名称字段
            if line_lower.startswith('name:'):
                # 保存上一个谱图的 ontology
                if current_name:
                    ontology_map[current_name] = current_ontology
                current_name = line.strip()[5:].strip()  # 去掉 "NAME:" 前缀
                current_ontology = ""
            
            # Ontology 字段（matchms 不解析）
            elif line_lower.startswith('ontology:'):
                current_ontology = line.strip()[9:].strip()  # 去掉 "Ontology:" 前缀
        
        # 保存最后一个谱图
        if current_name:
            ontology_map[current_name] = current_ontology
    
    return ontology_map


def msp_to_mgf(msp_path, mgf_path):
    """将MSP转换为MGF格式（DreaMS输入），同时提取元数据和保存谱图缓存"""
    from matchms.importing import load_from_msp
    import pickle
    
    # 预解析：提取 Ontology 字段（matchms 不解析）
    print(f"  预解析Ontology字段...")
    ontology_map = extract_ontology_from_msp(msp_path)
    print(f"  提取到 {len(ontology_map)} 条Ontology记录")
    
    # 快速预计算谱图数量
    total_spectra = count_spectra_in_msp(msp_path)
    print(f"  解析MSP: {msp_path} (预估 {total_spectra} 条)")
    
    # 解析所有谱图
    spectra = []
    for sp in tqdm(load_from_msp(msp_path), desc="  解析MSP", total=total_spectra, ncols=80, unit="条"):
        spectra.append(sp)
    
    print(f"  解析完成: {len(spectra)} 条谱图")
    
    # 收集元数据
    metadata = {
        'names': [],
        'smiles': [],
        'inchikeys': [],
        'precursor_mzs': [],
        'adducts': [],
        'ccs_list': [],
        'ontologies': [],
        'formulas': []  # 添加分子式字段
    }
    
    # 写入MGF（带进度条）
    with open(mgf_path, 'w', encoding='utf-8') as f:
        for i, sp in enumerate(tqdm(spectra, desc="  写入MGF", ncols=80)):
            f.write('BEGIN IONS\n')
            title = sp.get('name') or sp.get('compound_name') or f"spectrum_{i}"
            f.write(f'TITLE={title}\n')
            pmz = sp.get('precursor_mz')
            if pmz is not None:
                f.write(f'PEPMASS={pmz}\n')
            if sp.peaks is not None:
                for mz, intensity in zip(sp.peaks.mz, sp.peaks.intensities):
                    f.write(f'{mz} {intensity}\n')
            f.write('END IONS\n\n')
            
            # 收集元数据
            # name: 优先 compound_name，其次 name
            name_val = sp.get('compound_name') or sp.get('name') or title
            metadata['names'].append(name_val)
            metadata['smiles'].append(sp.get('smiles') or sp.get('SMILES') or '')
            metadata['inchikeys'].append(sp.get('inchikey') or sp.get('INCHIKEY') or '')
            metadata['precursor_mzs'].append(pmz if pmz else 0.0)
            metadata['adducts'].append(sp.get('adduct') or sp.get('precursor_type') or '')
            metadata['ccs_list'].append(sp.get('ccs') or sp.get('CCS') or '')
            # ontology: 从预解析的映射中查找（matchms不解析此字段）
            ontology_val = ontology_map.get(name_val, '')
            metadata['ontologies'].append(ontology_val)
            # formula: 从多个可能的字段提取
            formula_val = (sp.get('formula') or sp.get('FORMULA') or 
                          sp.get('molecular_formula') or sp.get('FORMULA_MOLECULAR') or '')
            metadata['formulas'].append(formula_val)
    
    print(f"  MGF生成: {mgf_path} ({len(spectra)} 条)")
    
    # 保存谱图缓存（供L1 MC匹配复用）
    pkl_path = mgf_path.replace('.mgf', '_spectra_cache.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(spectra, f)
    print(f"  谱图缓存: {pkl_path}")
    
    return len(spectra), metadata, spectra


def compute_embedding(mgf_path, output_npz, batch_size=64, metadata=None):
    """计算DreaMS embedding并保存为NPZ（含元数据）"""
    print(f"  加载 DreaMS 模型...", flush=True)
    from dreams.api import dreams_embeddings

    print(f"  计算DreaMS embedding...", flush=True)
    embeddings = dreams_embeddings(mgf_path, batch_size=batch_size, progress_bar=True)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # 保存embedding和元数据
    save_dict = {'embeddings': embeddings.astype(np.float32)}
    if metadata:
        # 转换为numpy数组
        save_dict['names'] = np.array(metadata['names'], dtype=object)
        save_dict['smiles'] = np.array(metadata['smiles'], dtype=object)
        save_dict['inchikeys'] = np.array(metadata['inchikeys'], dtype=object)
        save_dict['precursor_mzs'] = np.array(metadata['precursor_mzs'], dtype=np.float32)
        save_dict['adducts'] = np.array(metadata['adducts'], dtype=object)
        save_dict['ccs_list'] = np.array(metadata['ccs_list'], dtype=object)
        save_dict['ontologies'] = np.array(metadata['ontologies'], dtype=object)
        save_dict['formulas'] = np.array(metadata['formulas'], dtype=object)  # 添加分子式
    
    np.savez_compressed(output_npz, **save_dict)
    print(f"  保存: {output_npz} (shape={embeddings.shape}, 含元数据)")
    return embeddings.shape


def process_database(db_name, msp_path, force=False, batch_size=64, output_dir=None):
    """处理单个数据库的向量化
    
    参数:
        db_name: 数据库名称
        msp_path: MSP文件路径
        force: 强制重新计算
        batch_size: 批处理大小
        output_dir: 自定义输出目录（样品模式）
    """
    import pickle
    msp_path = Path(msp_path)
    if not msp_path.exists():
        print(f"  [SKIP] 文件不存在: {msp_path}")
        return False, None

    # 输出路径：自定义目录 或 与MSP同目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_npz = os.path.join(output_dir, 'embeddings.npz')
        pkl_path = os.path.join(output_dir, 'spectra_cache.pkl')
    else:
        output_npz = str(msp_path).replace(msp_path.suffix, '_dreams_emb.npz')
        pkl_path = str(msp_path).replace(msp_path.suffix, '_spectra_cache.pkl')

    if os.path.exists(output_npz) and not force:
        existing = np.load(output_npz)
        has_metadata = 'names' in existing
        print(f"  [SKIP] 已有缓存: {output_npz} (shape={existing['embeddings'].shape}, 元数据={'有' if has_metadata else '无'})")
        
        # 检查PKL缓存是否存在，不存在则单独生成
        if not os.path.exists(pkl_path):
            print(f"  [CACHE] 生成谱图缓存（含Ontology提取）...")
            from matchms.importing import load_from_msp
            
            # 预解析 Ontology
            ontology_map_cache = extract_ontology_from_msp(str(msp_path))
            
            spectra = []
            for sp in tqdm(load_from_msp(str(msp_path)), desc="  解析MSP", ncols=80, unit="条"):
                # 补充 ontology 到谱图 metadata
                name_val = sp.get('compound_name') or sp.get('name') or ''
                if name_val and name_val in ontology_map_cache:
                    sp.set('ontology', ontology_map_cache[name_val])
                spectra.append(sp)
            with open(pkl_path, 'wb') as f:
                pickle.dump(spectra, f)
            print(f"  谱图缓存: {pkl_path} ({len(spectra)} 条)")
        
        return True, output_npz

    print(f"  [COMPUTE] {msp_path.name} → embedding...")

    # MSP → 临时MGF + 元数据 + 谱图缓存
    # 样品模式：临时文件放在 output_dir
    if output_dir:
        mgf_path = os.path.join(output_dir, 'temp_for_dreams.mgf')
    else:
        mgf_path = str(msp_path).replace(msp_path.suffix, '_temp_for_dreams.mgf')
    n_spectra, metadata, spectra = msp_to_mgf(str(msp_path), mgf_path)
    
    # 移动 pkl 缓存到正确位置（msp_to_mgf 生成的 pkl 在 mgf 同目录）
    temp_pkl = mgf_path.replace('.mgf', '_spectra_cache.pkl')
    if os.path.exists(temp_pkl) and temp_pkl != pkl_path:
        import shutil
        shutil.move(temp_pkl, pkl_path)
        print(f"  谱图缓存: {pkl_path}")

    if n_spectra == 0:
        print(f"  [ERROR] 无谱图，跳过")
        return False, None

    # 计算embedding（含元数据）
    shape = compute_embedding(mgf_path, output_npz, batch_size, metadata)

    # 清理临时文件（MGF + DreaMS 生成的 HDF5 缓存）
    if os.path.exists(mgf_path):
        os.remove(mgf_path)
        print(f"  [CLEAN] 已删除临时文件: {Path(mgf_path).name}")
    hdf5_path = mgf_path.replace('.mgf', '.hdf5')
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        print(f"  [CLEAN] 已删除临时文件: {Path(hdf5_path).name}")

    return True, output_npz


def merge_msp_files(file_list, output_path):
    """合并多个 MSP 文件为一个"""
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for msp_file in file_list:
            if not os.path.exists(msp_file):
                print(f"  [WARN] 文件不存在，跳过: {msp_file}")
                continue
            with open(msp_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                content = f_in.read()
                if content:
                    f_out.write(content)
                    if not content.endswith('\n'):
                        f_out.write('\n')
            print(f"  [OK] {Path(msp_file).name}")


def main():
    """主函数：支持单文件模式、样品模式和数据库模式"""
    
    # 单文件模式（L2模拟库 或 L1样品）
    if MSP_FILE:
        # 判断是否为 JSON 列表（多文件合并）
        msp_files = None
        try:
            parsed = json.loads(MSP_FILE)
            if isinstance(parsed, list):
                msp_files = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        
        if msp_files and len(msp_files) > 1:
            # 多文件：先合并再向量化，输出到第一个文件的目录
            first_dir = str(Path(msp_files[0]).parent)
            merged_msp = os.path.join(first_dir, "simulated_merged.msp")
            
            print("=" * 70)
            print("L1-预处理: 多文件合并 + DreaMS 向量化")
            print("=" * 70)
            print(f"  输入文件: {len(msp_files)} 个")
            for f in msp_files:
                print(f"    - {f}")
            print(f"  合并输出: {merged_msp}")
            print("=" * 70)
            
            merge_msp_files(msp_files, merged_msp)
            target_msp = merged_msp
        else:
            # 单文件（或列表中只有一个）
            target_msp = msp_files[0] if msp_files else MSP_FILE
            
            # 判断是样品模式还是普通单文件模式
            mode_name = "样品" if OUTPUT_DIR else "单文件"
            print("=" * 70)
            print(f"L1-预处理: {mode_name} DreaMS 向量化")
            print("=" * 70)
            print(f"  输入文件: {target_msp}")
            if OUTPUT_DIR:
                print(f"  输出目录: {OUTPUT_DIR}")
            print("=" * 70)
        
        msp_path = Path(target_msp)
        if not msp_path.exists():
            print(f"[ERROR] 文件不存在: {target_msp}")
            return False
        
        # 输出路径：样品模式用 OUTPUT_DIR，否则与输入文件同目录
        if OUTPUT_DIR:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_npz = os.path.join(OUTPUT_DIR, 'embeddings.npz')
        else:
            output_npz = str(msp_path).replace(msp_path.suffix, '_dreams_emb.npz')
        
        if os.path.exists(output_npz) and not FORCE_COMPUTE:
            existing = np.load(output_npz)
            print(f"[SKIP] 已有缓存: {output_npz} (shape={existing['embeddings'].shape})")
            return True
        
        success, _ = process_database("input", target_msp, FORCE_COMPUTE, BATCH_SIZE, OUTPUT_DIR)
        
        print("\n" + "=" * 70)
        print(f"向量化完成: {'成功' if success else '失败'}")
        print(f"输出文件: {output_npz}")
        print("=" * 70)
        
        return success
    
    # 数据库模式（L1）
    if not ION_MODE:
        raise ValueError("错误：数据库模式需要设置 L1_ION_MODE 环境变量！")
    
    db_config = get_databases_for_mode(ION_MODE)
    if not db_config:
        print(f"[ERROR] 无 {ION_MODE} 模式的数据库配置")
        return False

    print("=" * 70)
    print("L1-预处理: L1数据库 DreaMS 向量化")
    print("=" * 70)
    print(f"  离子模式: {ION_MODE}")
    print(f"  数据库: {', '.join(db_config.keys())}")
    print("=" * 70)

    success_count = 0
    total_count = 0

    for db_name, msp_path in db_config.items():
        if not msp_path:
            print(f"\n[{db_name}] 无{ION_MODE}模式数据，跳过")
            continue

        total_count += 1
        print(f"\n[{db_name}] {ION_MODE}模式")
        print(f"  MSP: {msp_path}")

        if process_database(db_name, msp_path, FORCE_COMPUTE, BATCH_SIZE)[0]:
            success_count += 1

    print("\n" + "=" * 70)
    print(f"数据库向量化完成: {success_count}/{total_count} 成功")
    print("=" * 70)

    return success_count == total_count


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
