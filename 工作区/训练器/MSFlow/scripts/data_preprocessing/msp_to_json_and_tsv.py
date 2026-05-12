#!/usr/bin/env python3
"""
从MSP文件生成JSON和TSV (方案2: 分离步骤)

核心逻辑：
1. 读取MSP文件
2. 使用 run_sirius_batch.py 生成碎片树 (.sirius文件)
3. 使用 sirius_to_subformulae.py 转换为 subformulae JSON
4. 从MSP元数据中获取信息，同步写入TSV

用法:
    python msp_to_json_and_tsv.py \
        --msp_file /path/to/spectraverse.msp \
        --output_dir /path/to/output \
        --prefix spectraverse \
        --batch_size 500
"""
import os
import sys
import argparse
import json
import csv
import subprocess
import tempfile
import shutil
import time
import requests
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# SIRIUS 路径
SIRIUS_PATH = "/stor1/AIMS4Meta/code/SIRIUS/sirius-6.3.3-linux-x64/sirius/bin/sirius"

ATOMIC_MASSES = {
    'H': 1.00782503207, 'C': 12.0, 'N': 14.0030740048, 'O': 15.99491461956,
    'P': 30.97376163, 'S': 31.97207100, 'F': 18.99840322, 'Cl': 34.96885268,
    'Br': 78.9183371, 'I': 126.904473, 'Si': 27.9769265325, 'Na': 22.9897692809,
    'K': 38.96370668, 'Se': 79.9165213, 'B': 11.0093054,
}
PROTON_MASS = 1.00727646677

def _calc_mono_mass(formula: str) -> float:
    import re
    mass = 0.0
    for elem, count in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
        if elem in ATOMIC_MASSES:
            mass += ATOMIC_MASSES[elem] * (int(count) if count else 1)
    return mass

def _calc_theoretical_mz(formula: str, adduct: str) -> float:
    m = _calc_mono_mass(formula)
    if '[M+H]+' in adduct:   return m + PROTON_MASS
    elif '[M-H]-' in adduct: return m - PROTON_MASS
    elif '[M+Na]+' in adduct: return m + ATOMIC_MASSES['Na'] - PROTON_MASS + PROTON_MASS
    return m + PROTON_MASS

# run_sirius_batch.py 路径
RUN_SIRIUS_BATCH_PATH = "/stor1/AIMS4Meta/trainers/denovo/2_phase/scripts/data_preprocessing/run_sirius_batch.py"

# sirius_to_subformulae.py 路径
SIRIUS_TO_SUBFORMULAE_PATH = "/stor1/AIMS4Meta/trainers/denovo/2_phase/scripts/data_preprocessing/sirius_to_subformulae.py"

# SIRIUS 登录凭证
SIRIUS_USERNAME = "fanhl@whut.edu.cn"
SIRIUS_PASSWORD = "Kongtong@518936"


def standardize_ionization(ion: str) -> str:
    """标准化 ionization 表示"""
    # 正离子：铵根加合物
    if ion == '[M+NH4]+':
        return '[M+H3N+H]+'
    
    # 负离子：甲酸加合物
    if ion in ['[M+CH2O2-H]-', '[M+HCOOH-H]-']:
        return '[M+HCOOH-H]-'
    
    # 负离子：乙酸加合物
    if ion in ['[M+C2H4O2-H]-', '[M+CH3COOH-H]-']:
        return '[M+CH3COOH-H]-'
    
    return ion


def parse_args():
    parser = argparse.ArgumentParser(
        description='从MSP文件完整流程生成JSON和TSV'
    )
    parser.add_argument(
        '--msp_file', type=str, required=True,
        help='MSP 文件路径'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--prefix', type=str, default='spectraverse',
        help='spec ID 前缀 (默认: spectraverse)'
    )
    parser.add_argument(
        '--start_id', type=int, default=1,
        help='起始 ID (默认: 1)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=500,
        help='每批处理的谱图数 (默认: 500)'
    )
    parser.add_argument(
        '--profile', type=str, default='qtof',
        choices=['qtof', 'orbitrap', 'fticr'],
        help='仪器类型 (默认: qtof)'
    )
    parser.add_argument(
        '--sirius_path', type=str, default=SIRIUS_PATH,
        help=f'SIRIUS 可执行文件路径 (默认: {SIRIUS_PATH})'
    )
    parser.add_argument(
        '--port', type=int, default=8889,
        help='REST API 端口 (默认: 8889)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='试运行，不实际执行SIRIUS'
    )
    return parser.parse_args()


def parse_msp_spectrum(lines):
    """解析MSP谱图"""
    metadata = {
        'name': '',
        'formula': '',
        'smiles': '',
        'inchikey': '',
        'ionization': '',
        'precursor_mz': '',
        'collision_energy': ''
    }
    peaks = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Name:'):
            metadata['name'] = line.split(':', 1)[1].strip()
        elif line.startswith('Formula:'):
            metadata['formula'] = line.split(':', 1)[1].strip()
        elif line.startswith('SMILES:'):
            metadata['smiles'] = line.split(':', 1)[1].strip()
        elif line.startswith('InChIKey:'):
            metadata['inchikey'] = line.split(':', 1)[1].strip()
        elif line.startswith('PrecursorType:'):
            metadata['ionization'] = line.split(':', 1)[1].strip()
        elif line.startswith('PrecursorMZ:'):
            metadata['precursor_mz'] = line.split(':', 1)[1].strip()
        elif line.startswith('CollisionEnergy:'):
            metadata['collision_energy'] = line.split(':', 1)[1].strip()
        elif line.startswith('NumPeaks:'):
            continue
        else:
            # 解析峰数据
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    peaks.append((mz, intensity))
                except ValueError:
                    continue
    
    return metadata, peaks


def read_msp_file(msp_file: str):
    """读取MSP文件，返回所有谱图"""
    print(f"读取MSP文件: {msp_file}")
    spectra = []
    current_spectrum = []
    
    with open(msp_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '' and current_spectrum:
                metadata, peaks = parse_msp_spectrum(current_spectrum)
                if metadata['name'] and peaks:
                    spectra.append((metadata, peaks))
                current_spectrum = []
            else:
                current_spectrum.append(line)
        
        # 处理最后一个谱图
        if current_spectrum:
            metadata, peaks = parse_msp_spectrum(current_spectrum)
            if metadata['name'] and peaks:
                spectra.append((metadata, peaks))
    
    print(f"  读取到 {len(spectra)} 个谱图")
    return spectra


def login_sirius(sirius_path: str) -> bool:
    """登录SIRIUS"""
    try:
        env = os.environ.copy()
        env['SIRIUS_USERNAME'] = SIRIUS_USERNAME
        env['SIRIUS_PASSWORD'] = SIRIUS_PASSWORD
        
        result = subprocess.run(
            [sirius_path, 'login', '--user-env', 'SIRIUS_USERNAME', '--password-env', 'SIRIUS_PASSWORD'],
            capture_output=True, text=True, timeout=30, env=env
        )
        return result.returncode == 0 and 'Login successful' in result.stdout
    except Exception as e:
        print(f"登录异常: {e}")
        return False


def write_batch_msp(spectra_batch, output_file):
    """写入批次MSP文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for metadata, peaks in spectra_batch:
            f.write(f"Name: {metadata['name']}\n")
            if metadata['formula']:
                f.write(f"Formula: {metadata['formula']}\n")
            if metadata['smiles']:
                f.write(f"SMILES: {metadata['smiles']}\n")
            if metadata['inchikey']:
                f.write(f"InChIKey: {metadata['inchikey']}\n")
            if metadata['ionization']:
                f.write(f"PrecursorType: {metadata['ionization']}\n")
            if metadata['precursor_mz']:
                f.write(f"PrecursorMZ: {metadata['precursor_mz']}\n")
            if metadata['collision_energy']:
                f.write(f"CollisionEnergy: {metadata['collision_energy']}\n")
            
            f.write(f"NumPeaks: {len(peaks)}\n")
            for mz, intensity in peaks:
                f.write(f"{mz} {intensity}\n")
            f.write("\n")


def run_sirius_batch(batch_msp: str, output_dir: str, profile: str, sirius_path: str) -> bool:
    """对批次MSP运行SIRIUS完整流程 (formula + zodiac + structure)
    
    Args:
        batch_msp: 批次MSP文件路径
        output_dir: 输出目录路径
        profile: 仪器profile
        sirius_path: SIRIUS可执行文件路径
    
    Returns:
        str: .sirius项目文件路径，失败返回None
    """
    # SIRIUS输出文件名 = output_dir + .sirius（在MSP所在目录）
    batch_sirius = f"{output_dir}.sirius"
    
    # 运行完整的SIRIUS流程：formula -> zodiac -> structure
    # 这样才能生成碎片树 (aligned-features)
    cmd = [
        sirius_path,
        '--input', batch_msp,
        '--output', output_dir,
        'formula',
        '--profile', profile,
        '--candidates', '1',
        '--no-recalibration',
        'zodiac',  # 分子式排序
        'structure'  # 结构解析（生成碎片树）
    ]
    
    try:
        print(f"    运行SIRIUS完整流程 (formula+zodiac+structure)...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        # 检查.sirius文件是否生成（在MSP所在目录）
        if result.returncode == 0 and os.path.exists(batch_sirius):
            print(f"    SIRIUS成功: {batch_sirius}", flush=True)
            return batch_sirius
        else:
            print(f"    SIRIUS运行失败: {result.stdout[-300:] if result.stdout else 'no output'}", flush=True)
            if result.stderr:
                print(f"    stderr: {result.stderr[:300:]}", flush=True)
            return None
    except subprocess.TimeoutExpired:
        print(f"    SIRIUS运行超时 (30分钟)", flush=True)
        return None
    except Exception as e:
        print(f"    SIRIUS异常: {e}", flush=True)
        return None


def extract_fragtree_from_sirius_via_rest(sirius_file: str, port: int, batch_metadata: list):
    """
    通过REST API从SIRIUS文件提取碎片树
    
    Args:
        sirius_file: .sirius文件路径
        port: REST端口
        batch_metadata: 批次的元数据列表
    
    Returns:
        list: [(metadata, fragtree_data), ...]
    """
    base = f'http://localhost:{port}/api'
    project_id = Path(sirius_file).stem
    
    # 启动SIRIUS服务
    proc = subprocess.Popen(
        [SIRIUS_PATH, '-i', sirius_file, 'service', '--port', str(port), '-s'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # 等待服务启动
    print(f"    等待REST服务启动 (端口 {port})...", flush=True)
    for i in range(60):
        time.sleep(1)
        try:
            r = requests.get(f'http://localhost:{port}/actuator/health', timeout=10)
            if r.status_code == 200:
                print(f"    REST服务已启动 ({i+1}秒)", flush=True)
                break
        except:
            pass
        if proc.poll() is not None:
            print(f"    REST服务进程已退出!", flush=True)
            return []
    else:
        print(f"    REST服务启动超时!", flush=True)
        proc.terminate()
        return []
    
    try:
        # 登录
        print(f"    REST登录...", flush=True)
        r = requests.post(
            f'{base}/account/login',
            params={"acceptTerms": "true"},
            json={"username": SIRIUS_USERNAME, "password": SIRIUS_PASSWORD},
            timeout=15
        )
        if r.status_code != 200:
            print(f"    REST登录失败: {r.status_code}", flush=True)
            return []
        print(f"    REST登录成功", flush=True)
        
        # 打开项目
        print(f"    打开项目: {project_id}", flush=True)
        r = requests.put(
            f'{base}/projects/{project_id}',
            params={"pathToProject": sirius_file},
            timeout=30
        )
        if r.status_code != 200:
            print(f"    打开项目失败: {r.status_code}", flush=True)
            return []
        print(f"    打开项目成功", flush=True)
        
        # 获取features (SIRIUS 6.3 formula命令使用/features而非/aligned-features)
        print(f"    获取features...", flush=True)
        r = requests.get(f'{base}/projects/{project_id}/features', timeout=30)
        if r.status_code != 200:
            print(f"    获取features失败: {r.status_code}", flush=True)
            return []
        
        features = r.json()
        print(f"    获取到 {len(features)} 个features", flush=True)
        results = []
        
        # 提取碎片树
        print(f"    提取碎片树...", flush=True)
        for feat in features:
            fid = feat['alignedFeatureId']
            name = feat.get('name', f'unknown_{fid}')
            
            # 获取formulas
            r = requests.get(f'{base}/projects/{project_id}/aligned-features/{fid}/formulas', timeout=15)
            if r.status_code != 200 or not r.text:
                print(f"      {name[:30]}: 获取formulas失败", flush=True)
                continue
            formulas = r.json()
            if not formulas:
                print(f"      {name[:30]}: 无formulas", flush=True)
                continue
            
            top = formulas[0]
            formula_id = top['formulaId']
            mol_formula = top.get('molecularFormula', 'unknown')
            adduct = top.get('adduct', '').replace(' ', '')
            
            # 获取碎片树
            r = requests.get(
                f'{base}/projects/{project_id}/aligned-features/{fid}/formulas/{formula_id}/fragtree',
                timeout=15
            )
            if r.status_code != 200 or not r.text:
                print(f"      {name[:30]}: 获取fragtree失败", flush=True)
                continue
            tree = r.json()
            
            # 提取碎片
            fragments = tree.get('fragments', [])
            sub_frags = [f for f in fragments if f.get('intensity', 0) > 0]
            
            if not sub_frags:
                print(f"      {name[:30]}: 无碎片", flush=True)
                continue
            
            # 构建碎片树数据
            mz_list = [f['mz'] for f in sub_frags]
            inten_list = [f['intensity'] for f in sub_frags]
            formula_list = [f['molecularFormula'] for f in sub_frags]
            ions_list = [f.get('adduct', adduct).replace(' ', '') for f in sub_frags]
            
            # 按强度降序排列
            sorted_indices = sorted(range(len(inten_list)), key=lambda i: inten_list[i], reverse=True)
            mz_list = [mz_list[i] for i in sorted_indices]
            inten_list = [inten_list[i] for i in sorted_indices]
            formula_list = [formula_list[i] for i in sorted_indices]
            ions_list = [ions_list[i] for i in sorted_indices]
            
            fragtree_data = {
                "cand_form": mol_formula,
                "cand_ion": adduct,
                "output_tbl": {
                    "mz": mz_list,
                    "ms2_inten": inten_list,
                    "formula": formula_list,
                    "ions": ions_list
                }
            }
            
            # 查找对应的元数据
            metadata = None
            for meta in batch_metadata:
                if meta['name'] == name:
                    metadata = meta
                    break
            
            if metadata:
                results.append((metadata, fragtree_data))
                print(f"      {name[:30]}: 匹配成功 ({len(sub_frags)}碎片)", flush=True)
            else:
                print(f"      {name[:30]}: 未找到匹配的元数据!", flush=True)
                # 打印可用的元数据名称供调试
                if len(results) == 0:
                    print(f"      可用元数据: {[m['name'][:30] for m in batch_metadata[:3]]}...", flush=True)
        
        # 关闭项目 (注意：base已包含/api，不需要重复)
        requests.delete(f'{base}/projects/{project_id}', timeout=10)
        
        return results
    
    finally:
        # 关闭服务
        try:
            requests.post(f'{base}/actuator/shutdown', timeout=5)
            proc.wait(timeout=10)
        except:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except:
                proc.kill()


def split_msp_to_batches(msp_file: str, temp_dir: str, batch_size: int):
    """将MSP文件分割为批次，返回 (batch_files, all_metadata)"""
    os.makedirs(temp_dir, exist_ok=True)
    spectra = []
    current = []
    all_metadata = []

    with open(msp_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                if current:
                    spectra.append(current)
                    current = []
            else:
                current.append(line)
    if current:
        spectra.append(current)

    print(f"总共 {len(spectra)} 个谱图")

    # 提取元数据
    for sp in spectra:
        meta = {'name': '', 'formula': '', 'smiles': '', 'inchikey': '',
                'ionization': '', 'precursor_mz': '', 'collision_energy': ''}
        for line in sp:
            line = line.strip()
            if line.startswith('Name:'):          meta['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('Formula:'):     meta['formula'] = line.split(':', 1)[1].strip()
            elif line.startswith('SMILES:'):      meta['smiles'] = line.split(':', 1)[1].strip()
            elif line.startswith('InChIKey:'):    meta['inchikey'] = line.split(':', 1)[1].strip()
            elif line.startswith('PrecursorType:'): meta['ionization'] = line.split(':', 1)[1].strip()
            elif line.startswith('PrecursorMZ:'): meta['precursor_mz'] = line.split(':', 1)[1].strip()
            elif line.startswith('CollisionEnergy:'): meta['collision_energy'] = line.split(':', 1)[1].strip()
        all_metadata.append(meta)

    # 写批次文件
    batch_files = []
    for i in range(0, len(spectra), batch_size):
        batch = spectra[i:i + batch_size]
        batch_file = os.path.join(temp_dir, f'batch_{i//batch_size:05d}.msp')
        if not os.path.exists(batch_file):
            with open(batch_file, 'w', encoding='utf-8') as f:
                for sp in batch:
                    f.writelines(sp)
                    f.write('\n')
        batch_files.append((batch_file, all_metadata[i:i + batch_size]))

    print(f"分割成 {len(batch_files)} 个批次")
    return batch_files


def run_sirius_one_batch(batch_msp: str, sirius_dir: str, profile: str, sirius_path: str) -> bool:
    """对单个批次MSP运行SIRIUS，返回是否成功"""
    batch_name = Path(batch_msp).stem
    out_file = os.path.join(sirius_dir, f"{batch_name}.sirius")
    if os.path.exists(out_file):
        return True
    cmd = [
        sirius_path, '--input', batch_msp, '--output', out_file,
        'formula', '--profile', profile, '--candidates', '1', '--no-recalibration'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and os.path.exists(out_file)
    except Exception:
        return os.path.exists(out_file)


def convert_sirius_to_jsons(sirius_file: str, port: int) -> list:
    """
    启动SIRIUS REST服务，提取碎片树，返回 list of subformulae dict (含 'name' 字段)
    """
    import time
    project_id = Path(sirius_file).stem
    base = f'http://localhost:{port}/api'

    proc = subprocess.Popen(
        [SIRIUS_PATH, '-i', sirius_file, 'service', '--port', str(port), '-s'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    # 等待服务启动
    for _ in range(60):
        time.sleep(1)
        try:
            if requests.get(f'http://localhost:{port}/actuator/health', timeout=5).status_code == 200:
                break
        except Exception:
            pass
        if proc.poll() is not None:
            return []
    else:
        proc.terminate()
        return []

    results = []
    try:
        # 登录
        try:
            r = requests.post(f'{base}/account/login', params={"acceptTerms": "true"},
                              json={"username": SIRIUS_USERNAME, "password": SIRIUS_PASSWORD}, timeout=60)
            if r.status_code != 200:
                return []
        except Exception as e:
            print(f"    REST登录失败: {e}")
            return []

        # 打开项目
        r = requests.put(f'{base}/projects/{project_id}',
                         params={"pathToProject": sirius_file}, timeout=30)
        if r.status_code != 200:
            return []

        # 获取features
        r = requests.get(f'{base}/projects/{project_id}/aligned-features', timeout=30)
        if r.status_code != 200:
            return []
        features = r.json()

        for feat in features:
            fid = feat['alignedFeatureId']
            name = feat.get('name', f'unknown_{fid}')
            try:
                r = requests.get(f'{base}/projects/{project_id}/aligned-features/{fid}/formulas', timeout=15)
                if r.status_code != 200 or not r.text:
                    continue
                formulas = r.json()
                if not formulas:
                    continue
                top = formulas[0]
                formula_id = top['formulaId']
                mol_formula = top.get('molecularFormula', 'unknown')
                adduct = top.get('adduct', '').replace(' ', '')

                r = requests.get(
                    f'{base}/projects/{project_id}/aligned-features/{fid}/formulas/{formula_id}/fragtree',
                    timeout=15)
                if r.status_code != 200 or not r.text:
                    continue
                tree = r.json()
            except Exception:
                continue

            fragments = tree.get('fragments', [])
            sub_frags = [f for f in fragments if f.get('intensity', 0) > 0]
            if not sub_frags:
                continue

            mz_list = [f['mz'] for f in sub_frags]
            inten_list = [f['intensity'] for f in sub_frags]
            formula_list = [f['molecularFormula'] for f in sub_frags]
            ions_list = [f.get('adduct', adduct).replace(' ', '') for f in sub_frags]
            mono_mass_list = [_calc_theoretical_mz(fm, adduct) for fm in formula_list]
            abs_diff_list = [abs(mz - mono) for mz, mono in zip(mz_list, mono_mass_list)]
            ppm_list = [(abs_diff / mono * 1e6) if mono > 0 else 0.0
                        for abs_diff, mono in zip(abs_diff_list, mono_mass_list)]

            ion_mass = feat.get('ionMass')  # precursor m/z from SIRIUS feature
            sorted_idx = sorted(range(len(inten_list)), key=lambda i: inten_list[i], reverse=True)
            results.append({
                "name": name,
                "ion_mass": ion_mass,
                "cand_form": mol_formula,
                "cand_ion": adduct,
                "output_tbl": {
                    "mz": [mz_list[i] for i in sorted_idx],
                    "ms2_inten": [inten_list[i] for i in sorted_idx],
                    "mono_mass": [mono_mass_list[i] for i in sorted_idx],
                    "abs_mass_diff": [abs_diff_list[i] for i in sorted_idx],
                    "mass_diff": [ppm_list[i] for i in sorted_idx],
                    "formula": [formula_list[i] for i in sorted_idx],
                    "ions": [ions_list[i] for i in sorted_idx],
                }
            })

        requests.delete(f'{base}/projects/{project_id}', timeout=10)
        return results
    finally:
        try:
            requests.post(f'{base}/actuator/shutdown', timeout=5)
            proc.wait(timeout=10)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def main():
    args = parse_args()

    if not os.path.exists(args.msp_file):
        print(f"错误: MSP文件不存在: {args.msp_file}")
        sys.exit(1)

    # 目录结构
    os.makedirs(args.output_dir, exist_ok=True)
    json_output_dir = os.path.join(args.output_dir, 'subformulae')
    os.makedirs(json_output_dir, exist_ok=True)
    sirius_output_dir = os.path.join(args.output_dir, 'sirius_temp')
    temp_batches_dir = os.path.join(sirius_output_dir, 'temp_batches')
    sirius_files_dir = os.path.join(sirius_output_dir, 'fragmentation_trees')
    os.makedirs(sirius_files_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MSP -> JSON + TSV (逐批流水线)")
    print(f"{'='*60}")
    print(f"MSP文件: {args.msp_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"Spec ID 前缀: {args.prefix}")
    print(f"批次大小: {args.batch_size}")

    # 检查SIRIUS
    result = subprocess.run([args.sirius_path, '--version'], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print("错误: SIRIUS 不可用")
        sys.exit(1)

    # 分割MSP
    print(f"\n[1] 分割MSP文件...")
    batch_list = split_msp_to_batches(args.msp_file, temp_batches_dir, args.batch_size)
    total_batches = len(batch_list)

    # 确定已完成的批次数（用于断点续跑时恢复spec_id_counter）
    tsv_file = os.path.join(args.output_dir, 'labels.tsv')
    tsv_exists = os.path.exists(tsv_file)
    existing_spec_count = 0
    if tsv_exists:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            existing_spec_count = sum(1 for line in f) - 1  # 减去header
        print(f"  检测到已有TSV，已处理 {existing_spec_count} 条，从 {args.prefix}_{args.start_id + existing_spec_count:06d} 继续")

    spec_id_counter = args.start_id + existing_spec_count
    total_json = existing_spec_count
    total_failed = 0
    failed_batches = []  # [(batch_idx, batch_msp, batch_meta, sirius_file), ...]

    # 打开TSV（追加模式）
    tsv_mode = 'a' if tsv_exists else 'w'
    with open(tsv_file, tsv_mode, newline='', encoding='utf-8') as tsv_handle:
        fieldnames = ['spec', 'name', 'formula', 'smiles', 'inchikey',
                      'ionization', 'precursor_mz', 'collision_energy']
        tsv_writer = csv.DictWriter(tsv_handle, fieldnames=fieldnames, delimiter='\t')
        if not tsv_exists:
            tsv_writer.writeheader()

        print(f"\n[2] 逐批处理 (共 {total_batches} 批)...")
        for batch_idx, (batch_msp, batch_meta) in enumerate(batch_list):
            batch_name = Path(batch_msp).stem
            sirius_file = os.path.join(sirius_files_dir, f"{batch_name}.sirius")

            # 检查该批次JSON是否已全部生成（断点续跑）
            # 用TSV行数判断：如果已处理的行数覆盖了本批次，跳过
            batch_start_global = batch_idx * args.batch_size
            if batch_start_global + len(batch_meta) <= existing_spec_count:
                print(f"  [{batch_idx+1}/{total_batches}] {batch_name}: 已处理，跳过")
                continue

            print(f"\n  [{batch_idx+1}/{total_batches}] {batch_name} ({len(batch_meta)} 谱图)")

            # 运行SIRIUS
            if not os.path.exists(sirius_file):
                print(f"    运行 SIRIUS...")
                ok = run_sirius_one_batch(batch_msp, sirius_files_dir, args.profile, args.sirius_path)
                if not ok:
                    print(f"    SIRIUS 失败，跳过本批次")
                    total_failed += len(batch_meta)
                    continue
            else:
                print(f"    .sirius 已存在，跳过SIRIUS")

            # 转换为JSON
            print(f"    提取碎片树 (REST API)...")
            try:
                fragtrees = convert_sirius_to_jsons(sirius_file, args.port)
            except Exception as e:
                print(f"    提取异常: {e}，记录失败批次")
                fragtrees = []
            print(f"    提取到 {len(fragtrees)} 个碎片树")

            if not fragtrees:
                print(f"    无碎片树，记录失败批次")
                failed_batches.append((batch_idx, batch_msp, batch_meta, sirius_file))
                total_failed += len(batch_meta)
                continue

            # 建立 (name, precursor_mz) -> [meta, ...] 映射，按碰撞能量排序保证消费顺序稳定
            from collections import defaultdict
            name_mz_to_metas = defaultdict(list)
            for m in batch_meta:
                key = (m['name'], m['precursor_mz'])
                name_mz_to_metas[key].append(m)
            # 消费计数器
            name_mz_consumed = defaultdict(int)

            # 写JSON + TSV
            batch_written = 0
            for tree in fragtrees:
                name = tree.pop('name', '')
                ion_mass = tree.pop('ion_mass', None)  # precursor m/z from SIRIUS feature
                metadata = None

                if ion_mass is not None:
                    # 精确匹配：(name, precursor_mz)，用 ionMass 找最近的 key
                    best_key = None
                    best_diff = float('inf')
                    for (n, mz) in name_mz_to_metas:
                        if n == name:
                            try:
                                diff = abs(float(mz) - ion_mass)
                            except (TypeError, ValueError):
                                diff = float('inf')
                            if diff < best_diff:
                                best_diff = diff
                                best_key = (n, mz)
                    if best_key and best_diff < 0.02:  # 20 mDa 容差
                        idx = name_mz_consumed[best_key]
                        metas = name_mz_to_metas[best_key]
                        if idx < len(metas):
                            metadata = metas[idx]
                            name_mz_consumed[best_key] += 1
                else:
                    # 无 ionMass 时退回顺序消费
                    for (n, mz), metas in name_mz_to_metas.items():
                        if n == name:
                            idx = name_mz_consumed[(n, mz)]
                            if idx < len(metas):
                                metadata = metas[idx]
                                name_mz_consumed[(n, mz)] += 1
                                break
                if not metadata:
                    continue

                spec_id = f"{args.prefix}_{spec_id_counter:06d}"
                json_path = os.path.join(json_output_dir, f"{spec_id}.json")
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(tree, jf, indent=2, ensure_ascii=False)

                ion = standardize_ionization(tree['cand_ion'])
                tsv_writer.writerow({
                    'spec': spec_id,
                    'name': metadata['name'],
                    'formula': metadata['formula'],
                    'smiles': metadata['smiles'],
                    'inchikey': metadata['inchikey'],
                    'ionization': ion,
                    'precursor_mz': metadata['precursor_mz'],
                    'collision_energy': metadata['collision_energy']
                })
                tsv_handle.flush()

                spec_id_counter += 1
                batch_written += 1
                total_json += 1

            print(f"    写入 {batch_written} 条 (累计 {total_json})")

    print(f"\n{'='*60}")
    print(f"完成! 生成 {total_json} 个JSON + TSV记录，失败 {total_failed} 个")
    print(f"JSON目录: {json_output_dir}")
    print(f"TSV文件: {tsv_file}")
    print(f"{'='*60}")

    # 重试失败批次（最多2次）
    if failed_batches:
        failed_log = os.path.join(args.output_dir, 'failed_batches.txt')
        with open(failed_log, 'w') as fl:
            for batch_idx, batch_msp, _, _ in failed_batches:
                fl.write(f"batch_idx={batch_idx}\t{batch_msp}\n")
        print(f"\n失败批次已记录: {failed_log}")
        print(f"开始重试 {len(failed_batches)} 个失败批次...")

        with open(tsv_file, 'a', newline='', encoding='utf-8') as tsv_handle:
            fieldnames = ['spec', 'name', 'formula', 'smiles', 'inchikey',
                          'ionization', 'precursor_mz', 'collision_energy']
            tsv_writer = csv.DictWriter(tsv_handle, fieldnames=fieldnames, delimiter='\t')

            for retry in range(2):
                still_failed = []
                for batch_idx, batch_msp, batch_meta, sirius_file in failed_batches:
                    print(f"\n  [重试{retry+1}] batch_idx={batch_idx}")
                    try:
                        fragtrees = convert_sirius_to_jsons(sirius_file, args.port)
                    except Exception as e:
                        print(f"    重试异常: {e}")
                        fragtrees = []

                    if not fragtrees:
                        still_failed.append((batch_idx, batch_msp, batch_meta, sirius_file))
                        continue

                    name_mz_to_metas = defaultdict(list)
                    for m in batch_meta:
                        name_mz_to_metas[(m['name'], m['precursor_mz'])].append(m)
                    name_mz_consumed = defaultdict(int)

                    batch_written = 0
                    for tree in fragtrees:
                        name = tree.pop('name', '')
                        ion_mass = tree.pop('ion_mass', None)
                        metadata = None
                        if ion_mass is not None:
                            best_key = None
                            best_diff = float('inf')
                            for (n, mz) in name_mz_to_metas:
                                if n == name:
                                    try:
                                        diff = abs(float(mz) - ion_mass)
                                    except (TypeError, ValueError):
                                        diff = float('inf')
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_key = (n, mz)
                            if best_key and best_diff < 0.02:
                                idx = name_mz_consumed[best_key]
                                metas = name_mz_to_metas[best_key]
                                if idx < len(metas):
                                    metadata = metas[idx]
                                    name_mz_consumed[best_key] += 1
                        else:
                            for (n, mz), metas in name_mz_to_metas.items():
                                if n == name:
                                    idx = name_mz_consumed[(n, mz)]
                                    if idx < len(metas):
                                        metadata = metas[idx]
                                        name_mz_consumed[(n, mz)] += 1
                                        break
                        if not metadata:
                            continue
                        spec_id = f"{args.prefix}_{spec_id_counter:06d}"
                        json_path = os.path.join(json_output_dir, f"{spec_id}.json")
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump(tree, jf, indent=2, ensure_ascii=False)
                        ion = standardize_ionization(tree['cand_ion'])
                        tsv_writer.writerow({
                            'spec': spec_id, 'name': metadata['name'],
                            'formula': metadata['formula'], 'smiles': metadata['smiles'],
                            'inchikey': metadata['inchikey'], 'ionization': ion,
                            'precursor_mz': metadata['precursor_mz'],
                            'collision_energy': metadata['collision_energy']
                        })
                        tsv_handle.flush()
                        spec_id_counter += 1
                        batch_written += 1
                        total_json += 1
                    print(f"    重试写入 {batch_written} 条")

                failed_batches = still_failed
                if not failed_batches:
                    print("  所有失败批次重试成功!")
                    break

        if failed_batches:
            print(f"\n仍有 {len(failed_batches)} 个批次无法处理，已记录到 {failed_log}")
        print(f"\n最终: 生成 {total_json} 个JSON + TSV记录")


if __name__ == '__main__':
    main()
