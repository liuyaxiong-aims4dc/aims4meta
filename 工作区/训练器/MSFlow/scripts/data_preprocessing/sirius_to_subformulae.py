#!/usr/bin/env python3
"""
将 SIRIUS 6.3+ 生成的碎片树转换为 subformulae JSON 格式
供 MSFlow 编码器训练使用

通过 SIRIUS REST API 提取碎片树数据，转换为兼容 MSFlow 的 subformulae 格式

用法:
    # 使用已运行的 REST 服务批量转换 (推荐，高效)
    python sirius_to_subformulae.py \
        --input /path/to/fragmentation_trees \
        --output /path/to/subformulae_output \
        --use_running_service \
        --port 8901

    # 传统模式：每个文件单独启动服务 (慢)
    python sirius_to_subformulae.py \
        --input /path/to/fragmentation_trees \
        --output /path/to/subformulae_output

注意：
    - SIRIUS REST API 不支持中文路径，请确保输入路径为纯 ASCII
    - 使用 --use_running_service 时，需要先手动启动 SIRIUS 服务：
      sirius service --port 8901 -s
"""
import os
import sys
import argparse
import json
import subprocess
import signal
import time
import re
import requests
from pathlib import Path
from tqdm import tqdm

# SIRIUS 路径
SIRIUS_PATH = '/stor3/AIMS4Meta/源代码/SIRIUS/sirius-6.3.5-linux-x64/sirius/bin/sirius'

# SIRIUS 登录凭证
SIRIUS_USERNAME = "fanhl@whut.edu.cn"
SIRIUS_PASSWORD = "Kongtong@518936"

# 默认 REST 端口
DEFAULT_PORT = 8889

# 单同位素原子质量表
ATOMIC_MASSES = {
    'H': 1.00782503207,
    'C': 12.0,
    'N': 14.0030740048,
    'O': 15.99491461956,
    'P': 30.97376163,
    'S': 31.97207100,
    'F': 18.99840322,
    'Cl': 34.96885268,
    'Br': 78.9183371,
    'I': 126.904473,
    'Si': 27.9769265325,
    'Na': 22.9897692809,
    'K': 38.96370668,
    'Se': 79.9165213,
    'B': 11.0093054,
    'Li': 7.0160034,
    'Fe': 55.9349375,
    'Cu': 62.9295975,
    'Zn': 63.9291422,
    'Mg': 23.9850417,
    'Ca': 39.9625909,
    'Mn': 54.9380451,
    'Co': 58.9331950,
}


def calc_mono_mass(formula: str) -> float:
    """
    根据分子式计算中性单同位素质量
    支持如 'C9H10O2', 'C11NO2H15' 等格式
    """
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    mass = 0.0
    for elem, count in matches:
        if not elem:
            continue
        n = int(count) if count else 1
        if elem in ATOMIC_MASSES:
            mass += ATOMIC_MASSES[elem] * n
    return mass


# 质子质量 (用于 m/z 计算)
PROTON_MASS = 1.00727646677


def calc_theoretical_mz(formula: str, adduct: str) -> float:
    """
    根据碎片分子式和加合物类型计算理论 m/z
    
    SIRIUS 碎片树中的 formula 为中性碎片分子式，
    理论 m/z 需要加上加合物的质量贡献
    """
    neutral_mass = calc_mono_mass(formula)
    if '[M+H]+' in adduct or '[M +H]+' in adduct:
        return neutral_mass + PROTON_MASS
    elif '[M-H]-' in adduct or '[M -H]-' in adduct:
        return neutral_mass - PROTON_MASS
    elif '[M+Na]+' in adduct:
        return neutral_mass + ATOMIC_MASSES.get('Na', 22.9898) - PROTON_MASS + PROTON_MASS
    else:
        # 默认按 [M+H]+ 处理
        return neutral_mass + PROTON_MASS


def parse_args():
    parser = argparse.ArgumentParser(
        description='将 SIRIUS 6.3+ 碎片树转换为 subformulae JSON 格式 (REST API 方案)'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='SIRIUS 输出目录 (包含 .sirius 文件)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='输出目录路径 (每个谱图生成一个 JSON 文件)'
    )
    parser.add_argument(
        '--port', type=int, default=DEFAULT_PORT,
        help=f'SIRIUS REST 服务端口 (默认: {DEFAULT_PORT})'
    )
    parser.add_argument(
        '--sirius_path', type=str, default=SIRIUS_PATH,
        help=f'SIRIUS 可执行文件路径 (默认: {SIRIUS_PATH})'
    )
    parser.add_argument(
        '--use_running_service', action='store_true',
        help='使用已运行的 REST 服务 (推荐，避免重复启动)'
    )
    return parser.parse_args()


def find_sirius_projects(input_dir: str):
    """查找所有 .sirius 文件"""
    sirius_files = sorted(Path(input_dir).rglob('*.sirius'))
    return [str(f) for f in sirius_files]


def login_sirius_cli(sirius_path: str) -> bool:
    """通过 CLI 登录 SIRIUS (持久化 token)"""
    try:
        env = os.environ.copy()
        env['SIRIUS_USERNAME'] = SIRIUS_USERNAME
        env['SIRIUS_PASSWORD'] = SIRIUS_PASSWORD

        result = subprocess.run(
            [sirius_path, 'login',
             '--user-env', 'SIRIUS_USERNAME',
             '--password-env', 'SIRIUS_PASSWORD'],
            capture_output=True, text=True, timeout=30, env=env
        )
        return result.returncode == 0 and 'Login successful' in result.stdout
    except Exception as e:
        print(f"  CLI 登录异常: {e}")
        return False


def start_sirius_service(sirius_path: str, sirius_file: str, port: int):
    """
    启动 SIRIUS REST 服务

    Args:
        sirius_path: SIRIUS 可执行文件路径
        sirius_file: .sirius 项目文件路径
        port: REST 端口

    Returns:
        subprocess.Popen: 服务进程
    """
    cmd = [
        sirius_path,
        '-i', sirius_file,
        'service',
        '--port', str(port),
        '-s'  # 允许 REST shutdown
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # 等待服务启动 (最多 60 秒)
    base_url = f'http://localhost:{port}'
    for i in range(60):
        time.sleep(1)
        try:
            r = requests.get(f'{base_url}/actuator/health', timeout=10)
            if r.status_code == 200:
                return proc
        except (requests.ConnectionError, requests.exceptions.ReadTimeout):
            pass

        # 检查进程是否还活着
        if proc.poll() is not None:
            print(f"  SIRIUS 服务启动失败 (退出码: {proc.returncode})")
            return None

    print("  SIRIUS 服务启动超时")
    proc.terminate()
    return None


def stop_sirius_service(proc, port: int):
    """停止 SIRIUS REST 服务"""
    try:
        # 尝试通过 REST API 优雅关闭
        requests.post(f'http://localhost:{port}/actuator/shutdown', timeout=5)
        proc.wait(timeout=10)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def login_sirius_rest(port: int) -> bool:
    """通过 REST API 登录"""
    try:
        r = requests.post(
            f'http://localhost:{port}/api/account/login',
            params={"acceptTerms": "true"},
            json={"username": SIRIUS_USERNAME, "password": SIRIUS_PASSWORD},
            timeout=15
        )
        return r.status_code == 200
    except Exception as e:
        print(f"  REST 登录异常: {e}")
        return False


def open_project(port: int, project_id: str, sirius_file: str) -> bool:
    """打开 .sirius 项目"""
    try:
        r = requests.put(
            f'http://localhost:{port}/api/projects/{project_id}',
            params={"pathToProject": sirius_file},
            timeout=30
        )
        return r.status_code == 200
    except Exception as e:
        print(f"  打开项目异常: {e}")
        return False


def close_project(port: int, project_id: str):
    """关闭项目"""
    try:
        requests.delete(
            f'http://localhost:{port}/api/projects/{project_id}',
            timeout=10
        )
    except Exception:
        pass


def extract_fragtrees(port: int, project_id: str):
    """
    通过 REST API 提取所有碎片树

    Returns:
        list: [{"name": ..., "cand_form": ..., "cand_ion": ..., "output_tbl": {...}}, ...]
    """
    base = f'http://localhost:{port}/api'
    results = []

    # 获取所有 aligned features
    try:
        r = requests.get(f'{base}/projects/{project_id}/aligned-features', timeout=30)
        if r.status_code != 200:
            print(f"  获取 features 失败: {r.status_code}")
            return results
        features = r.json()
    except Exception as e:
        print(f"  获取 features 异常: {e}")
        return results

    for feat in features:
        fid = feat['alignedFeatureId']
        name = feat.get('name', f'unknown_{fid}')

        # 获取 formulas
        try:
            r = requests.get(
                f'{base}/projects/{project_id}/aligned-features/{fid}/formulas',
                timeout=15
            )
            if r.status_code != 200 or not r.text:
                continue
            formulas = r.json()
            if not formulas:
                continue
        except Exception:
            continue

        # 取 top-1 formula
        top = formulas[0]
        formula_id = top['formulaId']
        mol_formula = top.get('molecularFormula', 'unknown')
        # SIRIUS REST API 返回带空格的 adduct (如 "[M + H]+")，需去除空格以匹配 MSFlow 格式
        adduct = top.get('adduct', '').replace(' ', '')

        # 获取碎片树
        try:
            r = requests.get(
                f'{base}/projects/{project_id}/aligned-features/{fid}/formulas/{formula_id}/fragtree',
                timeout=15
            )
            if r.status_code != 200 or not r.text:
                continue
            tree = r.json()
        except Exception:
            continue

        # 提取碎片 (排除根节点 - 母离子, intensity=0)
        fragments = tree.get('fragments', [])
        sub_frags = [f for f in fragments if f.get('intensity', 0) > 0]

        if not sub_frags:
            continue

        # 计算每个碎片的 mono_mass, abs_mass_diff, mass_diff
        mz_list = [f['mz'] for f in sub_frags]
        inten_list = [f['intensity'] for f in sub_frags]
        formula_list = [f['molecularFormula'] for f in sub_frags]
        ions_list = [f.get('adduct', adduct).replace(' ', '') for f in sub_frags]
        mono_mass_list = [calc_theoretical_mz(fm, adduct) for fm in formula_list]
        abs_mass_diff_list = []
        mass_diff_list = []
        for mz_val, mono_val in zip(mz_list, mono_mass_list):
            abs_diff = abs(mz_val - mono_val)
            ppm_diff = (abs_diff / mono_val * 1e6) if mono_val > 0 else 0.0
            abs_mass_diff_list.append(abs_diff)
            mass_diff_list.append(ppm_diff)

        # 按强度降序排列所有字段 (与 DiffMS/CANOPUS 格式一致)
        sorted_indices = sorted(range(len(inten_list)), key=lambda i: inten_list[i], reverse=True)
        mz_list = [mz_list[i] for i in sorted_indices]
        inten_list = [inten_list[i] for i in sorted_indices]
        mono_mass_list = [mono_mass_list[i] for i in sorted_indices]
        abs_mass_diff_list = [abs_mass_diff_list[i] for i in sorted_indices]
        mass_diff_list = [mass_diff_list[i] for i in sorted_indices]
        formula_list = [formula_list[i] for i in sorted_indices]
        ions_list = [ions_list[i] for i in sorted_indices]

        # 转换为 subformulae 格式 (兼容 MSFlow / CANOPUS)
        subformulae = {
            "name": name,
            "cand_form": mol_formula,
            "cand_ion": adduct,
            "output_tbl": {
                "mz": mz_list,
                "ms2_inten": inten_list,
                "mono_mass": mono_mass_list,
                "abs_mass_diff": abs_mass_diff_list,
                "mass_diff": mass_diff_list,
                "formula": formula_list,
                "ions": ions_list
            }
        }
        results.append(subformulae)

    return results


def process_sirius_file(sirius_file: str, port: int, output_dir: str) -> int:
    """
    处理单个 .sirius 文件

    启动 REST 服务 -> 登录 -> 提取碎片树 -> 保存 JSON -> 关闭

    Returns:
        int: 成功提取的谱图数
    """
    project_id = Path(sirius_file).stem
    print(f"\n处理: {project_id}")
    
    # 检查是否已有 JSON 文件（跳过已处理的批次）
    existing_jsons = list(Path(output_dir).glob(f"{project_id}_*.json"))
    if existing_jsons:
        print(f"  跳过: 已存在 {len(existing_jsons)} 个 JSON 文件")
        return len(existing_jsons)
    
    # 启动 SIRIUS REST 服务
    print(f"  启动 REST 服务 (端口 {port})...")
    proc = start_sirius_service(
        SIRIUS_PATH, sirius_file, port
    )
    if not proc:
        return 0

    try:
        # 登录
        if not login_sirius_rest(port):
            print("  REST 登录失败")
            return 0

        # 打开项目
        if not open_project(port, project_id, sirius_file):
            print("  打开项目失败")
            return 0

        # 提取碎片树
        results = extract_fragtrees(port, project_id)
        print(f"  提取到 {len(results)} 个碎片树")

        # 保存 JSON
        count = 0
        for i, sub in enumerate(results):
            name = sub.pop('name', f'unknown_{i}')
            # 使用 project_id + 索引作为文件名
            filename = f"{project_id}_{i:04d}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sub, f, indent=2, ensure_ascii=False)
            count += 1

        # 关闭项目
        close_project(port, project_id)
        return count

    finally:
        stop_sirius_service(proc, port)


def process_with_running_service(session, sirius_file: str, port: int, output_dir: str) -> int:
    """
    使用已运行的 REST 服务处理单个 .sirius 文件 (高效模式)

    Args:
        session: 已登录的 requests.Session
        sirius_file: .sirius 文件路径
        port: REST 端口
        output_dir: 输出目录

    Returns:
        int: 成功提取的谱图数
    """
    project_id = Path(sirius_file).stem
    base = f'http://localhost:{port}/api'
    
    # 检查是否已有 JSON 文件（跳过已处理的批次）
    existing_jsons = list(Path(output_dir).glob(f"{project_id}_*.json"))
    if existing_jsons:
        return len(existing_jsons)
    
    try:
        # 打开项目
        r = session.put(
            f'{base}/projects/{project_id}',
            params={"pathToProject": sirius_file},
            timeout=60
        )
        if r.status_code != 200:
            print(f"  [{project_id}] 打开失败: {r.status_code}")
            return 0

        # 获取 features
        r = session.get(f'{base}/projects/{project_id}/aligned-features', timeout=30)
        if r.status_code != 200:
            print(f"  [{project_id}] 获取 features 失败: {r.status_code}")
            close_project(port, project_id)
            return 0

        features = r.json()
        if not features:
            print(f"  [{project_id}] 无 features")
            close_project(port, project_id)
            return 0

        # 提取碎片树
        results = []
        for feat in features:
            fid = feat['alignedFeatureId']
            name = feat.get('name', f'unknown_{fid}')

            # 获取 formulas
            try:
                r = session.get(f'{base}/projects/{project_id}/aligned-features/{fid}/formulas', timeout=15)
                if r.status_code != 200 or not r.text:
                    continue
                formulas = r.json()
                if not formulas:
                    continue
            except Exception:
                continue

            top = formulas[0]
            formula_id = top['formulaId']
            mol_formula = top.get('molecularFormula', 'unknown')
            adduct = top.get('adduct', '').replace(' ', '')

            # 获取碎片树
            try:
                r = session.get(
                    f'{base}/projects/{project_id}/aligned-features/{fid}/formulas/{formula_id}/fragtree',
                    timeout=15
                )
                if r.status_code != 200 or not r.text:
                    continue
                tree = r.json()
            except Exception:
                continue

            fragments = tree.get('fragments', [])
            sub_frags = [f for f in fragments if f.get('intensity', 0) > 0]

            if not sub_frags:
                continue

            # 构建输出
            mz_list = [f['mz'] for f in sub_frags]
            inten_list = [f['intensity'] for f in sub_frags]
            formula_list = [f['molecularFormula'] for f in sub_frags]
            ions_list = [f.get('adduct', adduct).replace(' ', '') for f in sub_frags]
            mono_mass_list = [calc_theoretical_mz(fm, adduct) for fm in formula_list]
            abs_mass_diff_list = []
            mass_diff_list = []
            for mz_val, mono_val in zip(mz_list, mono_mass_list):
                abs_diff = abs(mz_val - mono_val)
                ppm_diff = (abs_diff / mono_val * 1e6) if mono_val > 0 else 0.0
                abs_mass_diff_list.append(abs_diff)
                mass_diff_list.append(ppm_diff)

            # 按强度降序排列
            sorted_indices = sorted(range(len(inten_list)), key=lambda i: inten_list[i], reverse=True)
            mz_list = [mz_list[i] for i in sorted_indices]
            inten_list = [inten_list[i] for i in sorted_indices]
            mono_mass_list = [mono_mass_list[i] for i in sorted_indices]
            abs_mass_diff_list = [abs_mass_diff_list[i] for i in sorted_indices]
            mass_diff_list = [mass_diff_list[i] for i in sorted_indices]
            formula_list = [formula_list[i] for i in sorted_indices]
            ions_list = [ions_list[i] for i in sorted_indices]

            results.append({
                "name": name,
                "cand_form": mol_formula,
                "cand_ion": adduct,
                "output_tbl": {
                    "mz": mz_list,
                    "ms2_inten": inten_list,
                    "mono_mass": mono_mass_list,
                    "abs_mass_diff": abs_mass_diff_list,
                    "mass_diff": mass_diff_list,
                    "formula": formula_list,
                    "ions": ions_list
                }
            })

        # 保存 JSON
        count = 0
        for i, sub in enumerate(results):
            name = sub.pop('name', f'unknown_{i}')
            filename = f"{project_id}_{i:04d}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sub, f, indent=2, ensure_ascii=False)
            count += 1

        # 关闭项目
        close_project(port, project_id)
        return count

    except Exception as e:
        print(f"  [{project_id}] 异常: {e}")
        return 0


def main():
    args = parse_args()

    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    # 查找 .sirius 项目文件
    print("\n查找 SIRIUS 项目文件...")
    sirius_files = find_sirius_projects(args.input)
    print(f"找到 {len(sirius_files)} 个 .sirius 项目文件")

    if not sirius_files:
        print("警告: 没有找到 .sirius 文件")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    if args.use_running_service:
        # 使用已运行的 REST 服务 (高效模式)
        print(f"\n使用已运行的 REST 服务 (端口 {args.port})...")
        
        # 创建 session 并登录
        session = requests.Session()
        try:
            r = session.post(
                f'http://localhost:{args.port}/api/account/login',
                params={"acceptTerms": "true"},
                json={"username": SIRIUS_USERNAME, "password": SIRIUS_PASSWORD},
                timeout=120
            )
            if r.status_code != 200:
                print(f"错误: REST 登录失败: {r.status_code}")
                sys.exit(1)
            print("✓ REST 登录成功")
        except Exception as e:
            print(f"错误: 无法连接 REST 服务: {e}")
            sys.exit(1)

        # 批量处理
        total_count = 0
        for i, sirius_file in enumerate(tqdm(sirius_files, desc="处理进度")):
            count = process_with_running_service(session, sirius_file, args.port, args.output)
            total_count += count
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(sirius_files)} 个文件, 累计 {total_count} 个 JSON")

    else:
        # 传统模式：每个文件单独启动服务
        print("\n警告: 传统模式每个文件都会启动/关闭服务，速度较慢")
        print("建议使用 --use_running_service 参数")
        
        # 检查 SIRIUS
        if not os.path.exists(args.sirius_path):
            print(f"错误: SIRIUS 不存在: {args.sirius_path}")
            sys.exit(1)

        # CLI 登录 (持久化 token)
        print("登录 SIRIUS (CLI)...")
        if not login_sirius_cli(args.sirius_path):
            print("警告: SIRIUS CLI 登录失败，尝试继续")
        else:
            print("✓ SIRIUS CLI 登录成功")

        total_count = 0
        for i, sirius_file in enumerate(sirius_files):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(sirius_files)}] {Path(sirius_file).name}")
            print(f"{'='*60}")

            count = process_sirius_file(sirius_file, args.port, args.output)
            total_count += count

    # 最终统计
    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"  处理文件: {len(sirius_files)} 个 .sirius")
    print(f"  生成 JSON: {total_count} 个")
    print(f"  输出目录: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
