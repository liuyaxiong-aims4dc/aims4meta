#!/usr/bin/env python3
"""
SMILES 修复工具（辅助功能模块）

修复策略：
1. 双斜杠修复、金属电荷标准化、分隔符移除、括号鉴定
2. 多组分盐类（含"."）→ 移除无机离子，取最大分子量有机组分
3. RDKit Uncharger 质子化可逆电荷 + 手动质子化增强
4. 季铵盐检测：永久正电荷，标记为失败

使用方式：
    from SMILES修复 import fix_smiles, batch_fix_smiles
    
    fixed_smi, fix_msg = fix_smiles(original_smiles)
    candidates = batch_fix_smiles(smiles_list, show_progress=True)
"""

import re
from typing import Optional, Tuple, List, Dict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def fix_smiles(smiles: str, ion_mode: str = "[M+H]+") -> Tuple[Optional[str], str]:
    """
    修复 SMILES 字符串中的常见问题。

    参数:
        smiles: 原始 SMILES 字符串
        ion_mode: 离子模式（用于日志，不影响修复逻辑）

    返回:
        (修复后的SMILES, 修复说明)
        如果修复失败返回 (None, 失败原因)
    """
    if not isinstance(smiles, str):
        return None, "输入不是字符串"

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        smiles = smiles.strip()
        if not smiles:
            return None, "空SMILES"

        # 去除换行符和多余空白（CSV中SMILES可能跨行存储）
        if '\n' in smiles or '\r' in smiles:
            smiles = ''.join(smiles.split())  # 合并为单行，不留空格
            fixes_pre = ["去除换行符"]
        else:
            fixes_pre = []

        # 过滤分子式（如C15H26O、C30H48O4，不是SMILES）
        # 判据：无SMILES语法字符 + 存在元素后跟多位数（如C15、H26），SMILES环闭合只有单位数
        _smiles_syntax = set('()[]@-=/\\#.\\:')
        if not any(c in smiles for c in _smiles_syntax):
            if re.search(r'[A-Z][a-z]?\d{2,}', smiles) or re.search(r'H\d', smiles):
                return None, f"分子式非SMILES({smiles},不需要预测)"

        original_smiles = smiles
        fixes = fixes_pre

        # --- 检测多肽：三肽及以上清除 ---
        # 肽键特征：C(=O)-N，统计酰胺键数量
        peptide_bond_patterns = [
            r'C\(=O\)N',  # 标准肽键 C(=O)N
            r'C\(=O\)[^C]N',  # 带侧链的肽键
        ]
        peptide_bond_count = 0
        for pattern in peptide_bond_patterns:
            peptide_bond_count += len(re.findall(pattern, smiles))
        # 去重：连续鉴定可能重复计数，取保守估计
        peptide_bond_count = min(peptide_bond_count, smiles.count('N'))
        if peptide_bond_count >= 2:
            return None, f"三肽及以上({peptide_bond_count}个肽键),不支持"

        # --- 预处理：双斜杠、金属、分隔符、括号 ---
        smiles = re.sub(r'//C=C//', r'/C=C/', smiles)
        smiles = re.sub(r'/+C=C/+', r'/C=C/', smiles)

        metal_charges = {'[Fe]': '[Fe+3]', '[Fe+]': '[Fe+3]', '[Fe++]': '[Fe+3]'}
        for metal, charged in metal_charges.items():
            if metal in smiles:
                smiles = smiles.replace(metal, charged)
                fixes.append(f"金属电荷标准化:{metal}->{charged}")

        smiles = smiles.replace('=[O]', '=O').replace('[O]=', 'O=')
        smiles = re.sub(r'\|[^|]+\|', '', smiles)

        # 圆括号鉴定
        oc, cc = smiles.count('('), smiles.count(')')
        if oc > cc:
            smiles += ')' * (oc - cc)
            fixes.append(f"补全{oc - cc}个右括号")
        elif cc > oc:
            smiles = '(' * (cc - oc) + smiles
            fixes.append(f"补全{cc - oc}个左括号")

        # 方括号鉴定
        ob, cb = smiles.count('['), smiles.count(']')
        if ob > cb:
            smiles += ']' * (ob - cb)
            fixes.append(f"补全{ob - cb}个右方括号")
        elif cb > ob:
            smiles = '[' * (cb - ob) + smiles
            fixes.append(f"补全{cb - ob}个左方括号")

        # --- 1. 除盐处理 ---
        if '.' in smiles:
            try:
                components = smiles.split('.')
                organic_components = []
                for comp in components:
                    comp = comp.strip()
                    if not comp:
                        continue
                    mol = Chem.MolFromSmiles(comp, sanitize=False)
                    if mol:
                        try:
                            mol.UpdatePropertyCache(strict=False)
                        except Exception:
                            continue
                        atoms = set(a.GetSymbol() for a in mol.GetAtoms())
                        is_inorganic = len(atoms) <= 2 and atoms.issubset(
                            {'Na', 'K', 'Cl', 'Br', 'I', 'Ca', 'Mg', 'F', 'Li', 'O', 'H', 'S', 'P', 'N'})
                        if not is_inorganic or len(atoms) > 2:
                            try:
                                mw = Descriptors.MolWt(mol)
                            except Exception:
                                mw = 0
                            organic_components.append((comp, mw))
                if organic_components:
                    organic_components.sort(key=lambda x: x[1], reverse=True)
                    smiles = organic_components[0][0]
                    fixes.append(f"除盐:保留最大组分({organic_components[0][1]:.1f}Da)")
                else:
                    smiles = components[0].strip()
                    fixes.append("除盐:保留第一组分")
            except Exception as e:
                fixes.append(f"除盐异常:{e}")

        # --- 2. 电荷修复 ---
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None, f"SMILES解析失败; {'; '.join(fixes)}"

        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            return None, f"属性计算失败; {'; '.join(fixes)}"

        # 去除显式氢（FIORA/CFM-ID使用隐式氢表示）
        if any(a.GetSymbol() == 'H' and a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            try:
                mol = Chem.RemoveHs(mol)
                fixes.append("RDKit去除显式氢")
            except Exception:
                pass

        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            original_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())

            if original_charge != 0:
                try:
                    uncharger = rdMolStandardize.Uncharger()
                    mol = uncharger.uncharge(mol)
                    new_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
                    if original_charge != new_charge:
                        fixes.append(f"RDKit质子化(电荷{original_charge}→{new_charge})")
                except Exception as e:
                    fixes.append(f"RDKit Uncharger失败:{e}")

                current_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
                if current_charge != 0:
                    neutralized = False
                    editable_mol = Chem.RWMol(mol)
                    for atom in editable_mol.GetAtoms():
                        charge = atom.GetFormalCharge()
                        if charge == 0:
                            continue
                        symbol = atom.GetSymbol()
                        if charge < 0 and symbol in ['O', 'S', 'P', 'N', 'C']:
                            num_h = abs(charge)
                            atom.SetFormalCharge(0)
                            atom.SetNumExplicitHs(atom.GetTotalNumHs() + num_h)
                            neutralized = True
                        elif charge > 0:
                            if symbol == 'N' and atom.GetTotalDegree() >= 4:
                                fixes.append("季铵盐(永久正电荷)")
                            else:
                                cur_h = atom.GetTotalNumHs()
                                if cur_h >= charge:
                                    atom.SetFormalCharge(0)
                                    atom.SetNumExplicitHs(cur_h - charge)
                                    neutralized = True
                    if neutralized:
                        mol = editable_mol.GetMol()
                        fixes.append(f"手动质子化(电荷{current_charge}→{sum(a.GetFormalCharge() for a in mol.GetAtoms())})")
        except ImportError:
            fixes.append("RDKit Uncharger不可用")

        # --- 3. 验证 ---
        try:
            Chem.SanitizeMol(mol)
            final_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            final_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())

            # 过滤无键分子（单原子/无化学键，FIORA会ZeroDivisionError）
            num_bonds = mol.GetNumBonds()
            if num_bonds == 0:
                return None, f"无化学键(单原子分子,不需要预测)"

            # 过滤含金属原子的分子（FIORA/CFM-ID无法处理金属配位）
            METAL_SYMBOLS = {'Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Rb','Sr','Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi'}
            atom_symbols = set(a.GetSymbol() for a in mol.GetAtoms())
            if atom_symbols & METAL_SYMBOLS:
                metals_found = atom_symbols & METAL_SYMBOLS
                return None, f"含金属原子({','.join(sorted(metals_found))},不支持预测)"

            # 过滤含自由基的分子（FIORA/CFM-ID无法处理）
            if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
                return None, f"含自由基(不支持预测)"

            # 过滤分子量不在合理范围的化合物
            try:
                mol_weight = Descriptors.MolWt(mol)
                if mol_weight < 80:
                    return None, f"分子量过小({mol_weight:.1f}Da<80Da,不需要预测)"
                if mol_weight > 1500:
                    return None, f"分子量过大({mol_weight:.1f}Da>1500Da,高分辨质谱检测不了)"
            except Exception:
                pass  # 如果无法计算分子量，继续处理

            if final_charge != 0:
                charged = [(a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
                if any(s == 'N' and c > 0 for s, c in charged):
                    return None, "季铵盐(永久正电荷,不支持)"
                fixes.append(f"残留电荷({','.join(f'{s}{c:+d}' for s, c in charged[:3])})")
            return final_smiles, "; ".join(fixes) if fixes else "无需修复"
        except Exception as e:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
                final_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                fixes.append("部分sanitize成功")
                return final_smiles, "; ".join(fixes)
            except Exception:
                return None, f"sanitize失败:{e}; {'; '.join(fixes)}"

    except Exception as e:
        return None, f"修复异常:{e}"


def batch_fix_smiles(smiles_list: List[str], ion_mode: str = "[M+H]+", 
                     show_progress: bool = True) -> List[Dict]:
    """
    批量修复 SMILES 列表。

    参数:
        smiles_list: SMILES 字符串列表
        ion_mode: 离子模式
        show_progress: 是否显示进度条

    返回:
        修复结果列表，每个元素为字典:
        {
            'original': 原始SMILES,
            'fixed': 修复后的SMILES (失败则为None),
            'message': 修复说明,
            'success': 是否成功
        }
    """
    results = []
    iterator = tqdm(smiles_list, desc="SMILES修复", ncols=80) if show_progress else smiles_list
    
    for smi in iterator:
        fixed, msg = fix_smiles(smi, ion_mode)
        results.append({
            'original': smi,
            'fixed': fixed,
            'message': msg,
            'success': fixed is not None
        })
    
    return results


def calculate_precursor_mz(smiles: str, precursor_type: str) -> Optional[float]:
    """
    计算候选化合物的理论母离子 m/z。

    参数:
        smiles: SMILES 字符串
        precursor_type: 前体类型，如 '[M+H]+' 或 '[M-H]-'

    返回:
        理论母离子 m/z，失败返回 None
    """
    ADDUCT_MASS = {
        '[M+H]+': 1.007276,
        '[M-H]-': -1.007276,
    }
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        mass = Descriptors.ExactMolWt(mol)
        return mass + ADDUCT_MASS.get(precursor_type, 0)
    except Exception:
        return None


def generate_inchikey(smiles: str) -> str:
    """
    从 SMILES 生成 InChIKey。

    参数:
        smiles: SMILES 字符串

    返回:
        InChIKey 字符串，失败返回 ""
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey
        mol = Chem.MolFromSmiles(smiles)
        return MolToInchiKey(mol) if mol else ""
    except Exception:
        return ""


def generate_formula(smiles: str) -> str:
    """
    从 SMILES 生成分子式。

    参数:
        smiles: SMILES 字符串

    返回:
        分子式字符串，失败返回 ""
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        return rdMolDescriptors.CalcMolFormula(mol) if mol else ""
    except Exception:
        return ""


# 模块测试
if __name__ == "__main__":
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # 阿司匹林
        "CC(=O)[O-].[Na+]",  # 盐类
        "C[N+](C)(C)C",  # 季铵盐
        "[Fe]C1=CC=CC=C1",  # 金属
    ]
    
    print("=" * 60)
    print("SMILES 修复测试")
    print("=" * 60)
    
    for smi in test_smiles:
        fixed, msg = fix_smiles(smi)
        print(f"\n原始: {smi}")
        print(f"修复: {fixed}")
        print(f"说明: {msg}")


def dedup_smiles(smiles_list: List[str], ion_mode: str = "[M+H]+") -> Tuple[List[str], Dict]:
    """
    SMILES去重函数（基于原始SMILES，保留所有位置异构体）
    
    去重逻辑：
    - 直接比较原始SMILES字符串
    - 使用fix_smiles验证SMILES有效性
    
    参数:
        smiles_list: SMILES字符串列表
        ion_mode: 离子模式（用于fix_smiles验证）
    
    返回:
        (唯一SMILES列表, 统计信息)
        统计信息包含: total, unique, duplicate
    """
    seen = set()  # 原始SMILES集合
    unique_smiles = []
    stats = {'total': len(smiles_list), 'unique': 0, 'duplicate': 0}
    
    for smi in smiles_list:
        # 先用fix_smiles验证SMILES有效性
        fixed, msg = fix_smiles(smi, ion_mode)
        if fixed is None:
            continue
        
        # 直接用原始SMILES去重（保留所有位置异构体）
        if smi in seen:
            stats['duplicate'] += 1
        else:
            seen.add(smi)
            unique_smiles.append(smi)
            stats['unique'] += 1
    
    return unique_smiles, stats


def batch_dedup_smiles(smiles_list: List[str], ion_mode: str = "[M+H]+", 
                        show_progress: bool = False) -> Tuple[List[str], Dict]:
    """
    批量SMILES去重（基于原始SMILES，保留所有位置异构体）
    
    去重逻辑：
    - 直接比较原始SMILES字符串
    - 使用fix_smiles验证SMILES有效性
    
    参数:
        smiles_list: SMILES字符串列表
        ion_mode: 离子模式（用于fix_smiles验证）
        show_progress: 是否显示进度条
    
    返回:
        (唯一SMILES列表, 统计信息)
    """
    seen = set()  # 原始SMILES集合
    unique_smiles = []
    stats = {'total': len(smiles_list), 'unique': 0, 'duplicate': 0}
    
    iterator = tqdm(smiles_list, desc="去重") if show_progress else smiles_list
    
    for smi in iterator:
        # 先用fix_smiles验证SMILES有效性
        fixed, msg = fix_smiles(smi, ion_mode)
        if fixed is None:
            continue
        
        # 直接用原始SMILES去重（保留所有位置异构体）
        if smi in seen:
            stats['duplicate'] += 1
        else:
            seen.add(smi)
            unique_smiles.append(smi)
            stats['unique'] += 1
    
    return unique_smiles, stats


def preprocess_smiles(smiles_list: List[str], ion_mode: str = "[M+H]+", 
                       show_progress: bool = False) -> Tuple[List[Dict], Dict, Dict]:
    """
    SMILES预处理函数（去重 + 修复 + 元数据生成）
    
    参数:
        smiles_list: SMILES字符串列表
        ion_mode: 离子模式
        show_progress: 是否显示进度条
    
    返回:
        (预处理后的候选列表, 统计信息, SMILES索引映射)
        候选列表包含: smiles, fixed_smiles, precursor_mz, inchikey, formula
        索引映射: {smiles: [preprocessed_indices]} 用于元数据关联
    """
    # 第一步：去重
    unique_smiles, dedup_stats = dedup_smiles(smiles_list, ion_mode)
    
    # 第二步：批量修复
    fix_results = batch_fix_smiles(unique_smiles, ion_mode, show_progress)
    
    # 第三步：构建候选列表 + 索引映射
    candidates = []
    smiles_to_index = {}  # {smiles: [index_in_candidates]}
    for smi, result in zip(unique_smiles, fix_results):
        if not result['success']:
            continue
        
        fixed_smi = result['fixed']
        precursor_mz = calculate_precursor_mz(fixed_smi, ion_mode)
        if not precursor_mz:
            continue
        
        inchikey = generate_inchikey(fixed_smi)
        if not inchikey:
            continue
        
        formula = generate_formula(fixed_smi)
        
        smiles_to_index[smi] = len(candidates)  # 记录SMILES在candidates中的索引
        candidates.append({
            'smiles': smi,
            'fixed_smiles': fixed_smi,
            'precursor_mz': precursor_mz,
            'inchikey': inchikey,
            'formula': formula,
        })
    
    return candidates, dedup_stats, smiles_to_index


def preprocess_smiles_with_metadata(smiles_list: List[str], ion_mode: str = "[M+H]+", 
                                     show_progress: bool = False) -> Tuple[List[Dict], Dict, Dict]:
    """
    SMILES预处理函数（保留所有来源元数据）
    
    与preprocess_smiles的区别：
    - 相同SMILES会生成多条记录（保留所有来源信息）
    - 返回的candidates中每个条目包含 all_names 和 all_sources 字段
    
    参数:
        smiles_list: SMILES字符串列表
        ion_mode: 离子模式
        show_progress: 是否显示进度条
    
    返回:
        (预处理后的候选列表, 统计信息, SMILES索引映射)
        候选列表包含: smiles, fixed_smiles, precursor_mz, inchikey, formula, all_names, all_sources
    """
    # 第一步：收集所有SMILES及其出现次数（用于统计）
    all_smiles = []
    smiles_to_indices = {}  # {smiles: [original_indices]}
    for idx, smi in enumerate(smiles_list):
        all_smiles.append(smi)
        if smi not in smiles_to_indices:
            smiles_to_indices[smi] = []
        smiles_to_indices[smi].append(idx)
    
    # 第二步：对唯一SMILES进行预处理
    unique_smiles = list(dict.fromkeys(smiles_list))  # 保持顺序的去重
    dedup_stats = {'total': len(smiles_list), 'unique': len(unique_smiles), 'duplicate': len(smiles_list) - len(unique_smiles)}
    
    fix_results = batch_fix_smiles(unique_smiles, ion_mode, show_progress)
    
    # 第三步：构建候选列表
    candidates = []
    smiles_to_index = {}  # {smiles: first_valid_index}
    for smi, result in zip(unique_smiles, fix_results):
        if not result['success']:
            continue
        
        fixed_smi = result['fixed']
        precursor_mz = calculate_precursor_mz(fixed_smi, ion_mode)
        if not precursor_mz:
            continue
        
        inchikey = generate_inchikey(fixed_smi)
        if not inchikey:
            continue
        
        formula = generate_formula(fixed_smi)
        
        # 统计该SMILES出现的次数（来源数量）
        source_count = len(smiles_to_indices[smi])
        
        smiles_to_index[smi] = len(candidates)
        candidates.append({
            'smiles': smi,
            'fixed_smiles': fixed_smi,
            'precursor_mz': precursor_mz,
            'inchikey': inchikey,
            'formula': formula,
            'source_count': source_count,  # 该SMILES出现的次数
        })
    
    return candidates, dedup_stats, smiles_to_index
