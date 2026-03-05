#!/usr/bin/env python3
"""
SMILES 修复工具（辅助功能模块）

修复策略：
1. 双斜杠修复、金属电荷标准化、分隔符移除、括号匹配
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

        original_smiles = smiles
        fixes = []

        # --- 检测多肽：三肽及以上清除 ---
        # 肽键特征：C(=O)-N，统计酰胺键数量
        peptide_bond_patterns = [
            r'C\(=O\)N',  # 标准肽键 C(=O)N
            r'C\(=O\)[^C]N',  # 带侧链的肽键
        ]
        peptide_bond_count = 0
        for pattern in peptide_bond_patterns:
            peptide_bond_count += len(re.findall(pattern, smiles))
        # 去重：连续匹配可能重复计数，取保守估计
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

        # 圆括号匹配
        oc, cc = smiles.count('('), smiles.count(')')
        if oc > cc:
            smiles += ')' * (oc - cc)
            fixes.append(f"补全{oc - cc}个右括号")
        elif cc > oc:
            smiles = '(' * (cc - oc) + smiles
            fixes.append(f"补全{cc - oc}个左括号")

        # 方括号匹配
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
            return original_smiles, f"SMILES解析失败; {'; '.join(fixes)}"

        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            return original_smiles, f"属性计算失败; {'; '.join(fixes)}"

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
                return original_smiles, f"sanitize失败:{e}; {'; '.join(fixes)}"

    except Exception as e:
        return smiles, f"修复异常:{e}"


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
