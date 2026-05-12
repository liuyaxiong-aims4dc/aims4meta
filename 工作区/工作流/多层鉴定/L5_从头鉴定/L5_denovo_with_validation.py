#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L5从头鉴定 + 理论谱图验证
流程：
1. MSFlow从头鉴定 → 生成SMILES候选
2. CFM-ID/FIORA → 预测理论谱图
3. MatchMS/DreaMS → 计算相似度
4. 过滤高置信度结果 (相似度 ≥ 阈值)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
from rdkit import Chem
from rdkit import rdBase
import subprocess
import tempfile
import json

warnings.filterwarnings('ignore')
blocker = rdBase.BlockLogs()

# 添加MSFlow路径
MSFLOW_DIR = '/stor3/AIMS4Meta/源代码/MSFlow-main'
sys.path.insert(0, MSFLOW_DIR)
sys.path.insert(0, os.path.join(MSFLOW_DIR, 'ms_scripts/DiffMS/src'))

from modules.cond_lit_model import CondFlowMolBERTLitModule
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from utils.sample import cond_generate_mols
from utils.metrics import decode_tokens_to_smiles
from utils.functions import canonicalize
from configs import ID2TOK, TOK2ID, PAD
from DiffMS.src.mist.models.spectra_encoder import SpectraEncoderGrowing

torch.serialization.add_safe_globals([
    MixtureDiscreteProbPath,
    PolynomialConvexScheduler,
    torch.nn.modules.loss.CrossEntropyLoss
])


class MSFlowPredictor:
    """MSFlow从头鉴定预测器"""

    def __init__(self, encoder_path, decoder_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载编码器 (质谱 → CDDD)
        print("加载编码器...")
        checkpoint = torch.load(encoder_path, map_location=self.device)
        self.encoder = SpectraEncoderGrowing(
            inten_transform='float',
            inten_prob=0.1,
            remove_prob=0.5,
            peak_attn_layers=2,
            num_heads=8,
            pairwise_featurization=True,
            embed_instrument=False,
            cls_type='ms1',
            set_pooling='cls',
            spec_features='peakformula',
            mol_features='fingerprint',
            form_embedder='pos-cos',
            output_size=512,
            hidden_size=512,
            spectra_dropout=0.0,
            top_layers=1,
            refine_layers=4,
            magma_modulo=2048,
        )
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self._replace_sigmoid_with_tanh(self.encoder)
        self.encoder.to(self.device)
        self.encoder.eval()

        # 加载解码器 (CDDD → SMILES)
        print("加载解码器...")
        cfm_module = CondFlowMolBERTLitModule.load_from_checkpoint(decoder_path)
        self.decoder = cfm_module.model
        self.decoder.eval()
        self.decoder.to(self.device)

        print("MSFlow模型加载完成")

    def _replace_sigmoid_with_tanh(self, module):
        """替换Sigmoid为Tanh"""
        for name, child in module.named_children():
            if isinstance(child, nn.Sigmoid):
                setattr(module, name, nn.Tanh())
            else:
                self._replace_sigmoid_with_tanh(child)

    def predict_from_msp(self, msp_file, num_candidates=10, output_csv=None):
        """
        从MSP文件预测SMILES

        Args:
            msp_file: MSP文件路径
            num_candidates: 每个谱图生成的候选数
            output_csv: 输出CSV路径

        Returns:
            DataFrame: 包含query_name, rank, smiles, score列
        """
        print(f"\n读取MSP文件: {msp_file}")
        spectra = self._parse_msp(msp_file)
        print(f"共读取 {len(spectra)} 个谱图")

        results = []

        for spec in tqdm(spectra, desc="从头鉴定"):
            query_name = spec['name']
            precursor_mz = spec['precursor_mz']
            peaks = spec['peaks']

            # 编码质谱 → CDDD
            try:
                cddd = self._encode_spectrum(precursor_mz, peaks)

                # 解码CDDD → SMILES
                smiles_list = self._decode_cddd(cddd, num_samples=num_candidates)

                # 保存结果
                for rank, smi in enumerate(smiles_list, 1):
                    results.append({
                        'query_name': query_name,
                        'rank': rank,
                        'smiles': smi,
                        'score': 1.0 / rank  # 简单的排名分数
                    })
            except Exception as e:
                print(f"  警告: {query_name} 预测失败: {e}")
                continue

        df = pd.DataFrame(results)

        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\n结果已保存: {output_csv}")

        return df

    def _parse_msp(self, msp_file):
        """解析MSP文件"""
        spectra = []
        current_spec = {}
        peaks = []

        with open(msp_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if ':' in line and not line[0].isdigit():
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()

                    if key == 'NAME':
                        if current_spec and peaks:
                            current_spec['peaks'] = np.array(peaks)
                            spectra.append(current_spec)
                        current_spec = {'name': value}
                        peaks = []
                    elif key == 'PRECURSORMZ':
                        current_spec['precursor_mz'] = float(value)
                    elif key == 'NUM PEAKS':
                        current_spec['num_peaks'] = int(value)
                else:
                    # 峰数据
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            peaks.append([mz, intensity])
                        except:
                            pass

        # 最后一个谱图
        if current_spec and peaks:
            current_spec['peaks'] = np.array(peaks)
            spectra.append(current_spec)

        return spectra

    def _encode_spectrum(self, precursor_mz, peaks):
        """编码质谱为CDDD向量"""
        # TODO: 需要构建DiffMS数据格式
        # 这里需要根据DiffMS的数据格式构建输入
        # 暂时返回随机向量作为占位符
        return torch.randn(512, device=self.device)

    def _decode_cddd(self, cddd, num_samples=10):
        """解码CDDD为SMILES"""
        with torch.no_grad():
            samples = cond_generate_mols(
                self.decoder,
                cond=cddd,
                source_distribution='uniform',
                num_samples=num_samples,
                steps=128,
                guidance_scale=1.5,
                temperature=1,
                device=self.device,
            )

        _, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
        smiles = [canonicalize(s) for s in smiles if s]
        smiles = [s for s in smiles if s is not None]

        # 去重
        unique_smiles = []
        seen = set()
        for smi in smiles:
            if smi not in seen:
                unique_smiles.append(smi)
                seen.add(smi)

        return unique_smiles


def validate_with_cfmid(smiles_df, query_msp, ion_mode, similarity_threshold=0.7):
    """
    使用CFM-ID预测理论谱图并验证

    Args:
        smiles_df: L5预测的SMILES DataFrame
        query_msp: 查询谱图MSP文件
        ion_mode: 离子模式 (POS/NEG)
        similarity_threshold: 相似度阈值

    Returns:
        DataFrame: 验证后的结果
    """
    print("\n=== 步骤2: CFM-ID理论谱图预测 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 生成候选SMILES文件
        candidates_csv = os.path.join(tmpdir, 'candidates.csv')
        smiles_df[['smiles']].drop_duplicates().to_csv(candidates_csv, index=False, header=False)

        # 2. 调用L2_CFMID预测脚本
        cfmid_script = '/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L2_模拟数据库鉴定/L2_CFMID预测.py'
        cfmid_output = os.path.join(tmpdir, 'cfmid_library.msp')

        cmd = [
            'python', cfmid_script,
            '--input_csv', candidates_csv,
            '--output_msp', cfmid_output,
            '--ion_mode', ion_mode,
            '--collision_energy', '20'
        ]

        print(f"运行CFM-ID预测...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"CFM-ID预测失败: {result.stderr}")
            return pd.DataFrame()

        # 3. 调用MatchMS比对
        print("\n=== 步骤3: MatchMS相似度计算 ===")
        matchms_script = '/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L1_实验数据库鉴定/L1_matchMS鉴定.py'
        matchms_output = os.path.join(tmpdir, 'matchms_results.csv')

        cmd = [
            'python', matchms_script,
            '--query_msp', query_msp,
            '--library_msp', cfmid_output,
            '--output_csv', matchms_output,
            '--min_matched_peaks', '6',
            '--tolerance', '0.02'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"MatchMS比对失败: {result.stderr}")
            return pd.DataFrame()

        # 4. 读取结果并过滤
        if not os.path.exists(matchms_output):
            print("未生成MatchMS结果")
            return pd.DataFrame()

        matchms_df = pd.read_csv(matchms_output)

        # 过滤高相似度结果
        validated = matchms_df[matchms_df['CosineGreedy'] >= similarity_threshold].copy()

        print(f"\n验证结果:")
        print(f"  总候选数: {len(smiles_df)}")
        print(f"  相似度 ≥ {similarity_threshold}: {len(validated)}")

        return validated


def main():
    parser = argparse.ArgumentParser(description='L5从头鉴定 + 理论谱图验证')
    parser.add_argument('--input_msp', required=True, help='输入MSP文件 (L4未鉴定)')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--ion_mode', required=True, choices=['POS', 'NEG'], help='离子模式')
    parser.add_argument('--num_candidates', type=int, default=10, help='每个谱图生成的候选数')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='相似度阈值')
    parser.add_argument('--encoder_path', default='/stor3/AIMS4Meta/源代码/MSFlow-main/checkpoints/Encoder/encoder_msg_cddd.pt')
    parser.add_argument('--decoder_path', default='/stor3/AIMS4Meta/源代码/MSFlow-main/checkpoints/Decoder/MSFlow_cddds.ckpt')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("L5从头鉴定 + 理论谱图验证")
    print("=" * 80)
    print(f"输入MSP: {args.input_msp}")
    print(f"输出目录: {args.output_dir}")
    print(f"离子模式: {args.ion_mode}")
    print(f"候选数: {args.num_candidates}")
    print(f"相似度阈值: {args.similarity_threshold}")

    # 步骤1: MSFlow从头鉴定
    print("\n=== 步骤1: MSFlow从头鉴定 ===")
    predictor = MSFlowPredictor(args.encoder_path, args.decoder_path)

    l5_output = os.path.join(args.output_dir, 'L5_denovo_candidates.csv')
    smiles_df = predictor.predict_from_msp(
        args.input_msp,
        num_candidates=args.num_candidates,
        output_csv=l5_output
    )

    if smiles_df.empty:
        print("从头鉴定未生成任何候选")
        return

    # 步骤2-3: CFM-ID预测 + MatchMS验证
    validated_df = validate_with_cfmid(
        smiles_df,
        args.input_msp,
        args.ion_mode,
        args.similarity_threshold
    )

    # 保存最终结果
    if not validated_df.empty:
        final_output = os.path.join(args.output_dir, 'L5_validated_results.csv')
        validated_df.to_csv(final_output, index=False)
        print(f"\n最终结果已保存: {final_output}")

        # 统计
        print("\n=== 最终统计 ===")
        print(f"验证通过: {len(validated_df)} 个化合物")
        print(f"平均相似度: {validated_df['CosineGreedy'].mean():.3f}")
        print(f"最高相似度: {validated_df['CosineGreedy'].max():.3f}")
    else:
        print("\n未找到验证通过的结果")


if __name__ == '__main__':
    main()
