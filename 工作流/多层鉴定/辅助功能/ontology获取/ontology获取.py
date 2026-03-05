#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClassyFire Ontology分类获取脚本
用于获取化合物的分类信息（如黄酮类、甾体类等）
"""

import requests
import json
import time
import logging
from typing import Dict, Optional, List
import argparse
import pandas as pd
import os
import sys
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ontology_fetch.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ClassyFire API配置
CLASSYFIRE_API_URL = "http://classyfire.wishartlab.com"
CLASSYFIRE_TIMEOUT = 30
CLASSYFIRE_DELAY = 1.5  # 基础请求间隔（秒）
CLASSYFIRE_MAX_RETRIES = 5
CLASSYFIRE_CACHE_SAVE_INTERVAL = 50  # 每N次API请求保存一次缓存

# 缓存文件（放在脚本所在目录）
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontology_classyfire_cache.json")

class ClassyFireClient:
    """ClassyFire API客户端（自适应限速 + 去重批量查询）"""
    
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.last_was_cache_hit = False
        self.current_delay = CLASSYFIRE_DELAY  # 自适应延迟（遇429会自动增大）
        self._unsaved_count = 0  # 未保存的新缓存条目数
    
    def _is_valid_inchikey(self, inchikey: str) -> bool:
        """
        验证InChIKey格式是否有效
        有效格式：14个大写字母 + "-" + 10个大写字母 + "-" + 大写字母
        例如：KJRQQECDVUXBCO-UHFFFAOYSA-N
        """
        if not inchikey or not isinstance(inchikey, str):
            return False
        import re
        pattern = r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$'
        return bool(re.match(pattern, inchikey.strip()))
        
    def _load_cache(self) -> Dict:
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                return {}
        return {}
    
    def _save_cache(self, force=False):
        """保存缓存（批量保存，减少磁盘IO）"""
        self._unsaved_count += 1
        if not force and self._unsaved_count < CLASSYFIRE_CACHE_SAVE_INTERVAL:
            return
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            self._unsaved_count = 0
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def get_classification(self, inchi_key: str, classification_field: str = "class") -> Optional[str]:
        """
        获取化合物分类
        
        Args:
            inchi_key: InChI Key
            classification_field: 分类层级 ("kingdom", "superclass", "class", "subclass", "direct_parent")
            
        Returns:
            分类名称或None
        """
        if not inchi_key or inchi_key == "":
            self.last_was_cache_hit = True  # 无效输入视为已处理，不触发延迟
            return None
        
        # 验证InChIKey格式（14字母-10字母-N）
        inchi_key = str(inchi_key).strip()
        if not self._is_valid_inchikey(inchi_key):
            self.last_was_cache_hit = True  # 无效格式视为已处理，不触发延迟
            return None
            
        # 检查缓存
        cache_key = f"{inchi_key}_{classification_field}"
        if cache_key in self.cache:
            self.last_was_cache_hit = True
            return self.cache[cache_key]
        
        self.last_was_cache_hit = False
        
        # API请求
        url = f"{CLASSYFIRE_API_URL}/entities/{inchi_key}.json"
        
        for attempt in range(CLASSYFIRE_MAX_RETRIES):
            try:
                response = requests.get(
                    url, 
                    timeout=CLASSYFIRE_TIMEOUT,
                    headers={'Accept': 'application/json'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 提取指定层级的分类
                    if classification_field in data and data[classification_field]:
                        classification = data[classification_field]['name']
                        # 缓存结果
                        self.cache[cache_key] = classification
                        self._save_cache()
                        return classification
                    else:
                        # 该化合物无此层级分类
                        self.cache[cache_key] = None
                        self._save_cache()
                        return None
                        
                elif response.status_code == 404:
                    # 化合物未找到
                    self.cache[cache_key] = None
                    self._save_cache()
                    return None
                    
                elif response.status_code == 429:
                    # 请求过于频繁 —— 永久提高基础延迟 + 指数退避等待
                    self.current_delay = min(self.current_delay * 1.5, 15.0)
                    wait_time = self.current_delay * (2 ** attempt)
                    logger.warning(f"请求过于频繁 (429)，基础延迟提升至 {self.current_delay:.1f}s，本次等待 {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code >= 500:
                    # 服务器错误，指数退避重试
                    wait_time = CLASSYFIRE_DELAY * (2 ** attempt) * 3  # 500错误等待更久
                    logger.warning(f"服务器错误 ({response.status_code})，等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.warning(f"API请求失败 (状态码: {response.status_code})")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常 (尝试 {attempt + 1}/{CLASSYFIRE_MAX_RETRIES}): {e}")
                
            if attempt < CLASSYFIRE_MAX_RETRIES - 1:
                time.sleep(CLASSYFIRE_DELAY * (2 ** attempt))  # 指数退避
        
        return None

    def flush_cache(self):
        """强制保存缓存（处理结束时调用）"""
        self._save_cache(force=True)

    def batch_query(self, inchikeys: List[str], classification_field: str = "class") -> Dict[str, Optional[str]]:
        """
        批量查询唯一InChIKey的分类（去重后逐个查询，自适应限速）
        
        Returns:
            {inchikey: classification_or_None}
        """
        # 去重
        unique_keys = list(set(k for k in inchikeys if k and isinstance(k, str) 
                              and self._is_valid_inchikey(k.strip())))
        
        # 分离缓存命中与需查询的
        results = {}
        to_query = []
        for key in unique_keys:
            cache_key = f"{key.strip()}_{classification_field}"
            if cache_key in self.cache:
                results[key.strip()] = self.cache[cache_key]
            else:
                to_query.append(key.strip())
        
        logger.info(f"唯一InChIKey: {len(unique_keys)} 个，缓存命中: {len(results)} 个，需查询API: {len(to_query)} 个")
        
        # 逐个查询，带自适应限速
        for i, key in enumerate(tqdm(to_query, desc="[Ontology API查询]", unit="个", ncols=80)):
            classification = self.get_classification(key, classification_field)
            results[key] = classification
            
            # 仅在实际发起API请求时延迟
            if not self.last_was_cache_hit and i < len(to_query) - 1:
                time.sleep(self.current_delay)
        
        self.flush_cache()
        return results

def process_csv(input_csv: str, output_csv: str, classification_field: str = "class"):
    """
    处理CSV文件，为每个化合物获取ontology分类
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        classification_field: 分类层级
    """
    client = ClassyFireClient()
    
    # 检查文件是否存在且非空
    if not os.path.exists(input_csv):
        logger.warning(f"输入文件不存在: {input_csv}")
        return
    
    if os.path.getsize(input_csv) == 0:
        logger.warning(f"输入文件为空: {input_csv}")
        return
    
    # 读取输入文件
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except pd.errors.EmptyDataError:
        logger.warning(f"输入文件格式错误(空文件): {input_csv}")
        return
    
    if df.empty:
        logger.warning(f"输入文件无数据: {input_csv}")
        return
    
    logger.info(f"读取 {len(df)} 条记录")
    
    # 确定 InChIKey 列名
    inchikey_col = None
    for col in ['matched_inchikey', 'target_inchikey', 'inchikey', 'InChIKey', 'INCHIKEY']:
        if col in df.columns:
            inchikey_col = col
            break
    
    if not inchikey_col:
        logger.warning("未找到 InChIKey 列，无法获取 ontology")
        df.to_csv(output_csv, index=False, encoding='utf-8')
        return
    
    logger.info(f"使用 InChIKey 列: {inchikey_col}")
    
    # 填充 matched_ontology 列
    if 'matched_ontology' not in df.columns:
        df['matched_ontology'] = ""
    
    # 1. 统计已有ontology的行（跳过）
    has_ontology = df['matched_ontology'].apply(
        lambda x: bool(x) and str(x).strip() != '' and str(x) != 'nan'
    )
    skipped_count = has_ontology.sum()
    
    # 2. 收集需要查询的行的InChIKey（去重后批量查询）
    needs_query = ~has_ontology
    query_inchikeys = df.loc[needs_query, inchikey_col].dropna()
    query_inchikeys = query_inchikeys[query_inchikeys.astype(str).str.strip() != '']
    query_inchikeys = query_inchikeys[query_inchikeys.astype(str) != 'nan']
    
    unique_keys = list(set(str(k).strip() for k in query_inchikeys.values))
    logger.info(f"总记录: {len(df)}, 已有ontology跳过: {skipped_count}, 需查询行: {needs_query.sum()}, 唯一InChIKey: {len(unique_keys)}")
    
    # 3. 批量查询（去重 + 自适应限速）
    results_map = client.batch_query(unique_keys, classification_field)
    
    # 4. 将结果广播回DataFrame
    success_count = 0
    fail_count = 0
    for idx in df.index[needs_query]:
        inchi_key = df.at[idx, inchikey_col]
        if pd.isna(inchi_key) or str(inchi_key).strip() == '' or str(inchi_key) == 'nan':
            continue
        classification = results_map.get(str(inchi_key).strip())
        if classification:
            df.at[idx, 'matched_ontology'] = classification
            success_count += 1
        else:
            fail_count += 1
    
    # 5. 保存结果
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"处理完成:")
    logger.info(f"  已有ontology跳过: {skipped_count}")
    logger.info(f"  新获取分类: {success_count}")
    logger.info(f"  未能获取分类: {fail_count}")
    logger.info(f"  结果保存至: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="ClassyFire Ontology分类获取")
    parser.add_argument("--input_csv", required=True, help="输入CSV文件路径")
    parser.add_argument("--output_csv", required=True, help="输出CSV文件路径")
    parser.add_argument("--field", default="class", 
                       choices=["kingdom", "superclass", "class", "subclass", "direct_parent"],
                       help="分类层级 (默认: class)")
    
    args = parser.parse_args()
    
    process_csv(args.input_csv, args.output_csv, args.field)

if __name__ == "__main__":
    main()