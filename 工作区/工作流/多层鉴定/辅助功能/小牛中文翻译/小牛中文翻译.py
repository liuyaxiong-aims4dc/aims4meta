#!/usr/bin/env python3
"""
化合物名翻译脚本 - 用于多层鉴定结果翻译

功能：
1. 读取各层级鉴定结果CSV文件
2. 自动识别需要翻译的化合物名（排除已有中文的）
3. 调用小牛翻译API进行翻译
4. 输出带中文名称的结果文件

用法：
    python3 translate_results.py <results_dir> [--cache <cache_file>]
    
示例：
    python3 translate_results.py /path/to/L1_results
    python3 translate_results.py /path/to/L2_results
    python3 translate_results.py /path/to/L4_results
"""

import os
import sys
import re
import json
import time
import requests
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 缓存文件路径（相对于脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(SCRIPT_DIR, "小牛translation_cache.json")

# 配置
NIU_API_KEY = "015c3c382ecdec69f604f23c37527292"
NIU_API_URL = "https://api.niutrans.com/NiuTransServer/translation"

translation_cache = {}
failed_translations = {}
cache_lock = threading.Lock()  # 缓存锁


def call_niu_api_batch(texts, max_retries=3):
    """
    逐条调用小牛翻译 API（POST，domain=chemistry），替代旧的 GET 批量接口。

    旧的 GET 批量接口用 \\n 拼接/拆分文本，当化合物名超长或含特殊字符时
    会产生输出错位（A 的翻译被记到 B 名下），导致缓存大面积污染。
    逐条 POST 保证输入输出严格一一对应，外层 ThreadPoolExecutor 提供并发。
    """
    if not texts:
        return []

    results = []
    for text in texts:
        for retry in range(max_retries):
            try:
                translation = call_niu_api(text)
                results.append(translation)
                break
            except Exception:
                if retry < max_retries - 1:
                    time.sleep(1)
                else:
                    results.append(text)  # 失败返回原文
        # 逐条间短暂休眠，避免限流
        time.sleep(0.05)
    return results


def translate_batch_threaded(items, cache_file, max_workers=5):
    """
    多线程批量翻译

    参数:
        items: 待翻译项列表 [(idx, text), ...]
        cache_file: 缓存文件路径
        max_workers: 最大线程数

    返回:
        {idx: translation} 字典
    """
    results = {}
    items_to_translate = []  # 需要API翻译的项
    cached_results = {}  # 缓存命中的结果

    # 第一步: 检查缓存
    for idx, text in items:
        if not text or not isinstance(text, str):
            results[idx] = text
            continue

        text = text.strip()
        if not text or is_chinese(text):
            results[idx] = text
            continue

        # 检查缓存
        with cache_lock:
            if text in translation_cache:
                cached = translation_cache[text]
                if cached and is_chinese(cached):
                    cached_results[idx] = cached
                    results[idx] = cached
                    continue

        # 需要API翻译
        items_to_translate.append((idx, text))

    if not items_to_translate:
        return results

    # 第二步: 批量API翻译
    texts_to_translate = [text for idx, text in items_to_translate]
    translations = call_niu_api_batch(texts_to_translate)

    # 第三步: 更新缓存和结果
    for (idx, text), translation in zip(items_to_translate, translations):
        if translation and is_chinese(translation) and translation != text:
            with cache_lock:
                translation_cache[text] = translation
            results[idx] = translation
        else:
            # 翻译失败,缓存原文
            with cache_lock:
                translation_cache[text] = text
            results[idx] = text

    # 保存缓存
    with cache_lock:
        save_cache(cache_file)

    return results


def load_cache(cache_file):
    """加载翻译缓存"""
    global translation_cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                translation_cache = json.load(f)
            print(f"已加载缓存: {len(translation_cache)} 条记录")
        except Exception as e:
            print(f"加载缓存失败: {e}")
            translation_cache = {}


def save_cache(cache_file):
    """保存翻译缓存"""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(translation_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存缓存失败: {e}")


def is_chinese(text):
    """判断文本是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def has_chinese_translation(text):
    """判断文本是否已有中文翻译（原文是中文 或 已有 'English (中文)' 格式）

    注意：返回 False 不代表没有中文——可能是混合文本（英文前缀+中文后缀），
    此时虽然整体含中文，但英文前缀仍需翻译。调用方需进一步用 is_partial_translation() 判断。
    """
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    # 情况1: 原文本身包含中文 → 可能是 "纯中文" 或 "混合中英文"
    if is_chinese(text):
        # 如果英文占比大但含中文，不算"已有翻译"，需要进一步处理
        if is_partial_translation(text):
            return False
        return True
    # 情况2: 已有翻译格式 "English (中文)"
    # 检查是否以中文括号结尾
    match = re.search(r'\([^)]*[\u4e00-\u9fff][^)]*\)\s*$', text)
    if match:
        # 确保括号外没有未翻译的英文（排除纯数字/符号）
        prefix = text[:match.start()].strip()
        if not re.search(r'[a-zA-Z]{3,}', prefix):
            return True
    return False


def is_partial_translation(text):
    """检测是否为部分翻译：英文前缀 + 中文括号后缀

    例如: kaempferol-3-O-... ((2S)-1-[...]哌嗪-2-甲酰胺)
    英文主体未译，但后缀括号内已是中文。
    """
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if not is_chinese(text):
        return False
    # 必须有显著的英文部分（连续3个以上字母，排除H/C/N/O/P等单原子字符）
    if not re.search(r'[a-zA-Z]{3,}', text):
        return False
    return True


def split_partial_translation(text):
    """拆分混合文本：返回 (english_prefix, chinese_suffix)

    规则：
    1. 找到最后一个包含中文的括号组作为中文后缀的起点
    2. 之前的部分判定为英文前缀
    3. 如果英文前缀太短或无实质内容，返回 (None, None)
    """
    if not text:
        return None, None
    text = text.strip()
    # 找最后一个包含中文的括号组
    matches = list(re.finditer(r'\([^()]*[\u4e00-\u9fff][^()]*\)', text))
    if not matches:
        # 没有括号包裹的中文，尝试按第一个中文字符切分
        m = re.search(r'[\u4e00-\u9fff]', text)
        if m:
            prefix = text[:m.start()].strip()
            suffix = text[m.start():].strip()
            if prefix and re.search(r'[a-zA-Z]{3,}', prefix):
                return prefix, suffix
        return None, None

    last_match = matches[-1]
    # 检查该括号组前是否有英文（可能含括号的化学名）
    prefix = text[:last_match.start()].strip()
    suffix = text[last_match.start():].strip()
    if not prefix:
        return None, None
    if not re.search(r'[a-zA-Z]{3,}', prefix):
        return None, None
    # 检查前缀末尾是否是中文括号的配对前括号
    # 例如: kaempferol-3-O-((3'',4''-...)) (中文)
    # 需要把成对的英文括号也归入前缀
    return prefix, suffix


def call_niu_api(text, from_lang="en", to_lang="zh"):
    """调用小牛翻译API"""
    url = NIU_API_URL
    headers = {"Content-Type": "application/json"}
    data = {
        "from": from_lang,
        "to": to_lang,
        "apikey": NIU_API_KEY,
        "src_text": text,
        "domain": "chemistry"
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    result = response.json()
    
    if "tgt_text" in result:
        return result["tgt_text"]
    else:
        raise Exception(f"翻译失败: {result.get('error_msg', result)}")


def translate_text(text, cache_file):
    """翻译单个文本（优先使用缓存，API失败时返回原文）"""
    global translation_cache
    
    if not text or not isinstance(text, str):
        return text
    
    text = text.strip()
    if not text:
        return text
    
    # 检查是否包含中文（已有中文不需要翻译）
    if is_chinese(text):
        return text
    
    # 检查缓存
    if text in translation_cache:
        cached = translation_cache[text]
        # 如果缓存内容是中文，返回翻译结果
        if cached and is_chinese(cached):
            return cached
        # 如果缓存的是原文，说明之前翻译失败，直接返回原文
        return text
    
    # 尝试调用API
    try:
        translation = call_niu_api(text)
        # 校验翻译结果
        if translation and is_chinese(translation) and translation != text:
            translation_cache[text] = translation
            save_cache(cache_file)
            return translation
    except Exception:
        pass
    
    # API失败或结果无效，保存原文到缓存，下次不再尝试
    translation_cache[text] = text
    save_cache(cache_file)
    return text


def translate_csv_file(csv_path, name_column, output_path=None, cache_file=None):
    """翻译CSV文件中的化合物名和分类信息
    
    翻译格式：统一使用 "English (中文)" 格式直接替换原列值
    - matched_name → 在原列替换为 "English (中文)" 格式
    - matched_ontology → 在原列替换为 "English (中文)" 格式
    """
    print(f"\n处理文件: {csv_path}")
    print(f"  名称列: {name_column}")
    
    # 读取CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        print(f"  读取失败: {e}")
        return False
    
    if name_column not in df.columns:
        print(f"  错误: 未找到列 '{name_column}'")
        print(f"  可用列: {list(df.columns)}")
        return False
    
    # 定义需要翻译的列
    columns_to_translate = [name_column]
    if 'matched_ontology' in df.columns:
        columns_to_translate.append('matched_ontology')
    
    # 移除旧格式 matched_name_cn 列（已废弃，翻译直接写入 matched_name）
    if 'matched_name_cn' in df.columns:
        df = df.drop(columns=['matched_name_cn'])
    
    total_translated = 0
    total_success = 0
    
    for col in columns_to_translate:
        # 收集需要翻译的条目（纯英文 或 英文前缀+中文后缀的混合体）
        items_to_translate = []
        indices_to_translate = []
        partial_info = {}  # {idx: (english_prefix, chinese_suffix)}  部分翻译信息
        for idx in range(len(df)):
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if not val_str or val_str == 'nan' or val_str == 'N/A':
                continue
            # 检查是否已有完整中文翻译
            if has_chinese_translation(val_str):
                continue
            # 检查是否为部分翻译（英文前缀+中文后缀）
            if is_partial_translation(val_str):
                prefix, suffix = split_partial_translation(val_str)
                if prefix and suffix:
                    items_to_translate.append(prefix)  # 只翻译英文前缀
                    indices_to_translate.append(idx)
                    partial_info[idx] = (prefix, suffix)
                    continue
            # 纯英文，全量翻译
            items_to_translate.append(val_str)
            indices_to_translate.append(idx)
        
        if not items_to_translate:
            print(f"  [{col}] 无需翻译（已全部含中文）")
            continue
        
        print(f"  [{col}] 需翻译: {len(items_to_translate)} 条")

        # 使用批量翻译
        items = list(zip(indices_to_translate, items_to_translate))
        translations = translate_batch_threaded(items, cache_file, max_workers=5)

        # 更新DataFrame
        for idx, original_val in zip(indices_to_translate, items_to_translate):
            cn_val = translations.get(idx, original_val)
            if cn_val and cn_val != original_val and is_chinese(cn_val):
                # 统一: 在原列替换为 "English (中文)" 格式
                if idx in partial_info:
                    # 部分翻译：英文前缀 → "English (中文)"，再拼回中文后缀
                    english_prefix, chinese_suffix = partial_info[idx]
                    new_val = f"{english_prefix} ({cn_val}) {chinese_suffix}"
                else:
                    new_val = f"{original_val} ({cn_val})"
                df.at[idx, col] = new_val
                total_success += 1
            elif idx in partial_info and cn_val == original_val:
                # API 对英文前缀翻译失败（返回原文），保留混合原文
                pass
            total_translated += 1

        print(f"  [{col}] 完成: {total_translated} 条, 成功 {total_success} 条")
    
    if total_translated == 0:
        print("  无需翻译任何内容")
    else:
        print(f"  总计翻译: {total_translated} 条, 成功 {total_success} 条")
    
    # 保存结果（默认回写原文件）
    if output_path is None:
        output_path = csv_path
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  已保存: {output_path}")
    
    return True


def guess_name_column(columns):
    """猜测化合物名列（按候选词优先级鉴定，最具体的优先）"""
    candidates = ['matched_name', 'target_compound', 'compound_name', 'candidate_name', 
                 'formula_name', 'identified_name', 'structure_name']
    columns_lower = {col: col.lower() for col in columns}
    # 按候选词优先级遍历（最具体的优先鉴定）
    for cand in candidates:
        for col, col_lower in columns_lower.items():
            if cand == col_lower:  # 精确鉴定优先
                return col
    for cand in candidates:
        for col, col_lower in columns_lower.items():
            if cand in col_lower:  # 子串鉴定
                return col
    return None


def process_directory(results_dir, cache_file):
    """处理整个结果目录"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"错误: 目录不存在: {results_dir}")
        return
    
    print("=" * 60)
    print(f"处理目录: {results_dir}")
    print("=" * 60)
    
    # 加载缓存
    load_cache(cache_file)
    
    # 查找CSV文件
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    # 确定要处理的CSV和对应的名称列
    file_columns = {
        'L1_results.csv': 'matched_name',
        'L2_results.csv': 'matched_name',
        'L2_results_CN.csv': 'matched_name',  # 如果已有CN版本
    }
    
    # 对于L4，需要检查是否有CSV
    l3_csv = list(results_dir.glob("*compound*.csv")) + list(results_dir.glob("*formula*.csv"))
    if l3_csv:
        csv_files.extend(l3_csv)
    
    processed = 0
    
    for csv_file in csv_files:
        # 确定名称列
        name_col = file_columns.get(csv_file.name)
        if name_col is None:
            # 尝试自动检测
            try:
                df = pd.read_csv(csv_file, nrows=1, encoding='utf-8')
                name_col = guess_name_column(df.columns)
            except:
                continue
        
        if name_col is None:
            print(f"\n跳过 {csv_file.name}: 无法确定化合物名列")
            continue
        
        # 翻译（回写原文件）
        if translate_csv_file(str(csv_file), name_col, str(csv_file), cache_file):
            processed += 1
    
    print("\n" + "=" * 60)
    print(f"处理完成: {processed} 个文件")
    print(f"缓存记录: {len(translation_cache)} 条")
    print(f"失败记录: {len(failed_translations)} 条")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='化合物名翻译脚本')
    parser.add_argument('directory', help='结果目录路径')
    parser.add_argument('--cache', '-c', default=DEFAULT_CACHE, 
                        help='翻译缓存文件路径')
    parser.add_argument('--input', '-i', help='指定输入CSV文件')
    parser.add_argument('--column', '-col', help='化合物名列名')
    parser.add_argument('--output', '-o', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 处理
    if args.input:
        # 单文件模式
        load_cache(args.cache)
        name_col = args.column
        if name_col is None:
            df = pd.read_csv(args.input, nrows=1, encoding='utf-8')
            name_col = guess_name_column(df.columns)
        if name_col is None:
            print("错误: 无法自动确定化合物名列，请使用 --column 指定")
            return
        translate_csv_file(args.input, name_col, args.output, args.cache)
    else:
        # 目录模式
        process_directory(args.directory, args.cache)


if __name__ == '__main__':
    main()
