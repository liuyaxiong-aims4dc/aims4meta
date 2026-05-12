#!/usr/bin/env python3
"""
L2: DreaMS鉴定（模拟数据库）

策略：
- 复用L1的DreaMS鉴定逻辑
- 候选库：L2生成的模拟库embedding（FIORA + CFM-ID）
- 参数：使用L2特定的容差和阈值

输入：原始样品MSP + 样品embedding + 模拟库MSP + 模拟库embedding
输出：L2_DreaMS_results.csv
"""

import os
import sys
import importlib.util

###############################################################################
# 复用L1的DreaMS鉴定逻辑
###############################################################################

# 导入L1的DreaMS鉴定脚本
_l1_dreams_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'L1_实验数据库鉴定', 'L1_DreaMS鉴定.py')
_spec = importlib.util.spec_from_file_location("L1_DreaMS", os.path.abspath(_l1_dreams_path))
_l1_dreams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_l1_dreams)

# 复用L1的main函数
main = _l1_dreams.main

if __name__ == "__main__":
    result = main()
    # main()返回匹配数(>=0为正常), None或负数为异常
    # 0匹配是正常结果,不应视为失败
    sys.exit(0 if result is not None and result >= 0 else 1)
