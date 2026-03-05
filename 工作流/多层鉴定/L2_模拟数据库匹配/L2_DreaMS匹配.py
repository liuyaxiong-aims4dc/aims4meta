#!/usr/bin/env python3
"""
L2: DreaMS匹配（模拟数据库）

策略：
- 复用L1的DreaMS匹配逻辑
- 候选库：L2生成的模拟库embedding（FIORA + CFM-ID）
- 参数：使用L2特定的容差和阈值

输入：原始样品MSP + 样品embedding + 模拟库MSP + 模拟库embedding
输出：L2_DreaMS_results.csv
"""

import os
import sys
import importlib.util

###############################################################################
# 复用L1的DreaMS匹配逻辑
###############################################################################

# 导入L1的DreaMS匹配脚本
_l1_dreams_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'L1_真实数据库匹配', 'L1_DreaMS匹配.py')
_spec = importlib.util.spec_from_file_location("L1_DreaMS", os.path.abspath(_l1_dreams_path))
_l1_dreams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_l1_dreams)

# 复用L1的main函数
main = _l1_dreams.main

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
