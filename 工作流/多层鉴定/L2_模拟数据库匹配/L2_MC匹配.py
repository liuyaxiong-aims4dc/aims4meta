#!/usr/bin/env python3
"""
L2: MC匹配（模拟数据库）

策略：
- 复用L1的MC匹配逻辑
- 候选库：L2生成的模拟库（FIORA + CFM-ID）
- 参数：使用L2特定的容差和阈值

输入：原始样品MSP + 模拟库MSP
输出：L2_MC_results.csv
"""

import os
import sys
import importlib.util

###############################################################################
# 复用L1的MC匹配逻辑
###############################################################################

# 导入L1的MC匹配脚本
_l1_mc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'L1_真实数据库匹配', 'L1_MC匹配.py')
_spec = importlib.util.spec_from_file_location("L1_MC", os.path.abspath(_l1_mc_path))
_l1_mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_l1_mc)

# 复用L1的main函数
main = _l1_mc.main

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
