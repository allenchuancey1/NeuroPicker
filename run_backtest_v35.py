"""
================================================================================
v35.0 回测脚本
基于backtest_engine运行v35生成的调仓文件回测
================================================================================
运行: python run_backtest_v35.py
================================================================================
"""

import os
import sys
import json

# 引入回测引擎（仓库内 backtest/ 子目录，与本项目一同发布）
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'backtest'))
from backtest_engine import BacktestEngine

# 配置：复用主配置中的输出目录，便于统一路径
from config_v35 import PATHS
REBALANCE_DIR = PATHS['rebalance_dir']
OUTPUT_DIR = os.path.join(os.path.dirname(PATHS['output_dir']), 'backtest_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("v35.0 回测")
print("=" * 70)

# 回测Top3和Top10
for top_n in [3, 10]:
    bt_input = os.path.join(REBALANCE_DIR, f'v35_bilstm_bt_{top_n}stocks.csv')

    if not os.path.exists(bt_input):
        print(f"\n  跳过 Top{top_n}: 文件不存在 {bt_input}")
        continue

    print(f"\n  Top{top_n} 回测...")
    print(f"  输入: {bt_input}")

    engine = BacktestEngine()
    result = engine.run(bt_input)

    result_path = os.path.join(OUTPUT_DIR, f'backtest_v35_bilstm_{top_n}stocks.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"  总收益率: {result['total_return']:.2%}")
    print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"  胜率: {result['win_rate']:.2%}")
    print(f"  结果保存: {result_path}")

print("\n" + "=" * 70)
print("v35.0 回测完成!")
print("=" * 70)
