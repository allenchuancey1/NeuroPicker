"""
================================================================================
回测系统使用示例
================================================================================

使用方法:
    python example_usage.py

说明:
    本文件展示如何使用配置化的回测系统。
    所有参数在 config.py 中统一管理，无需修改代码。
================================================================================
"""

import pandas as pd
from backtest_engine import run_backtest
from config import update_config, show_config


def example_1_basic():
    """示例1: 基本用法 - 使用默认配置运行回测。"""
    print("="*60)
    print("示例1: 基本用法")
    print("="*60)
    
    # 准备调仓数据
    rebalance_data = pd.DataFrame({
        'trade_date': ['2010-02-01', '2010-02-01', '2010-02-01', '2010-02-01',
                       '2010-03-02', '2010-03-02', '2010-03-02', '2010-03-02'],
        'sec_code': ['1', '2', '6', '7', '1', '2', '6', '7'],
        'weight': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    })
    rebalance_data.to_csv('example_strategy.csv', index=False)
    
    # 运行回测 (使用config.py中的默认配置)
    results = run_backtest('example_strategy.csv', 'example_result.json')
    
    print(f"\n回测完成!")
    print(f"总收益率: {results.get('total_return', 0):.2%}")
    print(f"夏普比率: {results.get('sharpe_ratio', 0)}")


def example_2_custom_params():
    """示例2: 临时修改参数运行回测。"""
    print("\n" + "="*60)
    print("示例2: 临时修改参数")
    print("="*60)
    
    # 运行回测时临时修改参数，不影响config.py
    results = run_backtest(
        'example_strategy.csv',
        initial_cash=500000,      # 临时改为50万
        commission=0.0005,         # 临时改为万5
        verbose=False              # 临时关闭详细日志
    )
    
    print(f"\n回测完成 (初始资金50万)!")
    print(f"总收益率: {results.get('total_return', 0):.2%}")


def example_3_show_config():
    """示例3: 查看当前配置。"""
    print("\n" + "="*60)
    print("示例3: 查看配置")
    print("="*60)
    
    show_config()


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    # 运行所有示例
    example_1_basic()
    example_2_custom_params()
    example_3_show_config()
    
    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60)
    print("\n提示: 修改 config.py 可永久调整回测参数")
