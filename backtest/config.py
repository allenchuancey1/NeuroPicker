"""
================================================================================
回测系统配置文件
================================================================================

所有可调参数集中在此文件，无需修改代码即可调整回测设置。

使用方法:
    from config import BACKTEST_CONFIG, DATA_CONFIG
    
    # 修改初始资金
    BACKTEST_CONFIG['initial_cash'] = 500000
================================================================================
"""

import os

# ============================================================================
# 数据配置
# ============================================================================

# 价格数据根目录：默认回测模块所在目录的上一级（项目根），
# 可通过环境变量 QUANT_V35_HOME 覆盖，与主系统 config_v35.PATHS 保持一致
_BACKTEST_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKTEST_ROOT)
_HOME = os.environ.get('QUANT_V35_HOME', _PROJECT_ROOT)
_PRICE_CSV = os.environ.get(
    'QUANT_V35_PRICE_DATA',
    os.path.join(_HOME, 'data', 'a_stock_raw_data', 'a_stock_all_data_20100101_20260611.csv'),
)

DATA_CONFIG = {
    # CSV文件路径
    'price_data_path': _PRICE_CSV,
    
    # 日期格式
    'date_format': '%Y-%m-%d',
    
    # 数据列名映射 (如果CSV列名不同，在此修改)
    'column_mapping': {
        'date': 'date',
        'asset': 'asset',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
    },
    
    # 数据加载参数
    'chunk_size': 100000,  # 分块读取大小
    'cache_enabled': True,  # 是否启用数据缓存
}


# ============================================================================
# 回测参数配置
# ============================================================================

BACKTEST_CONFIG = {
    # 初始资金 (元)
    'initial_cash': 1000000.0,
    
    # 手续费率 (默认万3)
    'commission': 0.0003,
    
    # 滑点率 (默认千1)
    'slippage': 0.001,
    
    # 保留现金比例 (防止资金不足)
    'reserve_cash': 0.05,
    
    # 最大持仓股票数
    'max_stocks': 50,
    
    # 调仓日后延长天数 (确保调仓能执行)
    'extend_days': 30,
    
    # 数据提前天数 (确保第一个调仓日有数据)
    'pre_days': 30,
    
    # 是否打印详细日志
    'verbose': True,
    
    # 是否绘制图表
    'plot_results': False,
}


# ============================================================================
# 策略默认配置
# ============================================================================

STRATEGY_CONFIG = {
    # 默认调仓周期 (天)
    'rebalance_period': 20,
    
    # 权重阈值 (低于此值不买入)
    'weight_threshold': 0.01,
    
    # 是否允许部分成交
    'partial_fill': True,
}


# ============================================================================
# 输出配置
# ============================================================================

OUTPUT_CONFIG = {
    # 结果保存路径
    'result_dir': './results',
    
    # 默认结果文件名
    'default_result_file': 'backtest_result.json',
    
    # 图表保存路径
    'chart_file': 'backtest_chart.png',
    
    # 是否保存每日持仓
    'save_daily_positions': False,
    
    # 是否保存交易记录
    'save_trades': True,
}


# ============================================================================
# 便捷函数
# ============================================================================

def get_config(config_name='backtest'):
    """
    获取配置字典。
    
    参数:
        config_name: 'backtest', 'data', 'strategy', 'output'
        
    返回:
        dict: 配置字典
    """
    configs = {
        'backtest': BACKTEST_CONFIG,
        'data': DATA_CONFIG,
        'strategy': STRATEGY_CONFIG,
        'output': OUTPUT_CONFIG,
    }
    return configs.get(config_name, BACKTEST_CONFIG)


def update_config(config_name, **kwargs):
    """
    更新配置参数。
    
    示例:
        update_config('backtest', initial_cash=500000, commission=0.0005)
    """
    config = get_config(config_name)
    config.update(kwargs)
    print(f"已更新 {config_name} 配置: {kwargs}")


def show_config():
    """显示所有配置。"""
    print("="*60)
    print("回测系统配置")
    print("="*60)
    
    print("\n【数据配置】")
    for k, v in DATA_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\n【回测配置】")
    for k, v in BACKTEST_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\n【策略配置】")
    for k, v in STRATEGY_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\n【输出配置】")
    for k, v in OUTPUT_CONFIG.items():
        print(f"  {k}: {v}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    show_config()
