
"""
================================================================================
v35.0 统一配置模块
================================================================================
所有配置参数集中管理，避免分散在各文件中
修改参数只需编辑此文件
================================================================================
"""

import os

# ============================================================================
# 0. 根目录解析 - 让仓库可移植，不再依赖固定盘符路径
# ============================================================================
# 项目根目录（本文件所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_home():
    """
    解析数据/输出根目录 HOME，按以下优先级：
      1. 环境变量 QUANT_V35_HOME（最高优先级，便于复用既有数据目录）
      2. 自动探测：若上层目录已存在 data/a_stock_raw_data，则沿用（兼容
         原有的 d:\\trace\\stock 本地布局，保证既有工作流不中断）
      3. 默认：项目根目录（新克隆仓库时的自包含布局）
    """
    env = os.environ.get('QUANT_V35_HOME')
    if env:
        return env
    candidates = [PROJECT_ROOT, os.path.dirname(os.path.dirname(PROJECT_ROOT))]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, 'data', 'a_stock_raw_data')):
            return cand
    return PROJECT_ROOT


HOME = _resolve_home()


# ============================================================================
# 1. 路径配置 - 数据和输出文件的存放位置
# ============================================================================
# 所有路径基于 HOME 拼接，可移植；如需指向既有数据目录，设置环境变量：
#   Windows:  set QUANT_V35_HOME=d:\trace\stock
#   Linux/Mac: export QUANT_V35_HOME=/data/trace/stock
PATHS = {
    'raw_data_path':   os.path.join(HOME, 'data', 'a_stock_raw_data', 'a_stock_all_data_20100101_20260511.csv'),
    'sector_path':     os.path.join(HOME, 'data', 'sector_mapping.csv'),
    'benefit_path':    os.path.join(HOME, 'data', 'fundamental', 'benefit_ths.csv'),
    'debt_path':       os.path.join(HOME, 'data', 'fundamental', 'debt_ths.csv'),
    'cash_path':       os.path.join(HOME, 'data', 'fundamental', 'cash_ths.csv'),
    'preprocess_dir':  os.path.join(HOME, 'preprocess', 'v35.0'),
    'tmp_memmap_dir':  os.path.join(HOME, 'tmp_memmap', 'v35'),
    'output_dir':      os.path.join(HOME, 'output', 'v35.0'),
    'model_dir':       os.path.join(HOME, 'models'),
    'rebalance_dir':   os.path.join(HOME, 'output', 'rebalance'),
}

# ============================================================================
# 2. 时间范围配置 - 训练、验证、预测的时间段
# ============================================================================
TIME = {
    'train_start': '2020-01-01',      # 训练数据起始日期
    'train_end': '2025-12-31',        # 训练数据结束日期
    'val_split_date': '2025-01-01',   # 训练集/验证集划分日期
    'predict_start': '2026-01-01',    # 预测起始日期
    'predict_end': '2026-06-03',      # 预测结束日期
}

# ============================================================================
# 3. 预处理配置 - 数据清洗和特征工程参数
# ============================================================================
PREPROCESS = {
    'lookback_days': 20,              # 回看天数（输入序列长度）
    'forward_days': 20,               # 预测天数（标签超前天数）
    'min_trading_days': 200,          # 股票最少交易天数筛选
    'max_suspension_days': 20,        # 最长连续停牌天数
    'mad_n': 3.0,                     # MAD去极值倍数（行业标准3倍）
    'mad_window': 252,                # MAD滚动窗口（约1年交易日）
    'multi_scale_windows': [5],       # 多尺度滚动窗口
    'label_type': 'rank_neutral',     # 标签类型：'rank_neutral'或'triple_residual'
}

# ============================================================================
# 4. IC特征筛选配置 - 用于选择有效因子
# ============================================================================
IC_FILTER = {
    'min_ic': 0.03,                   # 最小IC均值阈值
    'min_icir': 0.3,                  # 最小ICIR阈值
    'min_features': 15,               # 最少保留特征数
    'max_features': 40,               # 最多保留特征数
}

# ============================================================================
# 5. 模型架构配置 - BiLSTM+Attention模型参数
# ============================================================================
MODEL = {
    'input_size': 40,                 # 输入特征维度（实际由IC筛选动态决定）
    'hidden_size': 128,               # LSTM隐藏层维度（减小以避免内存溢出）
    'num_layers': 3,                  # LSTM层数（减小以避免过拟合）
    'dropout': 0.5,                   # Dropout比率（防止过拟合）
    'attention_heads': 4,              # 多头注意力头数（减小以避免内存溢出）
}

# ============================================================================
# 6. 训练过程配置
# ============================================================================
TRAIN = {
    'batch_size': 4096,              # 批次大小（减小以避免内存溢出）
    'max_epochs': 150,               # 最大训练轮数
    'patience': 15,                  # 早停耐心值（连续15轮不改善则停止）
    'lr': 3e-5,                     # 学习率
    'weight_decay': 1e-3,            # 权重衰减（L2正则化）
    'grad_clip': 0.5,                # 梯度裁剪阈值
    'use_amp': False,                # 禁用混合精度训练（避免nan）
    'lookback': 20,                  # 回看天数（与PREPROCESS保持一致）
    'forward': 20,                   # 预测天数（与PREPROCESS保持一致）
    'num_workers': 2,                # 数据加载worker数（减少内存占用）
}

# ============================================================================
# 7. 因子列定义 - 各类特征的名称列表
# ============================================================================

# 基础技术因子（15个）- 动量、波动率、量价类因子
TECH_FACTOR_COLS = [
    'price_accel_5d', 'overnight_gap', 'momentum_accel_20_60',
    'atr_14', 'realized_vol_5d', 'return_skew_20d', 'max_drawdown_20d',
    'smart_money', 'buy_pressure', 'buy_pressure_ma5', 'shrink_rise', 'intraday_return',
    'amihud_illiq', 'turnover_accel', 'vol_change',
]

# 基本面因子（9个）- 注意：已移除PE/PB，无总股本数据无法正确计算
FUNDAMENTAL_FACTOR_COLS = [
    'roe', 'debt_ratio', 'revenue_growth', 'profit_growth',
    'cash_flow_ratio', 'gross_margin', 'roa', 'revenue_accel', 'roe_stability',
]

# 情绪因子（3个）- 市场情绪相关因子
SENTIMENT_FACTOR_COLS = [
    'turnover_rate_change', 'price_volume_corr', 'volatility_regime',
]

# v29新增技术因子（10个）
V29_FACTOR_COLS = [
    'close_position', 'volume_surge', 'ret_vol_ratio', 'cum_momentum_20',
    'kline_body_ratio', 'vol_price_divergence', 'limit_freq_20',
    'intraday_range_ratio', 'overnight_ret_stability', 'turnover_accel_vol',
]

# 合并所有原始因子列（共37个 = 15+9+3+10）
FACTOR_COLS = TECH_FACTOR_COLS + FUNDAMENTAL_FACTOR_COLS + SENTIMENT_FACTOR_COLS + V29_FACTOR_COLS

# 行业动量特征（5个）
INDUSTRY_MOMENTUM_COLS = [
    'industry_mom_5d', 'industry_mom_10d', 'industry_mom_20d',
    'industry_turnover_chg', 'industry_vol',
]

# PCA隐含因子（3个）
PCA_FACTOR_COLS = ['pca_tech_1', 'pca_tech_2', 'pca_tech_3']

# 风格概率特征（5个）
STYLE_COLS = ['style_value', 'style_growth', 'style_quality', 'style_cycle', 'style_dividend']


# ============================================================================
# 辅助函数：确保所有输出目录存在
# ============================================================================
def ensure_dirs():
    """创建所有必要的输出目录，避免运行时找不到文件夹"""
    for key in ['preprocess_dir', 'tmp_memmap_dir', 'output_dir', 'model_dir', 'rebalance_dir']:
        os.makedirs(PATHS[key], exist_ok=True)


# 初始化：创建必要目录
ensure_dirs()

