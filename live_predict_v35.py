"""
================================================================================
v35.0 实盘预测模块
================================================================================
功能: 输入预测日期，生成次日调仓文件
用法:
  python live_predict_v35.py --date 2026-06-05
  python live_predict_v35.py --date 2026-06-05 --top 10
  python live_predict_v35.py --date 2026-06-05 --model BiLSTM_Attention_v35_best.pth

特性:
  - 直接导入 v35.0 的模型定义和配置（架构100%一致）
  - 自动检测预处理数据是否覆盖目标日期，不够时从原始行情实时补算因子
  - 剔除北交所(8xxxxx/9xxxxx)和科创板(688xxx)
  - 支持选择不同预处理版本和模型文件

输出:
  output/rebalance/live_{date}_top{N}.csv
    - date, rank, asset, score, weight
================================================================================
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# 导入v35模块
from config_v35 import PATHS, TIME, PREPROCESS, MODEL as MODEL_CONFIG
from model_v35 import BiLSTMAttentionV2

warnings.filterwarnings('ignore')


# ============================================================================
# 常量配置
# ============================================================================

# 剔除板块：北交所(8xxxxx/9xxxxx)、科创板(688xxx)
EXCLUDE_PREFIXES = ('8', '9', '688')

# 默认模型文件名
DEFAULT_MODEL = 'BiLSTM_Attention_v35_best.pth'


# ============================================================================
# 因子计算函数（用于扩展预处理数据到最新日期）
# ============================================================================

def compute_basic_factors(df):
    """
    计算基础技术因子（15个）
    
    包含:
    - 价格加速度(price_accel_5d)
    - 隔夜跳空(overnight_gap)
    - 动量加速度(momentum_accel_20_60)
    - ATR波动率(atr_14)
    - 实现波动率(realized_vol_5d)
    - 收益偏度(return_skew_20d)
    - 最大回撤(max_drawdown_20d)
    - 聪明钱(smart_money)
    - 买盘压力(buy_pressure, buy_pressure_ma5)
    - 量缩价涨(shrink_rise)
    - 日内收益(intraday_return)
    - Amihud非流动性(amihud_illiq)
    - 换手率加速度(turnover_accel)
    - 成交量变化(vol_change)
    """
    df = df.sort_values(['asset', 'date']).copy()
    df['daily_ret'] = df['close'].pct_change()

    # 价格加速度 5日
    ret_5 = df.groupby('asset')['close'].pct_change(5)
    ret_1 = df.groupby('asset')['close'].pct_change(1)
    df['price_accel_5d'] = ret_5 - ret_1

    # 隔夜跳空（开盘价相对前收盘的变化）
    df['overnight_gap'] = df['open'] / df['close'].shift(1) - 1

    # 动量加速度 20-60日
    ret_20 = df.groupby('asset')['close'].pct_change(20)
    ret_60 = df.groupby('asset')['close'].pct_change(60)
    df['momentum_accel_20_60'] = ret_20 - ret_60

    # ATR 14日（平均真实波幅）
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = df.groupby('asset')[tr.name].transform(
        lambda x: x.rolling(14, min_periods=7).mean())

    # 实现波动率 5日
    df['realized_vol_5d'] = df.groupby('asset')['daily_ret'].transform(
        lambda x: x.rolling(5, min_periods=3).std())

    # 收益偏度 20日
    df['return_skew_20d'] = df.groupby('asset')['daily_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).skew())

    # 最大回撤 20日
    def max_drawdown(s):
        cum = (1 + s).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()
    df['max_drawdown_20d'] = df.groupby('asset')['daily_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).apply(max_drawdown, raw=True))

    # 聪明钱（收盘价在日内高低范围的位置）
    df['smart_money'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # 买入压力（收盘价相对开盘价的位置）
    df['buy_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)

    # 买入压力 5日均值
    df['buy_pressure_ma5'] = df.groupby('asset')['buy_pressure'].transform(
        lambda x: x.rolling(5, min_periods=3).mean())

    # 量缩价涨（成交量减少但价格上涨）
    vol_chg = df.groupby('asset')['volume'].pct_change()
    ret_sign = np.sign(df['daily_ret'])
    df['shrink_rise'] = ret_sign * vol_chg

    # 日内收益（收盘价相对开盘价的变化率）
    df['intraday_return'] = (df['close'] - df['open']) / df['open']

    # Amihud非流动性（收益绝对值除以成交额）
    df['amihud_illiq'] = (df['daily_ret'].abs() / (df['volume'] * df['close'] + 1e-8)).groupby(df['asset']).transform(
        lambda x: x.rolling(20, min_periods=10).mean())

    # 换手率加速度（相对20日均量的变化）
    vol_ma20 = df.groupby('asset')['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    df['turnover_accel'] = df['volume'] / (vol_ma20 + 1e-8) - 1

    # 成交量变化（5日变化率）
    df['vol_change'] = vol_chg

    return df


def compute_sentiment_factors(df):
    """
    计算情绪因子（3个）
    
    包含:
    - 换手率变化(turnover_rate_change)
    - 量价相关性(price_volume_corr)
    - 波动率regime(volatility_regime)
    """
    df = df.sort_values(['asset', 'date']).copy()

    # 换手率变化 = 当日量/20日均量 - 1
    avg_vol_20 = df.groupby('asset')['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    df['turnover_rate_change'] = df['volume'] / (avg_vol_20 + 1e-8) - 1

    # 量价相关性：20日收益与成交量的相关系数
    df['price_volume_corr'] = df.groupby('asset').apply(
        lambda g: g['daily_ret'].rolling(20, min_periods=10).corr(g['volume'])
    ).reset_index(level=0, drop=True)

    # 波动率regime = 当前波动/60日均波动 - 1
    vol_60 = df.groupby('asset')['realized_vol_5d'].transform(
        lambda x: x.rolling(60, min_periods=30).mean())
    df['volatility_regime'] = df['realized_vol_5d'] / (vol_60 + 1e-8) - 1

    return df


def compute_v29_factors(df):
    """
    计算v29新增技术因子（10个）
    
    包含:
    - 价格位置(close_position)
    - 成交量突变(volume_surge)
    - 收益波动比(ret_vol_ratio)
    - 累积动量(cum_momentum_20)
    - K线实体比率(kline_body_ratio)
    - 量价背离(vol_price_divergence)
    - 涨跌停频率(limit_freq_20)
    - 日内振幅比(intraday_range_ratio)
    - 隔夜收益稳定性(overnight_ret_stability)
    - 换手率加速波动(turnover_accel_vol)
    """
    df = df.sort_values(['asset', 'date']).copy()

    # 价格位置（在20日高低点之间的位置）
    low_20 = df.groupby('asset')['low'].transform(lambda x: x.rolling(20, min_periods=10).min())
    high_20 = df.groupby('asset')['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
    df['close_position'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-8)

    # 成交量突变（相对20日均量的倍数）
    avg_vol_20 = df.groupby('asset')['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    df['volume_surge'] = df['volume'] / (avg_vol_20 + 1e-8)

    # 收益波动比（20日收益均值/标准差）
    ret_mean_20 = df.groupby('asset')['daily_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    ret_std_20 = df.groupby('asset')['daily_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).std())
    df['ret_vol_ratio'] = ret_mean_20 / (ret_std_20 + 1e-8)

    # 累积动量（20日累积收益率）
    df['cum_momentum_20'] = df.groupby('asset')['daily_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).apply(lambda s: (1 + s).prod() - 1, raw=True))

    # K线实体比率（实体/整根K线的比例）
    df['kline_body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)

    # 量价背离（价格涨跌与成交量变化的符号差异）
    vol_chg_20 = df.groupby('asset')['volume'].pct_change(20)
    ret_20d_sum = df.groupby('asset')['daily_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).apply(lambda s: s.sum(), raw=True))
    df['vol_price_divergence'] = np.sign(ret_20d_sum) * (1 - np.sign(vol_chg_20))

    # 涨跌停频率（简化处理，实时数据无涨跌停标记）
    df['limit_freq_20'] = 0

    # 日内振幅比（最高-最低）/收盘价
    df['intraday_range_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-8)

    # 隔夜收益稳定性（负的隔夜收益波动）
    df['overnight_ret_stability'] = -df.groupby('asset')['overnight_gap'].transform(
        lambda x: x.rolling(20, min_periods=10).std())

    # 换手率加速波动（换手率变化的波动率）
    df['turnover_accel_vol'] = df.groupby('asset')['turnover_rate_change'].transform(
        lambda x: x.rolling(20, min_periods=10).std())

    return df


def compute_multi_scale(df, factor_cols, windows=None):
    """
    多尺度时序融合
    
    对给定因子计算滚动均值，用于捕捉不同时间尺度的趋势
    """
    if windows is None:
        windows = PREPROCESS.get('multi_scale_windows', [5])
    ms_cols = []
    for window in windows:
        for col in factor_cols:
            if col not in df.columns:
                continue
            ms_col = f'ms_{window}d_{col}'
            df[ms_col] = df.groupby('asset')[col].transform(
                lambda x: x.rolling(window, min_periods=max(window // 2, 3)).mean())
            ms_cols.append(ms_col)
    return df, ms_cols


def extend_factor_data(factor_data, target_date, preprocess_version='v35.0'):
    """
    如果预处理数据不覆盖目标日期，从原始行情补算因子
    
    策略：
    1. 取预处理数据最后90天 + 原始行情新增数据
    2. 重新计算所有因子
    3. 只保留新增日期的结果拼接到预处理数据后面
    
    参数:
        factor_data: 已加载的预处理因子数据
        target_date: 目标预测日期
        preprocess_version: 预处理版本目录名
    
    返回:
        扩展后的因子数据
    """
    max_preprocess_date = factor_data['date'].max()
    target_dt = pd.Timestamp(target_date)

    # 如果预处理数据已经覆盖目标日期，无需补算
    if target_dt <= max_preprocess_date:
        print(f"  ✓ 预处理数据已覆盖目标日期 {target_dt.date()}")
        return factor_data

    print(f"  ⚠ 预处理数据截止到 {max_preprocess_date.date()}，目标日期 {target_dt.date()}")
    print(f"  从原始行情补算因子...")

    # 计算需要加载的原始行情日期范围
    raw_start = (max_preprocess_date - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    raw_end = target_dt.strftime('%Y-%m-%d')

    print(f"  读取原始行情 ({raw_start} ~ {raw_end})...")
    chunks = []
    for chunk in pd.read_csv(PATHS['raw_data_path'], chunksize=500000,
                             parse_dates=['date']):
        mask = (chunk['date'] >= raw_start) & (chunk['date'] <= raw_end)
        chunks.append(chunk[mask])
        if chunk['date'].max() >= target_dt:
            break

    if not chunks:
        print(f"  ✗ 原始行情中无目标日期数据")
        return factor_data

    raw = pd.concat(chunks, ignore_index=True)
    raw['date'] = pd.to_datetime(raw['date'])
    print(f"  原始行情记录数: {len(raw):,}")

    # 获取预处理数据中的特征列（从meta文件）
    meta_file = os.path.join(PATHS['preprocess_dir'], 'preprocess_meta.json')
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        all_feature_cols = meta.get('selected_features', [])
    else:
        # 从factor_data推断特征列
        all_feature_cols = [c for c in factor_data.columns
                            if c not in ['asset', 'date', 'label_rank', 'noise_weight', 'sector']]

    # 计算技术因子
    print(f"  计算技术因子...")
    raw = compute_basic_factors(raw)
    raw = compute_sentiment_factors(raw)
    raw = compute_v29_factors(raw)

    # 基本面因子：从前向填充（实时无法获取最新财报）
    fundamental_cols = ['roe', 'debt_ratio', 'revenue_growth', 'profit_growth',
                       'cash_flow_ratio', 'gross_margin', 'roa', 'revenue_accel', 'roe_stability']
    for col in fundamental_cols:
        if col in factor_data.columns:
            latest_fund = factor_data.groupby('asset')[col].last().to_dict()
            raw[col] = raw['asset'].map(latest_fund)

    # 多尺度因子
    base_factor_cols = [
        'price_accel_5d', 'overnight_gap', 'momentum_accel_20_60',
        'atr_14', 'realized_vol_5d', 'return_skew_20d', 'max_drawdown_20d',
        'smart_money', 'buy_pressure', 'buy_pressure_ma5', 'shrink_rise',
        'intraday_return', 'amihud_illiq', 'turnover_accel', 'vol_change',
        'turnover_rate_change', 'price_volume_corr', 'volatility_regime',
        'close_position', 'volume_surge', 'ret_vol_ratio', 'cum_momentum_20',
        'kline_body_ratio', 'vol_price_divergence', 'limit_freq_20',
        'intraday_range_ratio', 'overnight_ret_stability', 'turnover_accel_vol',
    ]
    available_base = [c for c in base_factor_cols if c in raw.columns]
    raw, ms_cols = compute_multi_scale(raw, available_base)

    # 行业动量：简化处理，设为0
    for col in ['industry_mom_5d', 'industry_mom_10d', 'industry_mom_20d',
                'industry_turnover_chg', 'industry_vol']:
        if col in all_feature_cols and col not in raw.columns:
            raw[col] = 0

    # PCA因子：简化处理，设为0
    for col in ['pca_tech_1', 'pca_tech_2', 'pca_tech_3']:
        if col in all_feature_cols and col not in raw.columns:
            raw[col] = 0

    # 风格概率：简化处理，设为0.5
    for col in ['style_value', 'style_growth', 'style_quality', 'style_cycle', 'style_dividend']:
        if col in all_feature_cols and col not in raw.columns:
            raw[col] = 0.5

    # 只保留预处理数据中已有的列
    common_cols = ['asset', 'date'] + [c for c in all_feature_cols if c in raw.columns]
    new_data = raw[common_cols].copy()

    # 只保留预处理数据截止日期之后的新数据
    new_data = new_data[new_data['date'] > max_preprocess_date].copy()
    print(f"  新增因子数据: {len(new_data):,} 条 ({new_data['date'].min().date()} ~ {new_data['date'].max().date()})")

    # 拼接
    factor_data = pd.concat([factor_data, new_data], ignore_index=True)
    print(f"  拼接后总记录数: {len(factor_data):,}")

    return factor_data


# ============================================================================
# 数据加载
# ============================================================================

def load_preprocessed_data(preprocess_version='v35.0', target_date=None):
    """
    加载预处理因子数据
    
    参数:
        preprocess_version: 预处理版本目录名
        target_date: 目标预测日期（用于自动补算）
    
    返回:
        factor_data: 因子数据DataFrame
        feature_cols: 选中的特征列表
    """
    print("\n[1/4] 加载预处理数据...")

    preprocess_dir = os.path.join(os.path.dirname(PATHS['preprocess_dir']), preprocess_version)
    factor_file = os.path.join(preprocess_dir, 'factor_data.csv')
    feature_file = os.path.join(preprocess_dir, 'selected_features.json')

    # 检查文件是否存在
    if not os.path.exists(factor_file):
        print(f"  ✗ 预处理数据不存在: {factor_file}")
        print(f"  请先运行 preprocess_v35.py 生成预处理数据")
        sys.exit(1)
    if not os.path.exists(feature_file):
        print(f"  ✗ 特征列表不存在: {feature_file}")
        sys.exit(1)

    # 加载数据
    factor_data = pd.read_csv(factor_file)
    factor_data['date'] = pd.to_datetime(factor_data['date'])

    with open(feature_file, 'r') as f:
        feature_cols = json.load(f)

    print(f"  预处理版本: {preprocess_version}")
    print(f"  总记录数: {len(factor_data):,}")
    print(f"  股票数量: {factor_data['asset'].nunique():,}")
    print(f"  时间范围: {factor_data['date'].min().date()} ~ {factor_data['date'].max().date()}")
    print(f"  特征数量: {len(feature_cols)}")

    # 自动补算（如需要）
    if target_date:
        factor_data = extend_factor_data(factor_data, target_date, preprocess_version)

    return factor_data, feature_cols


# ============================================================================
# 模型加载
# ============================================================================

def load_model(model_name=None):
    """
    加载训练好的BiLSTM+Attention模型
    
    参数:
        model_name: 模型文件名（默认为v35最佳模型）
    
    返回:
        model: 加载的模型
        device: 计算设备
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    print(f"\n[2/4] 加载模型...")

    model_path = os.path.join(PATHS['model_dir'], model_name)
    if not os.path.exists(model_path):
        print(f"  ✗ 模型文件不存在: {model_path}")
        print(f"  可用模型:")
        for f in sorted(os.listdir(PATHS['model_dir'])):
            if f.endswith('.pth'):
                print(f"    - {f}")
        sys.exit(1)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  模型文件: {model_name}")
    print(f"  设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型权重（从权重推断输入维度）
    sd = torch.load(model_path, map_location='cpu', weights_only=True)
    input_size = sd['input_projection.0.weight'].shape[1]
    print(f"  输入维度: {input_size}")

    # 创建模型
    model = BiLSTMAttentionV2(
        input_size=input_size,
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=0.0,  # 预测时不使用dropout
        attention_heads=MODEL_CONFIG['attention_heads']
    )

    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # 统计参数量
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,}")

    return model, device


# ============================================================================
# 预测
# ============================================================================

def predict(model, device, factor_data, feature_cols, target_date, lookback=20):
    """
    对目标日期所有A股进行预测
    
    参数:
        model: 加载的模型
        device: 计算设备
        factor_data: 因子数据
        feature_cols: 特征列表
        target_date: 目标预测日期
        lookback: 回看天数
    
    返回:
        result: 预测结果DataFrame，包含asset和pred_score
    """
    print(f"\n[3/4] 生成预测 (目标日期: {target_date})...")

    target_dt = pd.Timestamp(target_date)

    # 找到目标日期之前最近的交易日
    available_dates = factor_data['date'].sort_values().unique()
    valid_dates = available_dates[available_dates <= target_dt]
    if len(valid_dates) == 0:
        print(f"  ✗ 目标日期之前无数据")
        return None

    actual_date = valid_dates[-1]
    if actual_date != target_dt:
        print(f"  目标日期非交易日，使用最近交易日: {actual_date.date()}")

    # 获取该日期的数据
    date_data = factor_data[factor_data['date'] == actual_date]
    print(f"  候选股票数: {len(date_data)}")

    # 构建预测批次
    batch_X = []
    batch_assets = []

    for _, row in date_data.iterrows():
        asset = row['asset']

        # 剔除北交所和科创板
        asset_str = str(asset).zfill(6)
        if asset_str.startswith(EXCLUDE_PREFIXES):
            continue

        # 获取该股票的历史数据
        asset_hist = factor_data[
            (factor_data['asset'] == asset) &
            (factor_data['date'] <= actual_date)
        ].sort_values('date').tail(lookback)

        if len(asset_hist) < lookback:
            continue

        # 提取特征并处理缺失值
        feat_vals = asset_hist[feature_cols].values.astype(np.float32)
        feat_df = pd.DataFrame(feat_vals)
        feat_df = feat_df.ffill().bfill().fillna(0)
        feat_vals = feat_df.values.astype(np.float32)

        batch_X.append(feat_vals)
        batch_assets.append(asset)

    if len(batch_X) == 0:
        print(f"  ✗ 无有效预测样本")
        return None

    print(f"  有效预测样本: {len(batch_X)} (已剔除北交所/科创板/数据不足)")

    # 批量预测
    X_tensor = torch.FloatTensor(np.array(batch_X)).to(device)
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                preds = model(X_tensor).cpu().numpy()
        else:
            preds = model(X_tensor).cpu().numpy()

    # 整理结果
    result = pd.DataFrame({
        'asset': batch_assets,
        'pred_score': preds.flatten()
    })
    result = result.sort_values('pred_score', ascending=False).reset_index(drop=True)

    print(f"  分数范围: [{result['pred_score'].min():.4f}, {result['pred_score'].max():.4f}]")

    return result


# ============================================================================
# 生成调仓文件
# ============================================================================

def generate_rebalance(result, target_date, top_n=3):
    """
    生成调仓文件
    
    输出格式:
        date, rank, asset, score, weight
    
    参数:
        result: 预测结果DataFrame
        target_date: 目标日期
        top_n: 选股数量
    
    返回:
        rebalance_file: 输出文件路径
    """
    print(f"\n[4/4] 生成调仓文件...")

    date_str = pd.Timestamp(target_date).strftime('%Y-%m-%d')

    # 选取Top N
    top = result.head(top_n).copy()
    top['rank'] = range(1, len(top) + 1)
    top['weight'] = round(1.0 / top_n, 6)

    # 构建调仓DataFrame
    rebalance = top[['rank', 'asset', 'pred_score', 'weight']].copy()
    rebalance.insert(0, 'date', date_str)
    rebalance = rebalance.rename(columns={'pred_score': 'score'})

    # 保存文件
    rebalance_file = os.path.join(PATHS['rebalance_dir'], f'live_{date_str}_top{top_n}.csv')
    rebalance.to_csv(rebalance_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ {rebalance_file}")

    # 打印调仓详情
    print(f"\n{'='*60}")
    print(f"  调仓日期: {date_str}    选股: Top {top_n}    等权分配")
    print(f"{'='*60}")
    for _, row in top.iterrows():
        print(f"  #{int(row['rank']):2d}  {row['asset']:>8s}  "
              f"分数:{row['pred_score']:+.4f}  权重:{row['weight']:.2%}")
    print(f"{'='*60}")

    return rebalance_file


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    实盘预测主函数
    
    命令行参数:
        --date: 预测日期 (必需)
        --top: 选股数量 (默认3)
        --model: 模型文件名 (默认BiLSTM_Attention_v35_best.pth)
        --preprocess: 预处理版本 (默认v35.0)
        --lookback: 回看天数 (默认20)
    """
    parser = argparse.ArgumentParser(
        description='v35.0 实盘预测 - 生成次日调仓文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python live_predict_v35.py --date 2026-06-05
  python live_predict_v35.py --date 2026-06-05 --top 10
  python live_predict_v35.py --date 2026-06-05 --model BiLSTM_Attention_v35_best.pth --preprocess v35.0
        """
    )
    parser.add_argument('--date', type=str, required=True,
                        help='预测日期 YYYY-MM-DD')
    parser.add_argument('--top', type=int, default=3,
                        help='选股数量 (默认3)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'模型文件名 (默认{DEFAULT_MODEL})')
    parser.add_argument('--preprocess', type=str, default='v35.0',
                        help='预处理版本目录名 (默认v35.0)')
    parser.add_argument('--lookback', type=int, default=PREPROCESS['lookback_days'],
                        help=f'回看天数 (默认{PREPROCESS["lookback_days"]})')
    args = parser.parse_args()

    # 打印配置
    print("=" * 60)
    print("  v35.0 实盘预测系统")
    print("=" * 60)
    print(f"  预测日期: {args.date}")
    print(f"  选股数量: Top {args.top}")
    print(f"  模型: {args.model}")
    print(f"  预处理: {args.preprocess}")
    print(f"  回看天数: {args.lookback}")
    print(f"  剔除板块: 北交所(8xxxxx/9xxxxx) + 科创板(688xxx)")
    print("=" * 60)

    t_start = time.time()

    # 加载数据
    factor_data, feature_cols = load_preprocessed_data(args.preprocess, args.date)
    
    # 加载模型
    model, device = load_model(args.model)
    
    # 生成预测
    result = predict(model, device, factor_data, feature_cols, args.date, args.lookback)

    if result is None:
        print("\n✗ 预测失败!")
        sys.exit(1)

    # 生成调仓文件
    rebalance_file = generate_rebalance(result, args.date, args.top)

    print(f"\n✓ 完成! 耗时 {time.time() - t_start:.1f}s")
    print(f"  调仓文件: {rebalance_file}")


if __name__ == '__main__':
    main()

