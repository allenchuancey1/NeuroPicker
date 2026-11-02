"""
================================================================================
v35.0 标签计算模块
================================================================================
职责: 标签计算、风格概率、噪声权重
修复:
  - PCA噪声剥离: 移除无效的1维PCA，简化为行业均值减除（已在行业中性化中完成）
  - 风格概率: turnover_accel_vol重命名为style_cycle_vol避免覆盖v29因子
================================================================================
"""

import gc
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config_v35 import PREPROCESS, STYLE_COLS

warnings.filterwarnings('ignore')


# ============================================================================
# 标签计算
# ============================================================================

def compute_industry_neutral_labels(df, sector_map, forward=20):
    """行业中性化标签: 个股收益减去行业平均收益"""
    print(f"[标签计算] 行业中性化标签 (forward={forward}天)...")
    t0 = time.time()

    df.sort_values(['asset', 'date'], inplace=True)
    df['future_return'] = df.groupby('asset')['close'].shift(-forward) / df['close'] - 1
    df['sector'] = df['asset'].map(sector_map.get('sector', ''))

    daily_industry_ret = df.groupby(['date', 'sector'])['future_return'].mean()
    ret_map = daily_industry_ret.to_dict()
    keys = list(zip(df['date'].values, df['sector'].values))
    df['industry_future_return'] = pd.Series([ret_map.get(k, np.nan) for k in keys], index=df.index)
    df['label_industry_neutral'] = df['future_return'] - df['industry_future_return']

    valid = df['label_industry_neutral'].notna()
    print(f"  有效样本: {valid.sum():,} / {len(df):,}")
    print(f"  降噪比例: {1 - df.loc[valid, 'label_industry_neutral'].std() / df.loc[valid, 'future_return'].std():.2%}")
    print(f"  耗时: {time.time()-t0:.1f}s")
    return df


def compute_triple_residual_labels(df, sector_map, forward=20):
    """
    三重残差标签: 行业中性+风格中性+时间序列去趋势
    
    修复: 移除原PCA噪声剥离步骤（在1维数据上做PCA无意义），
    简化为: 行业中性 + 风格中性 + 时序去趋势
    """
    print(f"[标签计算] 三重残差标签 (forward={forward}天)...")
    t0 = time.time()

    df.sort_values(['asset', 'date'], inplace=True)
    df['future_return'] = df.groupby('asset')['close'].shift(-forward) / df['close'] - 1
    df['sector'] = df['asset'].map(sector_map.get('sector', ''))

    # Step 1: 行业中性化
    daily_industry_ret = df.groupby(['date', 'sector'])['future_return'].mean()
    ret_map = daily_industry_ret.to_dict()
    keys = list(zip(df['date'].values, df['sector'].values))
    df['industry_future_return'] = pd.Series([ret_map.get(k, np.nan) for k in keys], index=df.index)
    df['r_ind'] = df['future_return'] - df['industry_future_return']

    # Step 2: 风格中性化
    df = _compute_style_neutral(df)

    # Step 3: 时间序列去趋势
    df['r_ts'] = df.groupby('asset')['r_style'].transform(
        lambda x: x - x.rolling(60, min_periods=20).mean()
    )

    # 不再做PCA噪声剥离（1维PCA无意义），直接使用时序去趋势结果
    df['label_triple_residual'] = df['r_ts']

    valid = df['label_triple_residual'].notna()
    print(f"  有效样本: {valid.sum():,} / {len(df):,}")
    if valid.sum() > 0:
        noise_ratio = 1 - df.loc[valid, 'r_ts'].std() / df.loc[valid, 'future_return'].std()
        print(f"  总降噪比例: {noise_ratio:.2%}")
    print(f"  耗时: {time.time()-t0:.1f}s")
    return df


def _compute_style_neutral(df):
    """风格中性化"""
    if 'market_cap_proxy' not in df.columns:
        df['market_cap_proxy'] = df['close'] * df['volume']

    for col, score_col in [
        ('market_cap_proxy', '_value_rank'),
        ('roe', '_growth_rank'),
        ('debt_ratio', '_quality_rank'),
        ('turnover_accel', '_cycle_rank'),
        ('realized_vol_5d', '_dividend_rank'),
    ]:
        if col in df.columns:
            df[score_col] = df.groupby('date')[col].rank(pct=True)
        else:
            df[score_col] = 0.5

    style_returns = {}
    for rank_col, style_name in [
        ('_value_rank', 'value'),
        ('_growth_rank', 'growth'),
        ('_quality_rank', 'quality'),
        ('_cycle_rank', 'cycle'),
        ('_dividend_rank', 'dividend'),
    ]:
        if style_name in ('value', 'quality', 'dividend'):
            top_mask = df[rank_col] <= 0.2
            bot_mask = df[rank_col] >= 0.8
        else:
            top_mask = df[rank_col] >= 0.8
            bot_mask = df[rank_col] <= 0.2

        top_ret = df[top_mask].groupby('date')['r_ind'].mean()
        bot_ret = df[bot_mask].groupby('date')['r_ind'].mean()
        style_ret = (top_ret - bot_ret).reset_index()
        style_ret.columns = ['date', f'style_ret_{style_name}']
        style_returns[style_name] = style_ret

    style_ret_map = {}
    for style_name, style_ret in style_returns.items():
        col = f'style_ret_{style_name}'
        for _, row in style_ret.iterrows():
            d = row['date']
            if d not in style_ret_map:
                style_ret_map[d] = {}
            style_ret_map[d][col] = row[col]

    style_ret_cols = [f'style_ret_{s}' for s in ['value', 'growth', 'quality', 'cycle', 'dividend']]
    for col in style_ret_cols:
        df[col] = df['date'].map(lambda d, c=col: style_ret_map.get(d, {}).get(c, np.nan))

    df['style_total_ret'] = df[style_ret_cols].mean(axis=1)
    df['r_style'] = df['r_ind'] - df['style_total_ret']

    temp_cols = ['_value_rank', '_growth_rank', '_quality_rank', '_cycle_rank', '_dividend_rank']
    temp_cols += style_ret_cols + ['style_total_ret']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns])
    return df


# ============================================================================
# 风格概率
# ============================================================================

def compute_style_probabilities(df, sector_map, fundamental_aligned=None):
    """
    计算风格概率(软标签条件)
    
    修复: 使用style_cycle_vol替代turnover_accel_vol，避免覆盖v29因子
    """
    print("[风格概率] 计算风格概率...")
    t0 = time.time()

    df['sector'] = df['asset'].map(sector_map.get('sector', ''))

    if 'roe' not in df.columns or 'debt_ratio' not in df.columns:
        if fundamental_aligned is not None and len(fundamental_aligned) > 0:
            fund = fundamental_aligned[['asset', 'date', 'roe', 'debt_ratio']].copy()
            fund['date'] = pd.to_datetime(fund['date'])
            df = df.drop(columns=['roe', 'debt_ratio'], errors='ignore')
            df = df.merge(fund, on=['asset', 'date'], how='left')
    if 'roe' not in df.columns:
        df['roe'] = np.nan
    if 'debt_ratio' not in df.columns:
        df['debt_ratio'] = np.nan

    if 'market_cap_proxy' not in df.columns:
        df['market_cap_proxy'] = df['close'] * df['volume']

    def industry_rank(series):
        return series.groupby(df['sector']).rank(pct=True)

    df['value_raw'] = 1.0 - industry_rank(df['market_cap_proxy'].fillna(
        df.groupby('asset')['market_cap_proxy'].transform('median')
    ))
    df['growth_raw'] = industry_rank(df['roe'].fillna(
        df.groupby('asset')['roe'].transform('median')
    ))
    df['quality_raw'] = 1.0 - industry_rank(df['debt_ratio'].fillna(
        df.groupby('asset')['debt_ratio'].transform('median')
    ))

    # 修复: 使用独立列名style_cycle_vol，不覆盖v29的turnover_accel_vol
    df['style_cycle_vol'] = df.groupby('asset')['turnover_accel'].transform(
        lambda x: x.rolling(60, min_periods=20).std()
    )
    df['cycle_raw'] = industry_rank(df['style_cycle_vol'].fillna(0.5))

    df['dividend_raw'] = 1.0 - industry_rank(df['realized_vol_5d'].fillna(
        df.groupby('asset')['realized_vol_5d'].transform('median')
    ))

    for raw_col in ['value_raw', 'growth_raw', 'quality_raw', 'cycle_raw', 'dividend_raw']:
        z_col = raw_col.replace('_raw', '_z')
        rolling_mean = df.groupby('asset')[raw_col].transform(
            lambda x: x.rolling(252, min_periods=60).mean()
        )
        rolling_std = df.groupby('asset')[raw_col].transform(
            lambda x: x.rolling(252, min_periods=60).std()
        )
        df[z_col] = (df[raw_col] - rolling_mean) / (rolling_std + 1e-8)

    for z_col, prob_col in [
        ('value_z', 'style_value'),
        ('growth_z', 'style_growth'),
        ('quality_z', 'style_quality'),
        ('cycle_z', 'style_cycle'),
        ('dividend_z', 'style_dividend'),
    ]:
        df[prob_col] = 1.0 / (1.0 + np.exp(-df[z_col].clip(-10, 10)))

    temp_cols = [c for c in df.columns if c.endswith('_raw') or c.endswith('_z')]
    temp_cols += ['style_cycle_vol']  # 清理临时列
    df = df.drop(columns=[c for c in temp_cols if c in df.columns])

    for col in STYLE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.5)

    print(f"  风格概率统计:")
    for col in STYLE_COLS:
        if col in df.columns:
            print(f"    {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
    print(f"  耗时: {time.time()-t0:.1f}s")
    gc.collect()
    return df


# ============================================================================
# 噪声权重
# ============================================================================

def compute_noise_weights(df):
    """分层噪声加权: 涨跌停/停牌/高波动/换手率突增/价格逻辑异常 均降权"""
    print("[噪声权重] 分层噪声加权...")
    t0 = time.time()

    df['noise_weight'] = 1.0

    # 高噪声
    high_noise = (
        df.get('is_limit_up', False) |
        df.get('is_limit_down', False) |
        df.get('is_suspended', False) |
        df.get('is_turnover_spike', False) |
        (df.get('price_logic_ok', True) == False) |
        (df.get('realized_vol_5d', 0) > 0.05)
    )
    df.loc[high_noise, 'noise_weight'] = 0.3

    medium_noise = (
        ~high_noise & (
            (df.get('realized_vol_5d', 0) > 0.03) |
            (df.get('atr_14', 0) > 0.03)
        )
    )
    df.loc[medium_noise, 'noise_weight'] = 0.7

    n_high = (df['noise_weight'] == 0.3).sum()
    n_medium = (df['noise_weight'] == 0.7).sum()
    n_low = (df['noise_weight'] == 1.0).sum()
    print(f"  高噪声: {n_high:,} ({n_high/len(df):.1%})")
    print(f"  中噪声: {n_medium:,} ({n_medium/len(df):.1%})")
    print(f"  低噪声: {n_low:,} ({n_low/len(df):.1%})")
    print(f"  耗时: {time.time()-t0:.1f}s")
    return df
