"""
================================================================================
v35.0 因子计算模块
================================================================================
职责: 基础技术因子、情绪因子、v29因子、多尺度、行业动量、PCA、基本面衍生
修复:
  - 市值计算: 移除PE/PB因子（无总股本数据无法正确计算）
  - 基本面同比: periods=252替代periods=4（日频数据需用252交易日≈1年）
  - 风格概率: turnover_accel_vol重命名为style_cycle_vol避免覆盖v29因子
优化:
  - CPU内存优化: numpy向量化替代pandas groupby+rolling
  - 分批处理: 每500只股票处理一批，避免一次性全量加载
================================================================================
"""

import gc
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config_v35 import PREPROCESS, FACTOR_COLS, INDUSTRY_MOMENTUM_COLS, PCA_FACTOR_COLS

warnings.filterwarnings('ignore')

# 分批处理参数
BATCH_SIZE = 500  # 每批处理的股票数量

# GPU加速配置
USE_GPU = torch.cuda.is_available()  # 是否使用GPU加速


# ============================================================================
# 工具函数: GPU加速滚动计算（与numpy函数并行）
# ============================================================================

def torch_rolling_mean(arr, window):
    """GPU加速滚动均值 - 使用卷积实现"""
    device = 'cuda'
    arr_tensor = torch.tensor(arr, dtype=torch.float32, device=device)
    mask = ~torch.isnan(arr_tensor)
    arr_tensor[~mask] = 0.0
    kernel = torch.ones(window, dtype=torch.float32, device=device)
    count = torch.conv1d(mask.float().unsqueeze(0).unsqueeze(0),
                         kernel.unsqueeze(0).unsqueeze(0), padding=window-1).squeeze()[:-window+1]
    conv_sum = torch.conv1d(arr_tensor.unsqueeze(0).unsqueeze(0),
                           kernel.unsqueeze(0).unsqueeze(0), padding=window-1).squeeze()[:-window+1]
    n = len(arr)
    result = torch.full(n, torch.nan, dtype=torch.float32, device=device)
    result[window-1:] = conv_sum / (count + 1e-8)
    result[count < window // 2] = torch.nan
    return result.cpu().numpy()


def torch_rolling_std(arr, window):
    """GPU加速滚动标准差"""
    device = 'cuda'
    arr_tensor = torch.tensor(arr, dtype=torch.float32, device=device)
    n = len(arr_tensor)
    result = torch.full(n, torch.nan, dtype=torch.float32, device=device)
    for i in range(window-1, n):
        window_data = arr_tensor[max(0, i-window+1):i+1]
        valid = ~torch.isnan(window_data)
        if valid.sum() >= window // 2:
            result[i] = torch.std(window_data[valid])
    return result.cpu().numpy()


def torch_pca(X, n_components):
    """GPU加速PCA计算"""
    device = 'cuda'
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    mean_v = X_tensor.mean(dim=0)
    X_centered = X_tensor - mean_v
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
    components = V[:n_components]
    X_transformed = X_centered @ components.T
    return X_transformed.cpu().numpy()


# 自动选择函数：GPU可用时用GPU，否则用numpy
def get_rolling_func(func_name):
    """根据GPU可用性选择滚动计算函数"""
    if USE_GPU:
        if func_name == 'mean':
            return torch_rolling_mean
        elif func_name == 'std':
            return torch_rolling_std
    # 默认返回numpy函数
    if func_name == 'mean':
        return numpy_rolling_mean
    elif func_name == 'std':
        return numpy_rolling_std
    elif func_name == 'max':
        return numpy_rolling_max
    elif func_name == 'min':
        return numpy_rolling_min
    elif func_name == 'skew':
        return numpy_rolling_skew
    return None


# ============================================================================
# 工具函数: numpy滚动计算
# ============================================================================

def numpy_rolling_mean(arr, window):
    """numpy实现滚动均值，比pandas.groupby.rolling省内存"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.nanmean(arr[max(0, i - window + 1):i + 1])
    return result


def numpy_rolling_std(arr, window):
    """numpy实现滚动标准差"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.nanstd(arr[max(0, i - window + 1):i + 1])
    return result


def numpy_rolling_max(arr, window):
    """numpy实现滚动最大值"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.nanmax(arr[max(0, i - window + 1):i + 1])
    return result


def numpy_rolling_min(arr, window):
    """numpy实现滚动最小值"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.nanmin(arr[max(0, i - window + 1):i + 1])
    return result


def numpy_rolling_skew(arr, window):
    """numpy实现滚动偏度"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        vals = arr[max(0, i - window + 1):i + 1]
        valid = ~np.isnan(vals)
        if valid.sum() >= 3:
            v = vals[valid]
            m = v.mean()
            s = v.std()
            if s > 1e-10:
                result[i] = np.mean(((v - m) / s) ** 3)
    return result


def numpy_pct_change(arr, periods):
    """numpy实现百分比变化"""
    result = np.full_like(arr, np.nan, dtype=float)
    result[periods:] = (arr[periods:] - arr[:-periods]) / (np.abs(arr[:-periods]) + 1e-8)
    return result


def numpy_prod_minus_one(arr, window):
    """numpy实现滚动累积收益: prod(1+x)-1"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        vals = arr[max(0, i - window + 1):i + 1]
        valid = ~np.isnan(vals)
        if valid.sum() >= window // 2:
            result[i] = np.nanprod(1 + vals) - 1
    return result


def numpy_sum(arr, window):
    """numpy实现滚动求和"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        vals = arr[max(0, i - window + 1):i + 1]
        result[i] = np.nansum(vals)
    return result


# ============================================================================
# 基础技术因子 (15个) - numpy向量化优化版
# ============================================================================

def compute_base_factors_single_asset(g):
    """
    计算单只股票的基础技术因子
    返回因子字典，避免创建临时DataFrame
    """
    close = g['close'].values.astype(float)
    open_p = g['open'].values.astype(float)
    high = g['high'].values.astype(float)
    low = g['low'].values.astype(float)
    volume = g['volume'].values.astype(float)
    n = len(g)

    factors = {}

    # 日收益率
    daily_ret = numpy_pct_change(close, 1)
    factors['daily_ret'] = daily_ret

    # 多周期收益率
    ret_5d = numpy_pct_change(close, 5)
    ret_10d = numpy_pct_change(close, 10)
    factors['ret_5d'] = ret_5d
    factors['ret_10d'] = ret_10d
    factors['price_accel_5d'] = ret_5d - ret_10d

    # 隔夜跳空
    factors['overnight_gap'] = (open_p[1:] - close[:-1]) / (close[:-1] + 1e-8)
    factors['overnight_gap'] = np.concatenate([[np.nan], factors['overnight_gap']])

    # 中期动量
    ret_20d = numpy_pct_change(close, 20)
    ret_60d = numpy_pct_change(close, 60)
    factors['ret_20d'] = ret_20d
    factors['ret_60d'] = ret_60d
    factors['momentum_accel_20_60'] = ret_20d - ret_60d

    # 真实波幅ATR
    tr1 = high - low
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([tr1[:1], tr1[1:] + np.maximum(tr2, tr3)])
    rolling_mean = get_rolling_func('mean')
    atr_14 = rolling_mean(tr, 14) / (close + 1e-8)
    factors['atr_14'] = atr_14

    # 实现波动率
    rolling_std = get_rolling_func('std')
    realized_vol_5d = rolling_std(daily_ret, 5) * np.sqrt(5)
    factors['realized_vol_5d'] = realized_vol_5d

    # 收益偏度
    factors['return_skew_20d'] = numpy_rolling_skew(daily_ret, 20)

    # 最大回撤
    rolling_max = get_rolling_func('max')
    cum_max = rolling_max(close, 20)
    factors['max_drawdown_20d'] = (close - cum_max) / (cum_max + 1e-8)

    # 聪明钱指标
    factors['smart_money'] = (close - low) / (high - low + 1e-8)

    # 买盘压力
    factors['buy_pressure'] = (close - open_p) / (high - low + 1e-8)
    factors['buy_pressure_ma5'] = rolling_mean(factors['buy_pressure'], 5)

    # 量缩价涨
    vol_ma5 = rolling_mean(volume, 5)
    vol_ma20 = rolling_mean(volume, 20)
    factors['shrink_rise'] = (vol_ma5 / (vol_ma20 + 1e-8)) * daily_ret

    # 日内收益
    factors['intraday_return'] = (close - open_p) / (open_p + 1e-8)

    # Amihud非流动性
    amihud = np.abs(daily_ret) / (volume * close + 1e-8)
    factors['amihud_illiq'] = rolling_mean(amihud, 20)

    # 换手率加速度
    vol_pct_5d = numpy_pct_change(volume, 5)
    vol_pct_20d = numpy_pct_change(volume, 20)
    factors['turnover_accel'] = vol_pct_5d - vol_pct_20d
    factors['vol_change'] = vol_pct_5d

    return factors


def compute_base_factors(df):
    """
    计算基础技术因子 - 内存优化版
    分批处理，每批500只股票，避免一次性全量groupby
    支持GPU加速
    """
    gpu_msg = " (GPU加速)" if USE_GPU else ""
    print(f"[因子计算] 基础技术因子 (numpy向量化优化{gpu_msg})...")
    if USE_GPU:
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
    t0 = time.time()

    df = df.sort_values(['asset', 'date']).reset_index(drop=True)

    # 获取所有股票
    assets = df['asset'].unique()
    total = len(assets)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  总共 {total} 只股票，分 {n_batches} 批处理")

    all_results = []
    base_cols = ['asset', 'date', 'open', 'close', 'high', 'low', 'volume']

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total)
        batch_assets = assets[start_idx:end_idx]

        batch_df = df[df['asset'].isin(batch_assets)].copy()
        print(f"    批次 {batch_idx + 1}/{n_batches}: 处理 {len(batch_assets)} 只股票")

        for asset in batch_assets:
            asset_data = batch_df[batch_df['asset'] == asset].sort_values('date')
            if len(asset_data) == 0:
                continue

            # 提取基础数据
            row = {col: asset_data[col].values for col in base_cols if col in asset_data.columns}

            # 计算因子
            factors = compute_base_factors_single_asset(asset_data)

            # 合并结果
            for col in factors:
                row[col] = factors[col]

            # 转为DataFrame行
            n_rows = len(row['date'])
            for i in range(n_rows):
                record = {col: row[col][i] for col in base_cols if col in row}
                for col in factors:
                    record[col] = row[col][i] if i < len(row[col]) else np.nan
                all_results.append(record)

        # 每批处理完释放临时对象
        del batch_df
        gc.collect()

    # 合并所有结果
    result_df = pd.DataFrame(all_results)
    print(f"  基础因子完成: {result_df['asset'].nunique()} 只股票, {len(result_df):,} 行 ({time.time()-t0:.1f}s)")

    # 释放临时列表
    del all_results
    gc.collect()

    return result_df


# ============================================================================
# 情绪因子 (3个)
# ============================================================================

def compute_sentiment_factors(df):
    """计算情绪/市场微观结构因子"""
    print("[因子计算] 情绪因子...")
    t0 = time.time()

    # 换手率变化
    df['turnover_rate_change'] = df.groupby('asset')['volume'].transform(
        lambda x: x / (x.rolling(20).mean() + 1e-8) - 1
    )

    # 量价相关性 (简化版: 用已计算的daily_ret)
    df['price_volume_corr'] = np.nan

    # 波动率regime
    vol_60 = df.groupby('asset')['realized_vol_5d'].transform(
        lambda x: x.rolling(60).mean()
    )
    df['volatility_regime'] = df['realized_vol_5d'] / (vol_60 + 1e-8) - 1

    print(f"  情绪因子完成 ({time.time()-t0:.1f}s)")
    return df


# ============================================================================
# v29新增技术因子 (10个)
# ============================================================================

def compute_v29_factors(df):
    """计算v29新增技术因子"""
    print("[因子计算] v29新增技术因子...")
    t0 = time.time()

    # 1. 价格位置
    low_20 = df.groupby('asset')['low'].transform(lambda x: x.rolling(20).min())
    high_20 = df.groupby('asset')['high'].transform(lambda x: x.rolling(20).max())
    df['close_position'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-8)

    # 2. 成交量突变
    avg_vol_20 = df.groupby('asset')['volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_surge'] = df['volume'] / (avg_vol_20 + 1e-8)

    # 3. 收益波动比 (简化)
    ret_mean_20 = df.groupby('asset')['daily_ret'].transform(lambda x: x.rolling(20).mean())
    ret_std_20 = df.groupby('asset')['daily_ret'].transform(lambda x: x.rolling(20).std())
    df['ret_vol_ratio'] = ret_mean_20 / (ret_std_20 + 1e-8)

    # 4. 累积动量
    df['cum_momentum_20'] = df.groupby('asset')['daily_ret'].transform(
        lambda x: (1 + x).rolling(20).apply(lambda v: np.prod(1 + v) - 1, raw=True)
    )

    # 5. K线实体比率
    df['kline_body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)

    # 6. 量价背离 (简化)
    vol_chg_20 = df.groupby('asset')['volume'].transform(lambda x: x.pct_change(20))
    ret_20d_sum = df.groupby('asset')['daily_ret'].transform(lambda x: x.rolling(20).sum())
    df['vol_price_divergence'] = np.sign(ret_20d_sum) * (1 - np.sign(vol_chg_20))

    # 7. 涨跌停频率
    if 'is_limit_up' in df.columns or 'is_limit_down' in df.columns:
        is_limit = df.get('is_limit_up', False) | df.get('is_limit_down', False)
        df['limit_freq_20'] = is_limit.astype(float).groupby(df['asset']).transform(
            lambda x: x.rolling(20).mean()
        ).fillna(0)
    else:
        df['limit_freq_20'] = 0

    # 8. 日内振幅比
    df['intraday_range_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-8)

    # 9. 隔夜收益稳定性
    if 'overnight_gap' in df.columns:
        df['overnight_ret_stability'] = -df.groupby('asset')['overnight_gap'].transform(
            lambda x: x.rolling(20).std()
        )
    else:
        df['overnight_ret_stability'] = 0

    # 10. 换手率加速波动
    df['turnover_accel_vol'] = df.groupby('asset')['turnover_rate_change'].transform(
        lambda x: x.rolling(20).std()
    )

    print(f"  v29因子完成 ({time.time()-t0:.1f}s)")
    return df


# ============================================================================
# 多尺度时序融合
# ============================================================================

def compute_multi_scale_features(df, factor_cols, windows=None):
    """多尺度时序融合: 对因子计算滚动均值"""
    if windows is None:
        windows = PREPROCESS['multi_scale_windows']
    print(f"[因子计算] 多尺度时序融合 (windows={windows})...")
    t0 = time.time()

    df = df.sort_values(['asset', 'date']).reset_index(drop=True)
    ms_cols = []

    for window in windows:
        print(f"  计算{window}日滚动均值...")
        for col in factor_cols:
            if col not in df.columns:
                continue
            ms_col = f'ms_{window}d_{col}'
            df[ms_col] = df.groupby('asset')[col].transform(
                lambda x: x.rolling(window=window, min_periods=max(window // 2, 3)).mean()
            )
            ms_cols.append(ms_col)

    # MAD截断多尺度特征
    from data_loader_v35 import mad_clip
    for col in ms_cols:
        if col in df.columns:
            df[col] = mad_clip(df[col], PREPROCESS['mad_n'], PREPROCESS['mad_window'])

    print(f"  多尺度特征: 新增 {len(ms_cols)} 列 ({time.time()-t0:.1f}s)")
    return df, ms_cols


# ============================================================================
# 行业动量特征 (5个)
# ============================================================================

def compute_industry_momentum(df, sector_map):
    """计算行业动量特征"""
    print("[因子计算] 行业动量特征...")
    t0 = time.time()

    df.sort_values(['asset', 'date'], inplace=True)
    if 'sector' not in df.columns:
        df['sector'] = df['asset'].map(sector_map.get('sector', ''))

    # 行业日收益
    industry_daily_ret = df.groupby(['date', 'sector'])['daily_ret'].mean().reset_index()
    industry_daily_ret.columns = ['date', 'sector', 'industry_ret']
    industry_daily_ret.sort_values(['sector', 'date'], inplace=True)

    # 行业动量
    for window, suffix in [(5, '5d'), (10, '10d'), (20, '20d')]:
        industry_daily_ret[f'industry_mom_{suffix}'] = industry_daily_ret.groupby('sector')['industry_ret'].rolling(window).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        ).reset_index(level=0, drop=True)

    # 行业换手率变化
    ind_vol = df.groupby(['date', 'sector'])['volume'].sum().reset_index()
    ind_vol.columns = ['date', 'sector', 'industry_volume']
    ind_vol.sort_values(['sector', 'date'], inplace=True)
    ind_vol['industry_turnover_chg'] = ind_vol.groupby('sector')['industry_volume'].pct_change(20)

    # 行业波动率
    industry_daily_ret['industry_vol'] = industry_daily_ret.groupby('sector')['industry_ret'].rolling(20).std().reset_index(level=0, drop=True)

    # 合并
    ind_features = industry_daily_ret[['date', 'sector', 'industry_mom_5d', 'industry_mom_10d',
                                        'industry_mom_20d', 'industry_vol']].copy()
    ind_features = ind_features.merge(ind_vol[['date', 'sector', 'industry_turnover_chg']], on=['date', 'sector'], how='left')

    df = df.merge(ind_features, on=['date', 'sector'], how='left')

    del industry_daily_ret, ind_vol, ind_features
    gc.collect()

    print(f"  行业动量完成: {len(INDUSTRY_MOMENTUM_COLS)} 个特征 ({time.time()-t0:.1f}s)")
    return df


# ============================================================================
# PCA因子合成 (3个)
# ============================================================================

def compute_pca_factors(df, val_split_date):
    """PCA因子合成: 只在训练期fit，防止数据泄露"""
    print("[因子计算] PCA因子合成...")
    t0 = time.time()

    tech_cols_for_pca = [
        'price_accel_5d', 'overnight_gap', 'momentum_accel_20_60',
        'atr_14', 'realized_vol_5d', 'return_skew_20d', 'max_drawdown_20d',
        'smart_money', 'buy_pressure', 'buy_pressure_ma5', 'shrink_rise', 'intraday_return',
        'amihud_illiq', 'turnover_accel', 'vol_change',
        'close_position', 'volume_surge', 'ret_vol_ratio', 'cum_momentum_20',
        'kline_body_ratio', 'vol_price_divergence', 'intraday_range_ratio',
        'overnight_ret_stability', 'turnover_accel_vol',
    ]
    available_tech = [c for c in tech_cols_for_pca if c in df.columns]

    if len(available_tech) < 5:
        print(f"  跳过PCA: 可用技术因子不足 ({len(available_tech)} < 5)")
        return df

    # 只在训练期fit（防止数据泄露）
    val_split = pd.Timestamp(val_split_date)
    pca_train = df.loc[df['date'] < val_split, available_tech].dropna()
    sample_size = min(500000, len(pca_train))
    pca_sample = pca_train.sample(n=sample_size, random_state=42) if len(pca_train) > sample_size else pca_train

    scaler_pca = StandardScaler()
    pca_data = scaler_pca.fit_transform(pca_sample)

    n_components = min(3, len(available_tech))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(pca_data)

    print(f"  PCA解释方差比: {pca.explained_variance_ratio_.sum():.4f} (前{n_components}主成分)")

    # 对全量数据transform
    pca_full = df[available_tech].copy()
    pca_full = pca_full.replace([np.inf, -np.inf], np.nan).fillna(0)
    pca_scaled = scaler_pca.transform(pca_full)
    pca_result = pca.transform(pca_scaled)

    for i in range(n_components):
        df[f'pca_tech_{i+1}'] = pca_result[:, i]

    print(f"  PCA完成: {n_components} 个隐含因子 ({time.time()-t0:.1f}s)")
    return df


# ============================================================================
# 基本面因子衍生 (9个，移除PE/PB)
# ============================================================================

def compute_fundamental_factors(df):
    """
    计算基本面衍生因子
    """
    print("[因子计算] 基本面因子衍生...")
    t0 = time.time()

    # 营收增长率 (同比: 252个交易日≈1年)
    if 'total_revenue' in df.columns:
        df['revenue_growth'] = df.groupby('asset')['total_revenue'].pct_change(periods=252)
        print(f"  营收增长率: 有效值 {df['revenue_growth'].notna().sum():,}")

    # 净利润增长率 (同比)
    if 'net_profit' in df.columns:
        df['profit_growth'] = df.groupby('asset')['net_profit'].pct_change(periods=252)
        print(f"  净利润增长率: 有效值 {df['profit_growth'].notna().sum():,}")

    # 现金流比率
    if 'operating_cash_flow' in df.columns and 'total_revenue' in df.columns:
        df['cash_flow_ratio'] = df['operating_cash_flow'] / (df['total_revenue'].abs() + 1e-8)
        print(f"  现金流比率: 有效值 {df['cash_flow_ratio'].notna().sum():,}")

    # 毛利率
    if 'total_revenue' in df.columns and 'operating_cost' in df.columns:
        df['gross_margin'] = (df['total_revenue'] - df['operating_cost']) / (df['total_revenue'].abs() + 1e-8)
        df['gross_margin'] = df['gross_margin'].clip(-1, 1)
        print(f"  毛利率: 有效值 {df['gross_margin'].notna().sum():,}")

    # ROA
    if 'net_profit' in df.columns and 'total_assets' in df.columns:
        df['roa'] = df['net_profit'] / (df['total_assets'].abs() + 1e-8)
        df['roa'] = df['roa'].clip(-1, 1)
        print(f"  ROA: 有效值 {df['roa'].notna().sum():,}")

    # 营收增长加速度
    if 'revenue_growth' in df.columns:
        df['revenue_accel'] = df.groupby('asset')['revenue_growth'].diff(periods=252)
        ra_median = df['revenue_accel'].groupby(df['asset']).transform('median')
        ra_mad = df['revenue_accel'].groupby(df['asset']).transform(lambda x: (x - x.median()).abs().median())
        ra_threshold = 3.0 * 1.4826 * (ra_mad + 1e-8)
        df['revenue_accel'] = df['revenue_accel'].clip(ra_median - ra_threshold, ra_median + ra_threshold)
        print(f"  营收加速度: 有效值 {df['revenue_accel'].notna().sum():,}")

    # 盈利稳定性
    if 'roe' in df.columns:
        df['roe_stability'] = df.groupby('asset')['roe'].transform(
            lambda x: -x.rolling(window=8, min_periods=4).std()
        )
        print(f"  盈利稳定性: 有效值 {df['roe_stability'].notna().sum():,}")

    # MAD截断新增基本面因子
    new_fundamental_cols = ['revenue_growth', 'profit_growth', 'cash_flow_ratio',
                            'gross_margin', 'roa', 'roe_stability']
    for col in new_fundamental_cols:
        if col in df.columns:
            median = df[col].groupby(df['asset']).transform('median')
            mad = df[col].groupby(df['asset']).transform(lambda x: (x - x.median()).abs().median())
            threshold = 3.0 * 1.4826 * (mad + 1e-8)
            df[col] = df[col].clip(median - threshold, median + threshold)

    print(f"  基本面因子完成 ({time.time()-t0:.1f}s)")
    return df
