"""
================================================================================
v35.0 数据加载与清洗模块
================================================================================
职责: 原始数据加载、规则引擎清洗、基本面对齐
================================================================================
"""

import gc
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

from config_v35 import PATHS, PREPROCESS

warnings.filterwarnings('ignore')


# ============================================================================
# 工具函数
# ============================================================================

def parse_chinese_number(s):
    """解析中文数字(如'1.2亿')为浮点数"""
    if pd.isna(s) or s == '' or s == 'False' or s == '--' or s == '-':
        return np.nan
    s = str(s).strip()
    match = re.match(r'^([+-]?\d+\.?\d*)\s*(万亿|亿|万)$', s)
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        if unit == '万亿':
            return num * 1e12
        elif unit == '亿':
            return num * 1e8
        elif unit == '万':
            return num * 1e4
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def mad_clip(series, n_mad=3.0, window=252):
    """MAD滚动截断: 用滚动中位数和MAD替代均值标准差，对异常值更稳健"""
    rolling_median = series.rolling(window=window, min_periods=30).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window, min_periods=30).median()
    upper = rolling_median + n_mad * rolling_mad
    lower = rolling_median - n_mad * rolling_mad
    return series.clip(lower=lower, upper=upper)


def winsorize_series(series, lower_q=0.01, upper_q=0.99):
    """Winsorize缩尾处理: 1%-99%分位数截断"""
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def compute_exp_decay_weights(dates, halflife_years=2.0):
    """指数衰减样本权重: 近期数据权重更高"""
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    days_to_end = (max_date - dates).dt.days.values
    total_days = halflife_years * 252
    weights = 0.5 ** (days_to_end / total_days)
    weights = 0.5 + 0.5 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
    return weights


# ============================================================================
# 规则引擎: 数据清洗第一道防线
# ============================================================================

def validate_price_logic(df):
    """
    价格逻辑校验: 确保OHLC满足基本逻辑关系
    违反规则的行自动修正: high=max(ohlc), low=min(ohlc)
    """
    print("[规则引擎] 价格逻辑校验...")
    t0 = time.time()

    df['price_logic_ok'] = (
        (df['high'] >= df['open']) & (df['high'] >= df['close']) &
        (df['high'] >= df['low']) & (df['low'] <= df['open']) &
        (df['low'] <= df['close']) & (df['high'] >= df['low'])
    )

    bad = (~df['price_logic_ok']).sum()
    print(f"  价格逻辑异常: {bad:,} 条 ({bad/len(df):.4%})")

    if bad > 0:
        ohlc = df[['open', 'high', 'low', 'close']].values
        df.loc[~df['price_logic_ok'], 'high'] = np.nanmax(ohlc[~df['price_logic_ok']], axis=1)
        df.loc[~df['price_logic_ok'], 'low'] = np.nanmin(ohlc[~df['price_logic_ok']], axis=1)
        df['price_logic_ok'] = (
            (df['high'] >= df['open']) & (df['high'] >= df['close']) &
            (df['high'] >= df['low']) & (df['low'] <= df['open']) &
            (df['low'] <= df['close']) & (df['high'] >= df['low'])
        )
        still_bad = (~df['price_logic_ok']).sum()
        print(f"  修正后仍异常: {still_bad:,} 条")

    print(f"  完成 ({time.time()-t0:.1f}s)")
    return df


def validate_limit_bounds(df):
    """
    涨跌停区间校验: 根据股票代码推断涨跌停限制
    主板±10%, 科创板688xxx±20%, 北交所8/9xxxxx±30%
    """
    print("[规则引擎] 涨跌停区间校验...")
    t0 = time.time()

    df['asset_str'] = df['asset'].astype(str).str.zfill(6)
    df['limit_pct'] = 0.10
    df.loc[df['asset_str'].str.startswith('688'), 'limit_pct'] = 0.20
    df.loc[df['asset_str'].str.startswith('8'), 'limit_pct'] = 0.30
    df.loc[df['asset_str'].str.startswith('9'), 'limit_pct'] = 0.30

    df = df.sort_values(['asset', 'date'])
    df['prev_close'] = df.groupby('asset')['close'].shift(1)
    df['daily_change'] = (df['close'] - df['prev_close']) / df['prev_close'].abs().clip(lower=1e-8)
    df['within_limit'] = df['daily_change'].abs() <= (df['limit_pct'] + 0.01)
    df.loc[df['prev_close'].isna(), 'within_limit'] = True

    bad = (~df['within_limit']).sum()
    print(f"  超出涨跌停区间: {bad:,} 条 ({bad/len(df):.4%})")

    df.drop(columns=['asset_str', 'prev_close', 'daily_change', 'limit_pct'], inplace=True, errors='ignore')
    print(f"  完成 ({time.time()-t0:.1f}s)")
    return df


def detect_turnover_spike(df, spike_threshold=10.0, price_vol_threshold=0.05):
    """
    换手率突增检测: 标记可能存在对倒或杀猪盘行为的交易日
    规则: 单日量/5日均量 > 10倍 且 日内波幅 > 5%
    """
    print("[规则引擎] 换手率突增检测...")
    t0 = time.time()

    df = df.sort_values(['asset', 'date'])
    df['vol_ma5'] = df.groupby('asset')['volume'].rolling(5).mean().reset_index(level=0, drop=True)
    df['turnover_approx'] = df['volume'] / df['vol_ma5'].clip(lower=1e-8)
    df['intraday_vol'] = (df['high'] - df['low']) / df['close'].clip(lower=1e-8)

    df['is_turnover_spike'] = (
        (df['turnover_approx'] > spike_threshold) &
        (df['intraday_vol'] > price_vol_threshold)
    )

    spike_count = df['is_turnover_spike'].sum()
    print(f"  换手率突增异常: {spike_count:,} 条 ({spike_count/len(df):.4%})")

    df.drop(columns=['vol_ma5', 'turnover_approx', 'intraday_vol'], inplace=True, errors='ignore')
    print(f"  完成 ({time.time()-t0:.1f}s)")
    return df


# ============================================================================
# 停牌与特殊状态处理
# ============================================================================

def mark_special_states(df):
    """标记涨跌停、一字板状态"""
    # 修复: pctChg列可能不存在，使用计算方式替代
    if 'pctChg' in df.columns:
        pct = df['pctChg']
    else:
        # 从价格计算涨跌幅
        prev_close = df.groupby('asset')['close'].shift(1)
        pct = (df['close'] - prev_close) / prev_close.abs().clip(lower=1e-8)

    df['is_limit_up'] = (df['close'] == df['high']) & (pct > 0.095)
    df['is_limit_down'] = (df['close'] == df['low']) & (pct < -0.095)
    df['is_one_word'] = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])
    return df


def handle_suspension(df, max_days=20):
    """停牌处理: 标记停牌日，超长停牌日设为无效"""
    df.sort_values(['asset', 'date'], inplace=True)
    df['is_suspended'] = (df['volume'] == 0)
    df['susp_group'] = (df['is_suspended'] != df['is_suspended'].shift(1)).cumsum()
    df['susp_days'] = df.groupby(['asset', 'susp_group'])['is_suspended'].cumsum()
    df['susp_days'] = df['susp_days'] * df['is_suspended']
    df['valid'] = df['susp_days'] <= max_days
    return df


# ============================================================================
# 数据加载
# ============================================================================

def load_all_stock_codes():
    """从行业映射表加载全部股票代码"""
    df = pd.read_csv(PATHS['sector_path'])
    df['asset'] = df['asset'].astype(str)
    return df['asset'].unique().tolist()


def load_sector_mapping():
    """加载行业映射"""
    df = pd.read_csv(PATHS['sector_path'])
    df['asset'] = df['asset'].astype(str)
    return df[['asset', 'sector', 'sector_id']].set_index('asset')


def load_fundamental_data():
    """加载并清洗基本面数据(利润表+资产负债表+现金流量表)"""
    print("[基本面] 加载并清洗基本面数据...")
    t0 = time.time()

    benefit_usecols = ['股票代码', '报告期', '*归属于母公司所有者的净利润',
                       '*营业总收入', '*扣除非经常性损益后的净利润', '其中：营业成本']
    debt_usecols = ['股票代码', '报告期', '*所有者权益（或股东权益）合计', '*资产合计', '*负债合计']

    benefit = pd.read_csv(PATHS['benefit_path'], usecols=benefit_usecols)
    debt = pd.read_csv(PATHS['debt_path'], usecols=debt_usecols)

    benefit['asset'] = benefit['股票代码'].astype(str)
    debt['asset'] = debt['股票代码'].astype(str)

    # 解析数值
    benefit['归属净利润_num'] = benefit['*归属于母公司所有者的净利润'].apply(parse_chinese_number)
    benefit['营业总收入_num'] = benefit['*营业总收入'].apply(parse_chinese_number)
    benefit['扣非净利润_num'] = benefit['*扣除非经常性损益后的净利润'].apply(parse_chinese_number)
    benefit['营业成本_num'] = benefit['其中：营业成本'].apply(parse_chinese_number)

    debt['净资产_num'] = debt['*所有者权益（或股东权益）合计'].apply(parse_chinese_number)
    debt['总资产_num'] = debt['*资产合计'].apply(parse_chinese_number)
    debt['总负债_num'] = debt['*负债合计'].apply(parse_chinese_number)

    # 公告日期 = 报告期 + 90天（避免未来信息泄露）
    benefit['effective_date'] = pd.to_datetime(benefit['报告期']) + timedelta(days=90)
    debt['effective_date'] = pd.to_datetime(debt['报告期']) + timedelta(days=90)

    # 合并
    benefit_clean = benefit[['asset', 'effective_date', '归属净利润_num', '营业总收入_num',
                              '扣非净利润_num', '营业成本_num']].copy()
    benefit_clean = benefit_clean.dropna(subset=['归属净利润_num'])

    debt_clean = debt[['asset', 'effective_date', '净资产_num', '总资产_num', '总负债_num']].copy()
    debt_clean = debt_clean.dropna(subset=['净资产_num', '总资产_num', '总负债_num'])

    merged = pd.merge(benefit_clean, debt_clean, on=['asset', 'effective_date'], how='outer')
    merged = merged.sort_values(['asset', 'effective_date']).reset_index(drop=True)

    # 计算衍生指标
    merged['roe'] = merged['归属净利润_num'] / (merged['净资产_num'] + 1e-8)
    merged['debt_ratio'] = merged['总负债_num'] / (merged['总资产_num'] + 1e-8)
    merged['net_profit'] = merged['归属净利润_num']
    merged['total_revenue'] = merged.get('营业总收入_num', np.nan)
    merged['total_equity'] = merged['净资产_num']
    merged['operating_cost'] = merged.get('营业成本_num', np.nan)
    merged['total_assets'] = merged['总资产_num']

    # 加载经营现金流
    if os.path.exists(PATHS['cash_path']):
        try:
            cash_df = pd.read_csv(PATHS['cash_path'],
                                  usecols=['股票代码', '报告期', '*经营活动产生的现金流量净额'])
            cash_df['asset'] = cash_df['股票代码'].astype(str)
            cash_df['operating_cash_flow'] = cash_df['*经营活动产生的现金流量净额'].apply(parse_chinese_number)
            cash_df['effective_date'] = pd.to_datetime(cash_df['报告期']) + timedelta(days=90)
            cash_clean = cash_df[['asset', 'effective_date', 'operating_cash_flow']].dropna()
            cash_clean = cash_clean.sort_values(['asset', 'effective_date']).drop_duplicates(
                subset=['asset', 'effective_date'], keep='last'
            )
            merged = pd.merge(merged, cash_clean, on=['asset', 'effective_date'], how='outer')
        except Exception as e:
            print(f"  警告: 加载经营现金流失败: {e}")
            merged['operating_cash_flow'] = np.nan
    else:
        merged['operating_cash_flow'] = np.nan

    # 截断极端值
    merged['roe'] = merged['roe'].clip(-10, 10)
    merged['debt_ratio'] = merged['debt_ratio'].clip(0, 5)

    print(f"  基本面有效: {merged['asset'].nunique()} 只股票, {len(merged)} 条记录 ({time.time()-t0:.1f}s)")

    return_cols = ['asset', 'effective_date', 'roe', 'debt_ratio',
                   'net_profit', 'total_revenue', 'total_equity',
                   'operating_cash_flow', 'operating_cost', 'total_assets']
    return merged[[c for c in return_cols if c in merged.columns]]


def align_fundamental_to_dates(df, fundamental_df):
    """将基本面数据时间对齐到交易日(前向填充)"""
    print("[基本面] 时间对齐到交易日...")
    t0 = time.time()

    fundamental_df = fundamental_df.copy()
    fundamental_df['effective_date'] = pd.to_datetime(fundamental_df['effective_date'])
    fundamental_df['asset'] = fundamental_df['asset'].astype(str)

    align_cols = ['roe', 'debt_ratio']
    extra_cols = [c for c in ['net_profit', 'total_revenue', 'total_equity',
                               'operating_cash_flow', 'operating_cost', 'total_assets']
                  if c in fundamental_df.columns]
    all_align_cols = align_cols + extra_cols

    # 按asset构建查找字典
    fund_dict = {}
    for asset, grp in fundamental_df.groupby('asset'):
        fund_dict[asset] = grp.sort_values('effective_date')[['effective_date'] + all_align_cols].values

    df_dates = df[['asset', 'date']].drop_duplicates().copy()
    df_dates['asset'] = df_dates['asset'].astype(str)
    df_dates['date'] = pd.to_datetime(df_dates['date'])
    df_dates = df_dates.sort_values(['asset', 'date']).reset_index(drop=True)

    assets = df_dates['asset'].values
    dates = df_dates['date'].values.astype('datetime64[ns]')
    n = len(assets)

    result_arrays = {col: np.full(n, np.nan) for col in all_align_cols}

    for asset, fund_arr in fund_dict.items():
        asset_mask = assets == asset
        if not asset_mask.any():
            continue
        asset_dates = dates[asset_mask]
        fund_dates = fund_arr[:, 0].astype('datetime64[ns]')
        idx = np.searchsorted(fund_dates, asset_dates, side='right') - 1
        valid = idx >= 0
        if valid.any():
            valid_idx = idx[valid]
            asset_positions = np.where(asset_mask)[0][valid]
            for col_idx, col_name in enumerate(all_align_cols):
                result_arrays[col_name][asset_positions] = fund_arr[valid_idx, col_idx + 1]

    aligned = pd.DataFrame({'asset': assets, 'date': dates})
    for col in all_align_cols:
        aligned[col] = result_arrays[col]

    print(f"  对齐完成: {len(aligned):,} 条记录 ({time.time()-t0:.1f}s)")
    for col in extra_cols:
        valid_count = aligned[col].notna().sum()
        print(f"    {col}: {valid_count:,} 条有效值 ({valid_count/len(aligned):.2%})")
    return aligned


def load_stock_data(codes, start_date, end_date):
    """分块加载原始行情数据"""
    print(f"[数据加载] 加载原始行情: {start_date} ~ {end_date}")
    t0 = time.time()
    pre_start = (pd.to_datetime(start_date) - timedelta(days=200)).strftime('%Y-%m-%d')
    codes_set = set(codes)

    chunks = []
    for chunk in pd.read_csv(PATHS['raw_data_path'], chunksize=500000,
                              usecols=['asset', 'date', 'open', 'close', 'high', 'low', 'volume', 'vwap', 'pctChg']):
        chunk['asset'] = chunk['asset'].astype(str)
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk = chunk[chunk['asset'].isin(codes_set)]
        chunk = chunk[(chunk['date'] >= pre_start) & (chunk['date'] <= end_date)]
        if len(chunk) > 0:
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(['asset', 'date']).reset_index(drop=True)

    # 过滤交易天数不足的股票
    trading_days = df.groupby('asset')['date'].count()
    valid_assets = trading_days[trading_days >= PREPROCESS['min_trading_days']].index.tolist()
    df = df[df['asset'].isin(valid_assets)].reset_index(drop=True)

    print(f"  数据加载完成: {len(df):,} 条, {df['asset'].nunique()} 只股票 ({time.time()-t0:.1f}s)")
    return df


def filter_by_training_period(df, train_start, train_end, min_days=200):
    """幸存者偏差修正: 只保留训练期有足够数据的股票"""
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    train_mask = (df['date'] >= train_start_dt) & (df['date'] <= train_end_dt)
    trading_days_in_train = df[train_mask].groupby('asset')['date'].count()
    valid_assets = trading_days_in_train[trading_days_in_train >= min_days].index
    original_count = df['asset'].nunique()
    print(f"  幸存者偏差修正: {original_count} -> {len(valid_assets)} 只")
    return df[df['asset'].isin(valid_assets)].reset_index(drop=True)
