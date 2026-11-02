"""
================================================================================
v35.0 IC特征筛选模块
================================================================================
职责: 截面IC计算与特征筛选
修复: IC计算已使用NaN安全的逐日截面方式（原代码已正确）
优化:
  - 临时CSV保存中间数据，避免内存溢出
  - 只加载需要的列，减少内存占用
  - 分批读取CSV，避免一次性全量加载
================================================================================
"""

import gc
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch

from config_v35 import PATHS, IC_FILTER

warnings.filterwarnings('ignore')

# GPU加速配置
USE_GPU = torch.cuda.is_available()


# ============================================================================
# GPU加速相关系数计算
# ============================================================================

def torch_corrcoef(x, y):
    """GPU加速相关系数计算"""
    device = 'cuda'
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # 有效数据
    mask = ~(torch.isnan(x_tensor) | torch.isnan(y_tensor))
    if mask.sum() < 20:
        return np.nan

    x_valid = x_tensor[mask]
    y_valid = y_tensor[mask]

    # 中心化
    x_mean = x_valid - x_valid.mean()
    y_mean = y_valid - y_valid.mean()

    # 计算相关系数
    numerator = (x_mean * y_mean).sum()
    denominator = torch.sqrt((x_mean ** 2).sum() * (y_mean ** 2).sum())

    if denominator < 1e-8:
        return np.nan

    return (numerator / denominator).cpu().numpy()


def compute_ic_batch(train_period, all_feature_cols, label_col, unique_dates, date_indices):
    """截面IC计算(NaN安全版本): 逐日计算截面相关系数"""
    valid_cols = [c for c in all_feature_cols if c in train_period.columns]
    if len(valid_cols) == 0:
        return []

    n_dates = len(unique_dates)
    n_features = len(valid_cols)
    daily_ics = np.zeros((n_features, n_dates), dtype=np.float32)

    for d in range(n_dates):
        mask = date_indices == d
        grp_size = mask.sum()
        if grp_size < 20:
            continue

        grp_df = train_period.loc[mask, valid_cols + [label_col]]
        label_d = grp_df[label_col].values.astype(np.float32)
        label_valid = ~np.isnan(label_d)
        n_valid_label = label_valid.sum()
        if n_valid_label < 20:
            continue

        label_v_full = label_d[label_valid]
        label_std = label_v_full.std()
        if label_std < 1e-10:
            continue

        for j, col in enumerate(valid_cols):
            feat_d = grp_df[col].values.astype(np.float32)
            valid_mask = label_valid & ~np.isnan(feat_d)
            n_valid = valid_mask.sum()
            if n_valid < 20:
                daily_ics[j, d] = 0
                continue

            if USE_GPU:
                # GPU加速计算相关系数
                corr = torch_corrcoef(feat_d, label_d)
                if np.isnan(corr):
                    daily_ics[j, d] = 0
                else:
                    daily_ics[j, d] = corr
            else:
                # 原numpy实现
                feat_v = feat_d[valid_mask]
                label_v = label_d[valid_mask]
                feat_mean = feat_v.mean()
                label_mean = label_v.mean()
                cov = ((feat_v - feat_mean) * (label_v - label_mean)).mean()
                feat_std = feat_v.std()
                if feat_std < 1e-10:
                    daily_ics[j, d] = 0
                else:
                    daily_ics[j, d] = cov / (feat_std * label_std)

    feature_ic_stats = []
    for i, col in enumerate(valid_cols):
        valid_ics = daily_ics[i, :][np.abs(daily_ics[i, :]) > 0]
        if len(valid_ics) > 10:
            ic_mean = np.mean(valid_ics)
            ic_std = np.std(valid_ics)
            ic_ir = ic_mean / (ic_std + 1e-10)
        else:
            ic_mean = 0
            ic_ir = 0

        selected = abs(ic_mean) >= IC_FILTER['min_ic'] or abs(ic_ir) >= IC_FILTER['min_icir']
        feature_ic_stats.append({
            'feature': col,
            'ic_mean': ic_mean,
            'ic_ir': ic_ir,
            'selected': selected
        })

    for col in all_feature_cols:
        if col not in valid_cols:
            feature_ic_stats.append({'feature': col, 'ic_mean': 0, 'ic_ir': 0, 'selected': False})

    return feature_ic_stats


def ic_feature_selection(df, all_feature_cols, label_col):
    """IC特征筛选: 只在训练期计算，防止数据泄露，支持GPU加速"""
    gpu_msg = " (GPU加速)" if USE_GPU else ""
    print(f"[IC筛选] v35.0 IC/IC_IR特征筛选{gpu_msg}...")
    if USE_GPU:
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
    print("-" * 50)

    # 保存到临时CSV释放内存
    tmp_csv = os.path.join(PATHS['tmp_memmap_dir'], 'factor_data_ic.csv')
    print(f"  [内存优化] 保存到临时CSV...")
    df.to_csv(tmp_csv, index=False)

    factor_data_available_columns = list(df.columns)
    del df
    gc.collect()

    # 加载IC计算需要的列
    ic_cols = all_feature_cols + [label_col, 'date']
    ic_cols = [c for c in ic_cols if c in factor_data_available_columns]
    print(f"  [内存优化] 加载IC计算所需{len(ic_cols)}列...")
    train_period = pd.read_csv(tmp_csv, usecols=ic_cols, parse_dates=['date'])

    # 只用训练期数据（防止数据泄露）
    ic_end_date = pd.Timestamp('2025-01-01') - pd.Timedelta(days=1)
    train_period = train_period.loc[
        (train_period['date'] >= pd.Timestamp('2020-01-01')) &
        (train_period['date'] <= ic_end_date) &
        train_period[label_col].notna()
    ]
    gc.collect()

    # 日期编码
    date_codes = train_period['date'].astype('int64').values
    unique_dates, date_indices = np.unique(date_codes, return_inverse=True)

    print(f"  NaN安全IC计算: {len(unique_dates)} 个交易日, {len(all_feature_cols)} 个特征")
    feature_ic_stats = compute_ic_batch(train_period, all_feature_cols, label_col, unique_dates, date_indices)

    # 排序打印
    feature_ic_stats.sort(key=lambda x: abs(x['ic_mean']), reverse=True)
    print("  特征IC排名 (Top 30):")
    print(f"  {'特征名':<25} {'IC均值':>8} {'IC_IR':>8} {'选中':>4}")
    print("  " + "-" * 50)
    for s in feature_ic_stats[:30]:
        print(f"  {s['feature']:<25} {s['ic_mean']:>8.4f} {s['ic_ir']:>8.4f} {'Y' if s['selected'] else 'N':>4}")

    # 选中的特征
    selected_features = [s['feature'] for s in feature_ic_stats if s['selected']]
    if len(selected_features) < IC_FILTER['min_features']:
        selected_features = [s['feature'] for s in feature_ic_stats[:IC_FILTER['min_features']]]
        print(f"  警告: 选中特征不足{IC_FILTER['min_features']}个，补充至{IC_FILTER['min_features']}个")
    if len(selected_features) > IC_FILTER['max_features']:
        selected_features = selected_features[:IC_FILTER['max_features']]
        print(f"  限制最大特征数: {len(selected_features)} 个")

    print(f"\n  特征筛选: {len(all_feature_cols)} -> {len(selected_features)} 个")

    del train_period
    gc.collect()

    # 从临时CSV重新加载需要的列
    needed_after_ic = list(set(selected_features + ['asset', 'date', label_col, 'sector', 'noise_weight', 'future_return']))
    with open(tmp_csv, 'r', encoding='utf-8') as f:
        import csv
        csv_cols = next(csv.reader(f))
    needed_after_ic = [c for c in needed_after_ic if c in csv_cols]

    chunk_size = 500000
    chunks = []
    for chunk in pd.read_csv(tmp_csv, usecols=needed_after_ic, parse_dates=['date'], chunksize=chunk_size):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    try:
        os.remove(tmp_csv)
    except:
        pass

    print(f"  factor_data重新加载完成: {len(df):,}行 x {len(df.columns)}列")
    return df, selected_features, feature_ic_stats
