"""
================================================================================
v35.0 数据预处理主入口
================================================================================
调用各子模块完成14阶段预处理流程
模块: config_v35, data_loader_v35, factor_v35, label_v35, ic_filter_v35
================================================================================
"""

import gc
import json
import os
import time
from datetime import datetime

import pandas as pd

from config_v35 import PATHS, TIME, PREPROCESS, IC_FILTER, FACTOR_COLS, INDUSTRY_MOMENTUM_COLS, PCA_FACTOR_COLS, STYLE_COLS
from data_loader_v35 import (
    load_all_stock_codes, load_sector_mapping, load_fundamental_data,
    load_stock_data, filter_by_training_period, align_fundamental_to_dates,
    validate_price_logic, validate_limit_bounds, detect_turnover_spike,
    handle_suspension, mark_special_states, winsorize_series, mad_clip,
    compute_exp_decay_weights
)
from factor_v35 import (
    compute_base_factors, compute_sentiment_factors, compute_v29_factors,
    compute_multi_scale_features, compute_industry_momentum,
    compute_pca_factors, compute_fundamental_factors
)
from label_v35 import (
    compute_industry_neutral_labels, compute_triple_residual_labels,
    compute_style_probabilities, compute_noise_weights
)
from ic_filter_v35 import ic_feature_selection


def main(force_regenerate=False):
    t_total = time.time()
    print("=" * 70)
    print("v35.0 数据预处理脚本（模块化版本）")
    print("=" * 70)
    print(f"训练期: {TIME['train_start']} ~ {TIME['train_end']}")
    print(f"验证分割: {TIME['val_split_date']}")
    print(f"标签类型: {PREPROCESS['label_type']}")
    print(f"IC筛选阈值: |IC|>={IC_FILTER['min_ic']} 或 |IC_IR|>={IC_FILTER['min_icir']}")
    print(f"最大特征数: {IC_FILTER['max_features']}")
    print("=" * 70)
    
    # 检查预处理文件是否已存在
    factor_data_path = os.path.join(PATHS['preprocess_dir'], 'factor_data.csv')
    selected_features_path = os.path.join(PATHS['preprocess_dir'], 'selected_features.json')
    
    if not force_regenerate and os.path.exists(factor_data_path) and os.path.exists(selected_features_path):
        print("\n[检查] 预处理文件已存在!")
        print(f"  factor_data.csv: {factor_data_path}")
        print(f"  selected_features.json: {selected_features_path}")
        print("\n  如需重新生成，请运行:")
        print("  python preprocess_v35.py --force")
        print("=" * 70)
        return

    # Phase 1: 加载数据
    print("\n[Phase 1] 数据加载")
    print("-" * 50)
    all_codes = load_all_stock_codes()
    sector_map = load_sector_mapping()
    print(f"总股票数: {len(all_codes)}")

    fundamental_df = load_fundamental_data()

    all_data = load_stock_data(all_codes, TIME['train_start'], TIME['predict_end'])
    all_data = filter_by_training_period(all_data, TIME['train_start'], TIME['train_end'], PREPROCESS['min_trading_days'])

    # 规则引擎层: 数据清洗第一道防线
    print("\n[规则引擎] 数据清洗第一道防线")
    print("=" * 50)
    all_data = validate_price_logic(all_data)
    all_data = validate_limit_bounds(all_data)
    all_data = detect_turnover_spike(all_data)
    print("=" * 50)

    # Phase 2: 预处理 + 基础因子
    print("\n[Phase 2] 预处理 + 基础技术因子")
    print("-" * 50)
    all_data = handle_suspension(all_data, PREPROCESS['max_suspension_days'])

    for col in ['open', 'high', 'low', 'close', 'vwap']:
        if col in all_data.columns:
            all_data[col] = all_data.groupby('asset')[col].ffill()

    all_data = mark_special_states(all_data)
    factor_data = compute_base_factors(all_data)
    del all_data
    gc.collect()
    print("  [内存优化] 释放all_data")

    # Phase 3: 情绪因子
    print("\n[Phase 3] 情绪因子")
    print("-" * 50)
    factor_data = compute_sentiment_factors(factor_data)

    # Phase 4: v29因子
    print("\n[Phase 4] v29新增技术因子")
    print("-" * 50)
    factor_data = compute_v29_factors(factor_data)

    # Phase 5: 多尺度特征
    print("\n[Phase 5] 多尺度时序融合")
    print("-" * 50)
    factor_data, ms_cols = compute_multi_scale_features(factor_data, FACTOR_COLS)
    gc.collect()

    # Phase 6: 行业动量
    print("\n[Phase 6] 行业动量特征")
    print("-" * 50)
    factor_data = compute_industry_momentum(factor_data, sector_map)

    # Phase 7: PCA因子合成
    print("\n[Phase 7] PCA因子合成")
    print("-" * 50)
    factor_data = compute_pca_factors(factor_data, TIME['val_split_date'])
    gc.collect()

    # 确定全部特征列
    all_feature_cols = FACTOR_COLS + ms_cols + INDUSTRY_MOMENTUM_COLS + PCA_FACTOR_COLS + STYLE_COLS
    print(f"\n  全部特征列: {len(all_feature_cols)} 个")
    print(f"    原始因子: {len(FACTOR_COLS)} 个")
    print(f"    多尺度因子: {len(ms_cols)} 个")
    print(f"    行业动量: {len(INDUSTRY_MOMENTUM_COLS)} 个")
    print(f"    PCA隐含因子: {len(PCA_FACTOR_COLS)} 个")
    print(f"    风格概率: {len(STYLE_COLS)} 个")

    # Phase 8: 基本面时间对齐与衍生
    print("\n[Phase 8] 基本面时间对齐与衍生")
    print("-" * 50)
    fundamental_aligned = align_fundamental_to_dates(factor_data, fundamental_df)
    del fundamental_df
    gc.collect()

    if len(fundamental_aligned) > 0:
        factor_data = factor_data.merge(fundamental_aligned, on=['asset', 'date'], how='left')
        print(f"  合并后基本面覆盖率: {factor_data['roe'].notna().mean():.2%}")

    factor_data = compute_fundamental_factors(factor_data)

    # 释放基本面原始列
    fundamental_raw_cols = ['net_profit', 'total_revenue', 'total_equity',
                           'operating_cash_flow', 'total_assets', 'total_liabilities',
                           'operating_cost']
    existing_raw = [c for c in fundamental_raw_cols if c in factor_data.columns]
    if existing_raw:
        for col in existing_raw:
            del factor_data[col]
        gc.collect()
        print(f"  [内存优化] 释放 {len(existing_raw)} 个基本面原始列")

    # Phase 9: 标签计算
    print("\n[Phase 9] 标签计算")
    print("-" * 50)
    if PREPROCESS['label_type'] == 'triple_residual':
        factor_data = compute_triple_residual_labels(factor_data, sector_map, PREPROCESS['forward_days'])
        label_col = 'label_triple_residual'
    elif PREPROCESS['label_type'] == 'rank_neutral':
        factor_data = compute_industry_neutral_labels(factor_data, sector_map, PREPROCESS['forward_days'])
        factor_data['label_rank'] = factor_data.groupby('date')['label_industry_neutral'].rank(pct=True)
        factor_data['label_rank'] = factor_data['label_rank'] * 2 - 1
        label_col = 'label_rank'
        print(f"  排序标签 std: {factor_data[label_col].std():.6f}")
    else:
        factor_data = compute_industry_neutral_labels(factor_data, sector_map, PREPROCESS['forward_days'])
        label_col = 'label_industry_neutral'

    # Phase 10: 风格概率
    print("\n[Phase 10] 风格概率")
    print("-" * 50)
    factor_data = compute_style_probabilities(factor_data, sector_map, fundamental_aligned)
    del fundamental_aligned
    gc.collect()
    print("  [内存优化] 释放fundamental_aligned")

    # Phase 11: 噪声权重
    print("\n[Phase 11] 噪声权重")
    print("-" * 50)
    factor_data = compute_noise_weights(factor_data)

    # 指数衰减权重
    print("\n[Phase 11b] 指数衰减样本权重")
    print("-" * 50)
    exp_weights = compute_exp_decay_weights(factor_data['date'], halflife_years=2.0)
    factor_data['noise_weight'] = factor_data['noise_weight'] * exp_weights
    print(f"  衰减后权重: mean={factor_data['noise_weight'].mean():.4f}")

    # Phase 12: 统计清洗 (三层防御)
    print("\n[Phase 12] 统计清洗 (三层防御)")
    print("-" * 50)

    # 第一层: Winsorize 1%-99%缩尾
    print("  [第一层] Winsorize 1%-99%缩尾...")
    winsorize_count = 0
    for col in FACTOR_COLS:
        if col in factor_data.columns:
            before = factor_data[col].copy()
            factor_data[col] = winsorize_series(factor_data[col], lower_q=0.01, upper_q=0.99)
            clipped = (before != factor_data[col]).sum()
            winsorize_count += clipped
    print(f"  Winsorize缩尾: {winsorize_count:,} 个值被修正")

    # 第二层: MAD截断 (n=3)
    print("  [第二层] MAD截断 (n=3)...")
    mad_clip_count = 0
    for col in FACTOR_COLS:
        if col in factor_data.columns:
            before = factor_data[col].copy()
            factor_data[col] = mad_clip(factor_data[col], 3.0, PREPROCESS['mad_window'])
            clipped = (before != factor_data[col]).sum()
            mad_clip_count += clipped
    print(f"  MAD截断: {mad_clip_count:,} 个值被修正")

    # Phase 13: IC特征筛选
    print("\n[Phase 13] IC特征筛选")
    print("-" * 50)
    factor_data, selected_features, feature_ic_stats = ic_feature_selection(
        factor_data, all_feature_cols, label_col
    )

    # Phase 14: 保存预处理结果
    print("\n[Phase 14] 保存预处理结果")
    print("-" * 50)
    os.makedirs(PATHS['preprocess_dir'], exist_ok=True)

    save_cols = list(set(selected_features + ['asset', 'date', label_col, 'sector', 'noise_weight', 'future_return']))
    save_cols = [c for c in save_cols if c in factor_data.columns]
    factor_data.sort_values(['asset', 'date'], inplace=True)

    factor_data[save_cols].to_csv(os.path.join(PATHS['preprocess_dir'], 'factor_data.csv'), index=False)
    print(f"  factor_data.csv: {len(factor_data):,}行 x {len(save_cols)}列")

    with open(os.path.join(PATHS['preprocess_dir'], 'selected_features.json'), 'w', encoding='utf-8') as f:
        json.dump(selected_features, f, ensure_ascii=False, indent=2)
    print(f"  selected_features.json: {len(selected_features)} 个特征")

    meta = {
        'version': 'v35.0',
        'train_start': TIME['train_start'],
        'train_end': TIME['train_end'],
        'val_split_date': TIME['val_split_date'],
        'predict_end': TIME['predict_end'],
        'label_type': PREPROCESS['label_type'],
        'label_col': label_col,
        'num_features': len(selected_features),
        'num_samples': len(factor_data),
        'num_assets': int(factor_data['asset'].nunique()),
        'selected_features': selected_features,
        'feature_ic_stats': feature_ic_stats,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'bugfixes': [
            '移除PE/PB因子(无总股本数据)',
            '基本面同比periods=252(日频数据)',
            '移除无效PCA噪声剥离',
            '风格概率turnover_accel_vol重命名',
        ],
    }
    with open(os.path.join(PATHS['preprocess_dir'], 'preprocess_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"  preprocess_meta.json: 已保存")

    stats_df = pd.DataFrame(feature_ic_stats)
    stats_df.to_csv(os.path.join(PATHS['preprocess_dir'], 'feature_ic_stats.csv'), index=False, encoding='utf-8-sig')
    print(f"  feature_ic_stats.csv: 已保存")

    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"v35.0 预处理完成! 总耗时: {elapsed/60:.1f} 分钟")
    print(f"股票数: {factor_data['asset'].nunique()} 只")
    print(f"样本数: {len(factor_data):,}")
    print(f"原始特征: {len(all_feature_cols)} 个")
    print(f"IC筛选后: {len(selected_features)} 个")
    print(f"标签列: {label_col}")
    print(f"输出目录: {PATHS['preprocess_dir']}")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    force_regenerate = '--force' in sys.argv
    main(force_regenerate=force_regenerate)
