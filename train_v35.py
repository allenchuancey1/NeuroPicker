"""
================================================================================
v35.0 BiLSTM + Attention 训练脚本
================================================================================
修复:
  - input_size动态设置（从selected_features.json读取，不再硬编码40）
  - IC计算改为截面IC均值（行业标准）
  - 调仓生成使用预建asset字典（性能优化）
  - 训练后释放GPU/内存资源
  - PyTorch 2.0+ API更新
================================================================================
"""

import json
import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta

from config_v35 import PATHS, TIME, MODEL, TRAIN
from model_v35 import BiLSTMAttentionV2

warnings.filterwarnings('ignore')


# ============================================================================
# 数据集类
# ============================================================================

class StockDataset(Dataset):
    """股票时序数据集"""
    def __init__(self, X, y, weights=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
        self._default_weight = torch.tensor(1.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.weights is not None:
            return self.X[idx], self.y[idx], self.weights[idx]
        return self.X[idx], self.y[idx], self._default_weight


# ============================================================================
# 数据加载
# ============================================================================

def load_preprocessed_data():
    """加载v35预处理数据"""
    print("\n[数据加载] 加载v35预处理数据...")

    factor_data_path = os.path.join(PATHS['preprocess_dir'], 'factor_data.csv')
    selected_features_path = os.path.join(PATHS['preprocess_dir'], 'selected_features.json')
    
    # 检查预处理文件是否存在
    if not os.path.exists(factor_data_path) or not os.path.exists(selected_features_path):
        print("  错误: 预处理文件不存在!")
        print(f"  factor_data.csv: {factor_data_path}")
        print(f"  selected_features.json: {selected_features_path}")
        print("\n  请先运行预处理脚本:")
        print("  python preprocess_v35.py")
        print("=" * 70)
        raise FileNotFoundError("预处理文件不存在，请先运行preprocess_v35.py")

    factor_data = pd.read_csv(factor_data_path)
    print(f"  因子数据: {len(factor_data):,} 行")

    with open(selected_features_path, 'r') as f:
        feature_cols = json.load(f)
    print(f"  特征数量: {len(feature_cols)}")

    factor_data['date'] = pd.to_datetime(factor_data['date'])
    return factor_data, feature_cols


def build_dataset(factor_data, feature_cols, start_date, end_date):
    """
    从预处理数据构建时序样本数据集 - 内存优化版
    
    优化点:
    - 先估算样本数量，预分配numpy数组
    - 分批处理，避免list append产生大量临时对象
    - 分批转换为tensor，避免一次性转换
    """
    t0 = time.time()
    print(f"\n[数据集构建] {start_date} ~ {end_date}")

    mask = (factor_data['date'] >= start_date) & (factor_data['date'] <= end_date)
    data = factor_data[mask].copy()

    if len(data) == 0:
        print("  警告: 该日期范围内无数据")
        return None, None, None, None

    data = data.sort_values(['asset', 'date']).reset_index(drop=True)

    feature_cols = [c for c in feature_cols if c in data.columns]
    label_col = 'label_rank' if 'label_rank' in data.columns else 'label_triple_residual'

    if label_col not in data.columns:
        print(f"  警告: 未找到标签列 {label_col}")
        return None, None, None, None

    lookback = TRAIN['lookback'] if 'lookback' in TRAIN else 20
    forward = TRAIN['forward'] if 'forward' in TRAIN else 20

    # 第一遍扫描：估算样本数量
    print(f"  估算样本数量...")
    total_assets = data['asset'].nunique()
    count = 0
    est_samples = 0
    for asset, grp in data.groupby('asset', sort=False):
        count += 1
        if count % 1000 == 0:
            print(f"    进度: {count}/{total_assets}")
        n = len(grp)
        if n >= lookback + forward + 10:
            est_samples += max(0, n - lookback - forward)
    
    print(f"  估算样本数: ~{est_samples:,}")
    
    # 预分配numpy数组（避免list append）
    est_samples = max(est_samples, 1000)  # 至少预留1000
    samples = np.full((est_samples, lookback, len(feature_cols)), 0.0, dtype=np.float32)
    labels = np.full(est_samples, 0.0, dtype=np.float32)
    weights = np.full(est_samples, 1.0, dtype=np.float32)
    sample_dates = [None] * est_samples
    
    # 第二遍扫描：填充数据
    print(f"  填充样本数据...")
    count = 0
    filled = 0
    
    for asset, grp in data.groupby('asset', sort=False):
        count += 1
        if count % 1000 == 0:
            print(f"    进度: {count}/{total_assets}, 已填充 {filled:,} 样本")
        
        grp = grp.sort_values('date').reset_index(drop=True)
        n = len(grp)
        
        if n < lookback + forward + 10:
            continue
        
        # NaN填充
        for fc in feature_cols:
            grp[fc] = grp[fc].ffill().bfill().fillna(0)
        
        feat_vals = grp[feature_cols].values.astype(np.float32)
        label_vals = grp[label_col].values.astype(np.float32)
        weight_vals = grp['noise_weight'].values.astype(np.float32) if 'noise_weight' in grp.columns else np.ones(n, dtype=np.float32)
        date_vals = grp['date'].values
        
        # 生成时序样本
        for i in range(lookback, n - forward):
            if not np.isnan(label_vals[i]):
                if filled >= est_samples:
                    # 扩展数组
                    samples = np.concatenate([samples, np.full((est_samples // 2, lookback, len(feature_cols)), 0.0, dtype=np.float32)])
                    labels = np.concatenate([labels, np.full(est_samples // 2, 0.0, dtype=np.float32)])
                    weights = np.concatenate([weights, np.full(est_samples // 2, 1.0, dtype=np.float32)])
                    sample_dates.extend([None] * (est_samples // 2))
                    est_samples = len(samples)
                
                samples[filled] = feat_vals[i - lookback:i]
                labels[filled] = label_vals[i]
                weights[filled] = weight_vals[i]
                sample_dates[filled] = date_vals[i]
                filled += 1
    
    if filled == 0:
        print(f"  错误: 未生成有效样本")
        return None, None, None, None
    
    # 裁剪到实际大小
    samples = samples[:filled]
    labels = labels[:filled]
    weights = weights[:filled]
    sample_dates = sample_dates[:filled]
    sample_dates_arr = np.array(sample_dates)

    print(f"  样本数: {filled:,} ({time.time()-t0:.1f}s)")
    print(f"  X shape: {samples.shape}")
    print(f"  y range: [{labels.min():.4f}, {labels.max():.4f}]")
    
    # 释放临时变量
    del data
    gc.collect()
    
    return samples, labels, weights, sample_dates_arr


# ============================================================================
# 截面IC计算（行业标准）
# ============================================================================

def compute_cross_sectional_ic(preds, targets, dates):
    """
    截面IC计算: 每日截面相关系数的均值
    修复: 原代码使用全局相关系数，不符合行业标准
    """
    unique_dates = np.unique(dates)
    daily_ics = []

    for d in unique_dates:
        mask = dates == d
        if mask.sum() < 20:  # 截面样本不足20只跳过
            continue
        d_pred = preds[mask]
        d_target = targets[mask]

        # 去除NaN
        valid = ~(np.isnan(d_pred) | np.isnan(d_target))
        if valid.sum() < 20:
            continue

        d_pred_v = d_pred[valid]
        d_target_v = d_target[valid]

        # 计算Pearson相关系数
        pred_std = d_pred_v.std()
        target_std = d_target_v.std()
        if pred_std < 1e-10 or target_std < 1e-10:
            continue

        ic = np.corrcoef(d_pred_v, d_target_v)[0, 1]
        if not np.isnan(ic):
            daily_ics.append(ic)

    if len(daily_ics) == 0:
        return 0.0
    return np.mean(daily_ics)


# ============================================================================
# 训练函数
# ============================================================================

def train_model(model, train_loader, val_loader, val_dates, device):
    """
    训练模型，支持混合精度与早停
    修复: IC计算改为截面IC均值
    """
    print("\n[训练开始]")
    print(f"设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN['lr'],
        weight_decay=TRAIN['weight_decay'],
        betas=(0.9, 0.999)
    )

    # OneCycleLR学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN['lr'],
        epochs=TRAIN['max_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # 混合精度训练（PyTorch 2.0+ API）
    use_amp = TRAIN['use_amp'] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_ic = -float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'ic': [], 'lr': []}

    # 打印初始GPU显存占用
    if torch.cuda.is_available():
        print(f"初始GPU显存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    for epoch in range(TRAIN['max_epochs']):
        # ===== 训练阶段 =====
        model.train()
        train_losses = []

        for batch_X, batch_y, batch_w in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = model(batch_X)
                    loss = torch.mean(batch_w * (pred - batch_y) ** 2)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(batch_X)
                loss = torch.mean(batch_w * (pred - batch_y) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN['grad_clip'])
                optimizer.step()

            train_losses.append(loss.item())
            scheduler.step()

        # ===== 验证阶段 =====
        model.eval()
        val_losses, val_preds, val_targets = [], [], []

        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        pred = model(batch_X)
                        loss = torch.mean((pred - batch_y) ** 2)
                else:
                    pred = model(batch_X)
                    loss = torch.mean((pred - batch_y) ** 2)

                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        # 截面IC计算（修复: 使用截面IC均值而非全局相关系数）
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        ic = compute_cross_sectional_ic(val_preds, val_targets, val_dates)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['ic'].append(ic)
        history['lr'].append(current_lr)

        # 打印GPU显存占用（提升GPU利用率）
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | IC: {ic:.4f} | LR: {current_lr:.6f} | GPU: {allocated:.2f}GB / {total:.2f}GB (峰值: {max_allocated:.2f}GB)")
        else:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | IC: {ic:.4f} | LR: {current_lr:.6f}")

        # 早停检查（基于IC）
        if ic > best_ic:
            best_ic = ic
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(PATHS['model_dir'], 'BiLSTM_Attention_v35_best.pth'))
            print(f"  -> 保存最佳模型 (IC: {ic:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= TRAIN['patience']:
            print(f"\n早停触发: 连续{TRAIN['patience']}轮IC无改善")
            break

    print(f"\n最佳验证IC: {best_ic:.4f}")
    return best_ic, history


# ============================================================================
# 调仓文件生成（性能优化版）
# ============================================================================

def generate_rebalance(model, factor_data, feature_cols, device):
    """
    生成预测期调仓文件
    修复: 预建asset分组字典，避免50万次全表扫描
    """
    print("\n" + "=" * 70)
    print("[调仓生成] 生成预测期调仓文件...")
    print("=" * 70)

    predict_mask = (factor_data['date'] >= TIME['predict_start']) & (factor_data['date'] <= TIME['predict_end'])
    predict_data = factor_data[predict_mask].copy()

    if len(predict_data) == 0:
        print("  警告: 预测期无数据")
        return

    predict_dates = sorted(predict_data['date'].unique())
    print(f"  预测期: {TIME['predict_start']} ~ {TIME['predict_end']}")
    print(f"  预测日期数: {len(predict_dates)}")

    # 修复: 预建asset分组字典，避免重复全表扫描
    lookback = TRAIN.get('lookback', 20)
    asset_groups = {asset: grp.sort_values('date') for asset, grp in factor_data.groupby('asset')}
    print(f"  预建 {len(asset_groups)} 个asset分组字典")

    all_predictions = []

    with torch.no_grad():
        for idx, pred_date in enumerate(predict_dates):
            date_data = predict_data[predict_data['date'] == pred_date]
            if len(date_data) == 0:
                continue

            batch_X = []
            batch_meta = []

            for _, row in date_data.iterrows():
                asset = row['asset']
                if asset not in asset_groups:
                    continue

                asset_hist = asset_groups[asset]
                hist_before = asset_hist[asset_hist['date'] < pred_date].tail(lookback)

                if len(hist_before) < lookback:
                    continue

                feat_vals = hist_before[feature_cols].values.astype(np.float32)
                feat_df = pd.DataFrame(feat_vals)
                feat_df = feat_df.ffill().bfill().fillna(0)
                feat_vals = feat_df.values.astype(np.float32)

                batch_X.append(feat_vals)
                batch_meta.append({'date': pred_date, 'asset': asset})

            if len(batch_X) == 0:
                continue

            X_tensor = torch.FloatTensor(np.array(batch_X)).to(device)
            preds = model(X_tensor).cpu().numpy()

            for meta, pred in zip(batch_meta, preds):
                meta['pred_score'] = float(pred)
                all_predictions.append(meta)

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  已处理 {idx + 1}/{len(predict_dates)} 日期, 累计 {len(all_predictions)} 条预测")

    if len(all_predictions) == 0:
        print("  错误: 未生成任何预测")
        return

    pred_df = pd.DataFrame(all_predictions)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    print(f"\n  总预测数: {len(pred_df):,}")
    print(f"  分数范围: [{pred_df['pred_score'].min():.4f}, {pred_df['pred_score'].max():.4f}]")

    # 生成调仓文件
    for top_n in [3, 10]:
        rebalance_records = []
        for date, group in pred_df.groupby('date'):
            top_stocks = group.nlargest(top_n, 'pred_score')
            for rank, (_, row) in enumerate(top_stocks.iterrows(), 1):
                rebalance_records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'asset': row['asset'],
                    'rank': rank,
                    'score': row['pred_score']
                })

        rebalance_df = pd.DataFrame(rebalance_records)
        output_path = os.path.join(PATHS['rebalance_dir'], f'v35_bilstm_rebalance_{top_n}stocks.csv')
        rebalance_df.to_csv(output_path, index=False)
        print(f"  Top{top_n}: {len(rebalance_df)} 条记录 -> {output_path}")

    # 回测引擎输入格式
    for top_n in [3, 10]:
        rebalance_records = []
        for date, group in pred_df.groupby('date'):
            top_stocks = group.nlargest(top_n, 'pred_score')
            for _, row in top_stocks.iterrows():
                rebalance_records.append({
                    'trade_date': date.strftime('%Y-%m-%d'),
                    'sec_code': row['asset'],
                    'weight': 1.0 / top_n
                })

        bt_df = pd.DataFrame(rebalance_records)
        bt_path = os.path.join(PATHS['rebalance_dir'], f'v35_bilstm_bt_{top_n}stocks.csv')
        bt_df.to_csv(bt_path, index=False)
        print(f"  回测输入 Top{top_n}: {bt_path}")

    print("\n调仓生成完成!")


# ============================================================================
# 主函数
# ============================================================================

def main():
    t_total = time.time()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载预处理数据
    factor_data, feature_cols = load_preprocessed_data()

    # 修复: 动态设置input_size，不再硬编码40
    input_size = len(feature_cols)
    print(f"  动态input_size: {input_size}")

    # 构建训练集与验证集
    val_split = pd.Timestamp(TIME['val_split_date'])
    train_end = val_split - timedelta(days=1)

    print("\n" + "=" * 70)
    print("构建训练集...")
    X_train, y_train, w_train, _ = build_dataset(
        factor_data, feature_cols, TIME['train_start'], train_end.strftime('%Y-%m-%d')
    )

    print("\n构建验证集...")
    X_val, y_val, w_val, val_dates = build_dataset(
        factor_data, feature_cols, TIME['val_split_date'], TIME['train_end']
    )

    if X_train is None or X_val is None:
        print("数据集构建失败!")
        return

    # 创建DataLoader
    train_dataset = StockDataset(X_train, y_train, w_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN['batch_size'], shuffle=True, num_workers=TRAIN['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=TRAIN['batch_size'], shuffle=False, num_workers=TRAIN['num_workers'])

    print(f"\n训练样本: {len(train_dataset):,}")
    print(f"验证样本: {len(val_dataset):,}")

    # 创建模型（使用动态input_size）
    print("\n" + "=" * 70)
    print("创建v35 BiLSTM + Attention模型...")
    model = BiLSTMAttentionV2(
        input_size=input_size,
        hidden_size=MODEL['hidden_size'],
        num_layers=MODEL['num_layers'],
        dropout=MODEL['dropout'],
        attention_heads=MODEL['attention_heads']
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")

    # 训练
    best_ic, history = train_model(model, train_loader, val_loader, val_dates, device)

    # 保存训练结果
    results = {
        'best_ic': float(best_ic),
        'input_size': input_size,
        'param_count': param_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(PATHS['output_dir'], 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(PATHS['output_dir'], 'training_history.csv'), index=False)

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳验证IC: {best_ic:.4f}")
    print(f"模型保存: {PATHS['model_dir']}/BiLSTM_Attention_v35_best.pth")
    print("=" * 70)

    # 修复: 释放训练资源（保留GPU缓存以提升后续GPU操作效率）
    del X_train, y_train, w_train, X_val, y_val, w_val, val_dates
    del train_dataset, val_dataset, train_loader, val_loader
    gc.collect()
    print("  [内存优化] CPU训练资源已释放")

    # 重新加载最佳模型，生成调仓文件
    model.load_state_dict(torch.load(
        os.path.join(PATHS['model_dir'], 'BiLSTM_Attention_v35_best.pth'),
        map_location=device,
        weights_only=True  # 修复: 安全加载
    ))
    model.eval()
    generate_rebalance(model, factor_data, feature_cols, device)

    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"v35.0 全部流程完成! 总耗时: {elapsed/60:.1f} 分钟")
    print("=" * 70)


if __name__ == '__main__':
    main()
