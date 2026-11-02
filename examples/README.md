# 示例 / Examples

本目录提供最小可运行的输入输出样例，便于理解数据格式。

## 1. 调仓文件模板 `input_template.csv`

回测引擎（`backtest/`）的输入格式，**也对应 `live_predict_v35.py` 生成的调仓文件**：

```csv
trade_date,sec_code,weight
2010-02-01,600519,0.3333
2010-02-01,000858,0.3333
2010-02-01,600276,0.3333
```

- `trade_date`：调仓日期（YYYY-MM-DD）
- `sec_code`：6 位股票代码
- `weight`：目标权重（同一调仓日权重之和建议为 1）

## 2. 典型运行流程

```bash
# 1) 数据预处理
python preprocess_v35.py

# 2) 模型训练（生成模型权重与回测调仓文件）
python train_v35.py

# 3) 回测验证
python run_backtest_v35.py

# 4) 实盘预测（输入日期生成次日调仓）
python live_predict_v35.py --date 2026-06-05 --top 10
```

详细参数见根目录 `README.md`。
