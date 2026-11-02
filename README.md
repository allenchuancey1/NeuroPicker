# v35.0 量化选股系统

<p align="center">
  <img src="https://img.shields.io/badge/version-v35.0-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/model-BiLSTM%20%2B%20Attention-orange.svg" alt="model">
  <img src="https://img.shields.io/badge/python-3.8%2B-green.svg" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="license">
</p>

基于 **BiLSTM + 多头注意力机制** 的 A 股深度学习量化选股系统。覆盖从原始行情清洗、因子工程、IC 特征筛选、模型训练，到回测验证与实盘调仓生成的完整链路。

> English: A deep-learning stock selection system for China A-share market, built on a BiLSTM + multi-head attention architecture. It covers the full pipeline from raw data cleaning, factor engineering, IC-based feature selection, model training, backtesting, to live rebalance generation.

---

## ✨ 特性

- **37 个原始因子**：技术面（15）+ 基本面（9）+ 情绪面（3）+ v29 增强（10），并合成行业动量、PCA 隐含、风格概率等衍生特征。
- **IC / IR 自动特征筛选**：基于信息系数自动挑选有效因子（15–40 个），剔除无效噪声。
- **BiLSTM + 多头注意力模型**：捕捉时序依赖与跨特征交互，含残差连接与 LayerNorm 正则化。
- **完整预处理流水线**：14 阶段处理，含去极值（Winsorize / MAD）、标准化、行业中性化、幸存者偏差修正。
- **规则引擎数据清洗**：价格逻辑校验、涨跌停区间校验、换手率异常检测、停牌处理。
- **实盘预测模块**：输入任意交易日期，自动补算因子并生成次日调仓文件（支持 Top-N、自动剔除北交所 / 科创板）。
- **回测引擎**：基于 Backtrader 的定期调仓回测，输出收益率、夏普比率、胜率等指标。
- **路径可移植**：默认基于项目相对路径，兼容原有本地布局，可通过 `QUANT_V35_HOME` 环境变量覆盖，**不依赖固定盘符**。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 数据加载 (data_loader_v35)                         │
│  - 原始行情 + 基本面读取                                       │
│  - 规则引擎清洗（价格逻辑 / 涨跌停 / 换手率异常 / 停牌）       │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2-7: 因子计算 (factor_v35)                            │
│  - 技术因子 / 情绪因子 / v29 因子                             │
│  - 多尺度融合 / 行业动量 / PCA 合成 / 基本面对齐              │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 8-11: 标签与权重 (label_v35)                          │
│  - 行业中性化标签 / 三残差标签                                │
│  - 风格概率 / 噪声权重                                        │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 12-14: 清洗与筛选 (preprocess_v35 + ic_filter_v35)    │
│  - Winsorize 缩尾 + MAD 截断                                 │
│  - IC / IR 特征筛选（15–40 个）                              │
│  - 保存预处理结果                                             │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  训练阶段 (train_v35 + model_v35)                            │
│  - 时序数据集（lookback=20 天）                              │
│  - BiLSTM + Attention 训练，早停（patience=15）              │
│  - 生成调仓文件（Top3 / Top10）                              │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  回测 / 实盘 (run_backtest_v35 + live_predict_v35)           │
│  - 回测引擎计算收益 / 夏普 / 胜率                            │
│  - 实盘预测：输入日期 → 次日调仓文件                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 目录结构

```
v35.0/
├── README.md                 # 本文件
├── LICENSE                  # MIT 许可证
├── .gitignore
├── requirements.txt         # 依赖清单
├── config_v35.py            # 统一配置（路径 / 时间 / 模型 / 训练 / 因子）
├── data_loader_v35.py       # 数据加载与规则引擎清洗
├── factor_v35.py            # 因子计算（技术 / 情绪 / v29 / 行业 / PCA / 基本面）
├── label_v35.py             # 标签计算（行业中性化 / 三残差 / 风格 / 噪声权重）
├── preprocess_v35.py        # 预处理主入口（14 阶段流水线）
├── ic_filter_v35.py         # IC / IR 特征筛选
├── model_v35.py             # BiLSTM + 多头注意力模型定义
├── train_v35.py             # 训练脚本
├── live_predict_v35.py      # 实盘预测（输入日期生成调仓）
├── run_backtest_v35.py      # 回测入口（调用 backtest/ 引擎）
├── backtest/                # 回测引擎（Backtrader，随仓库发布）
│   ├── backtest_engine.py
│   ├── config.py
│   ├── example_usage.py
│   └── input_template.csv
├── data/                    # 原始数据（不纳入版本控制，详见 data/README.md）
│   └── README.md
└── examples/                # 示例与模板
    ├── README.md
    └── input_template.csv
```

> 模块间采用扁平 import（如 `from config_v35 import ...`），保持可运行性；所有因子 / 标签 / 模型模块均可在不修改代码的前提下通过 `config_v35.py` 调整参数。

---

## 🚀 安装

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd v35.0

# 2. 创建虚拟环境（可选但推荐）
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 📊 数据准备

原始数据**不入库**（体积大且含隐私），请按 `data/README.md` 放置：

```
<HOME>/
├── data/
│   ├── a_stock_raw_data/a_stock_all_data_YYYYMMDD_YYYYMMDD.csv  # 日线行情（必需）
│   ├── sector_mapping.csv                                      # 行业映射（必需）
│   └── fundamental/{benefit,debt,cash}_ths.csv                 # 同花顺财报（可选）
├── preprocess/v35.0/   # 自动生成
├── models/             # 自动生成
└── output/             # 自动生成
```

- **默认（自包含）**：`<HOME>` 即项目根目录。
- **复用既有数据目录**：设置环境变量 `QUANT_V35_HOME`（如你原有的 `d:\trace\stock`），系统会自动沿用其下的 `data/`、`preprocess/`、`models/`、`output/`。

```bash
# Windows
set QUANT_V35_HOME=d:\trace\stock
# Linux / macOS
export QUANT_V35_HOME=/data/trace/stock
```

---

## 🎯 使用流程

### 1. 数据预处理

```bash
python preprocess_v35.py            # 首次生成
python preprocess_v35.py --force    # 强制重新生成
```

输出：`preprocess/v35.0/factor_data.csv`、`selected_features.json`、`feature_ic_stats.csv`

### 2. 模型训练

```bash
python train_v35.py
```

输出：`models/BiLSTM_Attention_v35_best.pth`、`output/v35.0/training_history.csv`、`output/rebalance/v35_bilstm_rebalance_*.csv`

### 3. 回测验证

```bash
python run_backtest_v35.py
```

输出：`output/backtest_results/backtest_v35_bilstm_*.json`（含总收益率、夏普比率、胜率）

### 4. 实盘预测

```bash
# 指定日期生成次日调仓（Top 3）
python live_predict_v35.py --date 2026-06-05

# 选 Top 10
python live_predict_v35.py --date 2026-06-05 --top 10

# 指定模型与预处理版本
python live_predict_v35.py --date 2026-06-05 \
    --model BiLSTM_Attention_v35_best.pth --preprocess v35.0
```

输出：`output/rebalance/live_{date}_top{N}.csv`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--date` | 预测日期（必需） | — |
| `--top` | 选股数量 | 3 |
| `--model` | 模型文件名 | `BiLSTM_Attention_v35_best.pth` |
| `--preprocess` | 预处理版本目录 | `v35.0` |
| `--lookback` | 回看天数 | 20 |

---

## ⚙️ 配置说明

所有参数集中于 `config_v35.py`：

- **`PATHS`**：数据与输出路径（基于 `QUANT_V35_HOME` 解析）
- **`TIME`**：训练 / 验证 / 预测时间范围
- **`PREPROCESS`**：回看天数、标签类型（`rank_neutral` / `triple_residual`）等
- **`IC_FILTER`**：IC / ICIR 筛选阈值与特征数量上下限
- **`MODEL`**：`hidden_size`、`num_layers`、`dropout`、`attention_heads`
- **`TRAIN`**：`batch_size`、`lr`、`patience`、`use_amp` 等

显存不足时：减小 `TRAIN['batch_size']`，或设 `use_amp=False`。

---

## 📄 许可证

本项目基于 [MIT License](./LICENSE) 开源。

---

## ⚠️ 免责声明

本项目仅供**学习与研究**使用，不构成任何投资建议。量化模型存在过拟合与实盘失效风险，使用前请充分回测与风险评估，盈亏自负。
