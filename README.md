# v35 量化选股系统 📈

<p align="center">
  <img src="https://img.shields.io/badge/version-v35.0-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/AI-BiLSTM%20%2B%20Attention-orange.svg" alt="model">
  <img src="https://img.shields.io/badge/市场-A股-green.svg" alt="market">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="license">
</p>

> **一句话看懂：** 把 15 年 A 股数据交给 AI，让它替你 24 小时盯盘、筛股、排雷，最后每天吐出一份「明天该关注哪些股票」的清单。

---

## 🤔 这是什么？（大白话版）

普通人买股票，最难的是两件事：**看不懂** 和 **拿不住**。

- 看不懂：几千支股票，每支都有价格、财报、消息，人脑根本看不过来。
- 拿不住：一涨就慌、一跌就怕，买卖全凭情绪，最后往往追高杀低。

这个系统，就是帮你把这两件事**交给 AI 去做**：

它读遍过去 15 年 A 股每一天的价格、每家公司的财报、以及市场的情绪冷热，学会「**什么样的股票，后来更可能涨**」，然后**每天给你一份优选清单**——该看哪些、该避开哪些，清清楚楚。

> English: A deep-learning stock-selection system for China's A-share market. It learns patterns from 15 years of market data with a BiLSTM + multi-head attention model, then produces a daily shortlist of stocks worth watching. For research and education only.

---

## 💡 它能帮你做什么

| 你以前要做的 | 现在系统替你做 |
|---|---|
| 翻行情、看财报、刷消息，累到眼花 | 自动读取并清洗全市场数据，剔除停牌、涨跌停、异常数据 |
| 凭直觉判断哪支「好像要涨」 | 从 **37 个角度**同时给每支股票打分（像 37 个分析师同时盯一支股票） |
| 纠结买几支、什么时候换 | 用 AI 模型挑出未来更可能上涨的一篮子股票，定期生成**调仓清单** |
| 不知道自己的方法靠不靠谱 | 用十几年历史数据**回测**，过去表现如何一目了然 |

**打个比方：** 相当于你请了一支「24 小时不睡觉、不收工资、只信数据、绝不情绪化」的投研小队，常驻在你电脑里。

---

## 🌟 为什么比「凭感觉炒股」更稳

- **不知疲倦**：37 个指标同时看一支股票，人做不到，AI 一秒算完。
- **不情绪化**：规则说了算，不会被恐惧和贪婪带偏节奏。
- **有凭有据**：每个结论都来自数据，不是群里的小道消息。
- **可检验**：用十几年的历史回测，这套方法到底行不行，数据说话。
- **会排雷**：自动剔除北交所 / 科创板等不适合的策略标的，避开异常股。

> ⚠️ 它**不是**「稳赚不赔的印钞机」。市场没有圣杯，任何模型都可能失效。它的价值是把**主观、随意**的决策，变成**客观、可检验**的流程。（详见文末免责声明）

---

## ✨ 核心亮点

- 🧠 **AI 选股大脑**：基于 **BiLSTM + 多头注意力** 深度学习模型，既看「时间上的走势」，又看「不同指标之间的关联」，比单一指标判断更全面。
- 🔍 **37 个选股视角**：技术面 + 基本面 + 情绪面 + 自研增强因子，并从中学出行业动量、风格概率等「隐藏信号」。
- 🏆 **自动挑有用的**：用「信息系数（IC）」自动筛掉没用的噪声因子，只保留真正有预测力的特征。
- 🧹 **数据会自己洗干净**：内置规则引擎，自动识别价格错误、涨跌停、换手率异常、停牌等脏数据。
- 📅 **每天给你清单**：输入一个日期，自动算出因子并生成「次日关注清单」（支持 Top 3 / Top 10）。
- 📊 **敢不敢用，回测见**：内置回测引擎，输出收益率、夏普比率（风险调整后收益）、胜率等关键指标。
- 🔧 **拿来就能跑**：路径可配置、兼容常见本地数据布局，不依赖固定盘符。

---

## 🛠️ 它是怎么工作的？（三步大白话）

```
第 1 步 · 学历史
   把 15 年 A 股每天的价格 / 财报 / 情绪，喂给 AI
   → AI 从历史里总结出「后来涨得好的股票，长什么样」

第 2 步 · 每天打分
   对最新一天的全市场股票，用 37 个视角逐个打分、排序
   → 谁更可能涨，谁该避开，心里有数

第 3 步 · 出清单
   挑出 Top N 支，生成调仓建议（买哪些 / 换哪些）
   → 你只需要看清单、做决定
```

下面这张图是同样的过程，但用工程师的语言画了一遍（看不懂没关系，上面三步就够了）：

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

## 🚀 三步上手（给想跑起来的你）

```bash
# 1. 拿到代码
git clone <your-repo-url>
cd v35.0

# 2. 装好环境
pip install -r requirements.txt

# 3. 按顺序跑：预处理 → 训练 → 回测 → 出清单
python preprocess_v35.py     # 把原始数据洗干净、算好因子
python train_v35.py          # 训练选股模型
python run_backtest_v35.py   # 看看历史表现
python live_predict_v35.py --date 2026-06-05 --top 10   # 生成某天的关注清单
```

每一步都会输出结果文件，过程与命令细节见下文。

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

## 🎯 使用流程（详细）

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

本项目基于 [MIT License](./LICENSE) 开源，可自由学习、修改、二次创作。

---

## ⚠️ 免责声明

本项目仅供**学习与研究**使用，**不构成任何投资建议**，也不承诺任何收益。

量化模型存在过拟合、市场环境变化导致失效等风险；历史回测表现不代表未来收益。使用本系统产生的任何投资决策与盈亏，均由使用者自行承担。请务必在充分理解风险的前提下谨慎使用。
