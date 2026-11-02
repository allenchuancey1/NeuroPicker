# 数据说明 / Data

本目录用于放置 **原始数据**。由于文件体积大且可能含个人隐私，**此目录不纳入版本控制**（见根目录 `.gitignore`，仅本说明文件被保留）。

## 1. 目录结构

将以下文件按路径放入（路径在 `config_v35.py` 的 `PATHS` 中配置，可通过环境变量 `QUANT_V35_HOME` 整体覆盖根目录）：

```
<HOME>/
├── data/
│   ├── a_stock_raw_data/
│   │   └── a_stock_all_data_YYYYMMDD_YYYYMMDD.csv   # 日线行情（必需）
│   ├── sector_mapping.csv                            # 行业映射（必需）
│   └── fundamental/                                  # 同花顺财报导出（可选，用于基本面因子）
│       ├── benefit_ths.csv
│       ├── debt_ths.csv
│       └── cash_ths.csv
├── preprocess/v35.0/                                # 预处理输出（自动生成）
├── models/                                          # 训练好的模型权重（自动生成）
└── output/                                          # 训练/回测/调仓输出（自动生成）
```

## 2. 原始行情文件（`raw_data_path`）

标准 A 股日线数据，**至少包含以下列**（列名缺省时由 `data_loader_v35.py` 兼容推导）：

| 列名 | 含义 | 类型 |
|------|------|------|
| `date` | 交易日期 | YYYY-MM-DD |
| `asset` | 6 位股票代码（如 `600519`） | str/int |
| `open` | 开盘价 | float |
| `high` | 最高价 | float |
| `low` | 最低价 | float |
| `close` | 收盘价 | float |
| `volume` | 成交量（股） | float |
| `pctChg` | 涨跌幅（可选，缺省时由 close 推导） | float |

数据来源可为 Tushare / 通联 / 聚宽 / 同花顺等导出的日线行情，需覆盖 `config_v35.py` 中 `TIME` 配置的训练与预测区间。

## 3. 行业映射（`sector_path`）

CSV 文件，至少包含 `asset`（股票代码）及对应行业/板块分类列，用于行业中性化标签与行业动量因子。

## 4. 基本面数据（`fundamental/`，可选）

同花顺（THS）财报导出格式，使用中文列名：

- `benefit_ths.csv`：`股票代码`, `报告期`, `*归属于母公司所有者的净利润`, ...
- `debt_ths.csv`：`股票代码`, `报告期`, `*所有者权益（或股东权益）合计`, `*资产合计`, `*负债合计`
- `cash_ths.csv`：`股票代码`, `报告期`, `*经营活动产生的现金流量净额`

不提供时，仅使用技术面与情绪面因子（37 个原始因子中的基本面部分会缺失）。

## 5. 快速获取示例数据

如需测试流程，可用任意 A 股日线样本（建议 ≥ 200 交易日、覆盖多只股票）放至上述路径，并相应调整 `config_v35.py` 的 `PATHS` 与 `TIME`。
