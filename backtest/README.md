# 量化回测系统 - 使用文档

## 概述

基于 Backtrader 框架的完整回测系统，支持定期调仓策略。**所有参数通过 config.py 配置，无需修改代码。**

## 文件结构

```
backtest/
├── backtest_engine.py    # 主回测引擎 (配置化)
├── config.py             # 配置文件 (所有参数集中管理)
├── input_template.csv    # 输入模板示例
├── example_usage.py      # 使用示例
└── README.md            # 本文档
```

## 快速开始

### 1. 准备调仓数据

创建CSV文件，包含三列：

```csv
trade_date,sec_code,weight
2010-02-01,1,0.25
2010-02-01,2,0.25
2010-03-02,1,0.30
2010-03-02,2,0.20
```

**注意：**
- `trade_date` 必须是**交易日**
- `sec_code` 使用CSV中的数字编码（如1, 2, 6, 7）
- `weight` 总和应接近1

### 2. 修改配置（可选）

编辑 `config.py` 调整参数：

```python
BACKTEST_CONFIG = {
    'initial_cash': 1000000,    # 初始资金
    'commission': 0.0003,        # 手续费率
    'slippage': 0.001,           # 滑点率
    'reserve_cash': 0.05,        # 保留现金比例
    'extend_days': 30,           # 调仓后延长天数
}
```

### 3. 运行回测

```bash
# 显示当前配置
python backtest_engine.py --config

# 运行回测
python backtest_engine.py
```

## 配置说明

### 数据配置 (DATA_CONFIG)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `price_data_path` | CSV文件路径 | 价格数据文件位置 |
| `chunk_size` | 100000 | 分块读取大小 |
| `cache_enabled` | True | 是否启用缓存 |

### 回测配置 (BACKTEST_CONFIG)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_cash` | 1000000 | 初始资金（元） |
| `commission` | 0.0003 | 手续费率（万3） |
| `slippage` | 0.001 | 滑点率（千1） |
| `reserve_cash` | 0.05 | 保留现金比例 |
| `max_stocks` | 50 | 最大持仓数量 |
| `extend_days` | 30 | 调仓后延长天数 |
| `pre_days` | 30 | 数据提前天数 |
| `verbose` | True | 是否打印详细日志 |
| `plot_results` | False | 是否绘制图表 |

## 使用方法

### 方法1: 使用默认配置

```python
from backtest_engine import run_backtest

results = run_backtest('input_template.csv', 'result.json')
print(f"总收益率: {results['total_return']:.2%}")
```

### 方法2: 临时修改参数

```python
# 临时修改参数，不影响config.py
results = run_backtest(
    'input_template.csv',
    initial_cash=500000,    # 临时改为50万
    commission=0.0005,       # 临时改为万5
    verbose=False            # 临时关闭日志
)
```

### 方法3: 修改配置文件

```python
from config import update_config

# 永久修改配置
update_config('backtest', initial_cash=2000000, commission=0.0002)
```

## 回测结果

运行后会输出以下指标：

| 指标 | 说明 |
|------|------|
| 总收益率 | 整个回测期间的收益率 |
| 年化收益率 | 按252个交易日年化 |
| 夏普比率 | 风险调整收益指标 |
| 最大回撤 | 最大亏损幅度 |
| 胜率 | 盈利交易占比 |
| 交易次数 | 总交易次数 |

## 注意事项

1. **交易日选择**：调仓日必须是交易日，非交易日（周末、节假日）会导致调仓失败
2. **股票代码**：使用CSV中的数字编码（如1, 2, 6, 7），不是传统代码
3. **权重总和**：每个调仓日的权重总和应接近1，系统会自动归一化
4. **数据范围**：确保CSV文件包含调仓日期范围内的数据

## 常见问题

### Q: 如何查看当前配置？
```bash
python backtest_engine.py --config
```

### Q: 如何修改数据文件路径？
编辑 `config.py` 中的 `DATA_CONFIG['price_data_path']`

### Q: 调仓日不是交易日会怎样？
系统会跳过该调仓日，建议在准备数据时使用交易日

### Q: 如何关闭详细日志？
```python
# 临时关闭
run_backtest('strategy.csv', verbose=False)

# 永久关闭
update_config('backtest', verbose=False)
```

## 依赖安装

```bash
pip install backtrader pandas numpy matplotlib
```
