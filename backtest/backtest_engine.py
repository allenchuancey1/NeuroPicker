"""
================================================================================
量化回测系统 - Backtrader版本 (配置化)
================================================================================

职责说明:
  基于Backtrader框架的完整回测系统，支持定期调仓策略。
  所有参数通过config.py配置，无需修改代码。

输入格式:
  CSV文件，包含三列:
    - trade_date: 调仓日期 (格式: YYYY-MM-DD)
    - sec_code: 股票代码 (CSV中的数字编码)
    - weight: 目标权重 (0-1之间)

使用方法:
  1. 修改config.py调整参数
  2. 准备调仓CSV文件
  3. 运行: python backtest_engine.py
================================================================================
"""

import backtrader as bt
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional
import warnings

# 导入配置
from config import (
    DATA_CONFIG, BACKTEST_CONFIG, OUTPUT_CONFIG,
    get_config, update_config, show_config
)


# ============================================================================
# 数据加载模块
# ============================================================================

class DataLoader:
    """数据加载器，支持缓存。"""
    
    _cache = None
    
    @classmethod
    def load_price_data(cls, force_reload: bool = False) -> pd.DataFrame:
        """加载价格数据CSV文件。"""
        if cls._cache is not None and not force_reload and DATA_CONFIG['cache_enabled']:
            return cls._cache
            
        data_path = DATA_CONFIG['price_data_path']
        chunk_size = DATA_CONFIG['chunk_size']
        
        print(f"正在加载价格数据: {data_path}")
        
        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            chunks.append(chunk)
            
        df = pd.concat(chunks, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df['asset'] = df['asset'].astype(str)
        
        print(f"数据加载完成: {len(df)} 条记录, {df['asset'].nunique()} 只股票")
        print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
        
        cls._cache = df
        return df
    
    @classmethod
    def get_stock_data(cls, start_date: str, end_date: str, stock_code: str) -> pd.DataFrame:
        """获取单只股票数据。"""
        all_data = cls.load_price_data()
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=BACKTEST_CONFIG['extend_days'])
        
        df = all_data[
            (all_data['asset'] == stock_code) &
            (all_data['date'] >= start_dt) &
            (all_data['date'] <= end_dt)
        ].copy()
        
        if len(df) == 0:
            raise ValueError(f"找不到股票 {stock_code} 在 {start_date} ~ {end_date} 的数据")
        
        return df[['date', 'asset', 'open', 'high', 'low', 'close', 'volume']]


# ============================================================================
# 回测策略
# ============================================================================

class RebalanceStrategy(bt.Strategy):
    """定期调仓策略。"""
    
    params = (
        ('rebalance_df', None),
        ('reserve_cash', 0.05),
        ('verbose', True),
        ('max_stocks', 50),
    )

    def __init__(self):
        if self.p.rebalance_df is None:
            raise ValueError("必须提供rebalance_df参数")
            
        self.rebalance_df = self.p.rebalance_df.copy()
        self.rebalance_df['trade_date'] = pd.to_datetime(self.rebalance_df['trade_date'])
        self.trade_dates = sorted(self.rebalance_df['trade_date'].unique())
        
        self.order_list = []
        self.buy_stocks_pre = []
        
    def log(self, txt: str, dt=None):
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def next(self):
        dt = pd.Timestamp(self.datas[0].datetime.date(0))
        total_value = self.broker.getvalue()
        self.log(f'当前总资产: {total_value:,.2f}')
        
        if dt not in self.trade_dates:
            return
            
        print(f"\n{'='*60}")
        print(f"调仓日: {dt.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        self._cancel_pending_orders()
        target_positions = self._get_target_positions(dt)
        
        if not target_positions:
            self.log("警告: 当前调仓日没有目标持仓")
            return
            
        sell_list = self._get_sell_list(target_positions)
        self._execute_sells(sell_list)
        self._execute_rebalance(target_positions)
        self.buy_stocks_pre = list(target_positions.keys())
        
    def _cancel_pending_orders(self):
        if self.order_list:
            for order in self.order_list:
                self.cancel(order)
            self.order_list = []
            self.log("已取消所有未完成订单")
            
    def _get_target_positions(self, dt) -> Dict[str, float]:
        day_data = self.rebalance_df[self.rebalance_df['trade_date'] == dt]
        
        total_weight = day_data['weight'].sum()
        if total_weight > 1.01:
            # 权重超过100%时归一化（防止错误数据）
            warnings.warn(f"调仓日 {dt} 权重总和为 {total_weight:.4f}，将进行归一化")
            day_data = day_data.copy()
            day_data['weight'] = day_data['weight'] / total_weight
        # 权重<1.0时不归一化，剩余部分作为现金（支持仓位管理）
            
        if len(day_data) > self.p.max_stocks:
            day_data = day_data.nlargest(self.p.max_stocks, 'weight')
            day_data['weight'] = day_data['weight'] / day_data['weight'].sum()
            
        return dict(zip(day_data['sec_code'], day_data['weight']))
        
    def _get_sell_list(self, target_positions: Dict[str, float]) -> List[str]:
        sell_list = [s for s in self.buy_stocks_pre if s not in target_positions]
        if sell_list:
            self.log(f"上期持仓: {self.buy_stocks_pre}")
            self.log(f"需要卖出: {sell_list}")
        return sell_list
        
    def _execute_sells(self, sell_list: List[str]):
        for stock in sell_list:
            try:
                data = self.getdatabyname(stock)
                if data is None:
                    continue
                pos = self.getposition(data)
                if pos.size > 0:
                    order = self.close(data=data)
                    self.order_list.append(order)
                    self.log(f"卖出 {stock}: 数量 {pos.size}")
            except Exception as e:
                self.log(f"卖出 {stock} 时出错: {e}")
                
    def _execute_rebalance(self, target_positions: Dict[str, float]):
        self.log(f"目标持仓: {list(target_positions.keys())}")
        
        for stock, weight in target_positions.items():
            try:
                data = self.getdatabyname(stock)
                if data is None:
                    continue
                if data.volume[0] == 0:
                    self.log(f"警告: 股票 {stock} 可能停牌，跳过")
                    continue
                    
                target_percent = weight * (1 - self.p.reserve_cash)
                order = self.order_target_percent(data=data, target=target_percent)
                
                if order:
                    self.order_list.append(order)
                    self.log(f"调仓 {stock}: 目标权重 {weight:.4f} -> 实际 {target_percent:.4f}")
                    
            except Exception as e:
                self.log(f"调仓 {stock} 时出错: {e}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            action = '买入' if order.isbuy() else '卖出'
            self.log(f'{action}成交: {order.data._name}, 价格: {order.executed.price:.2f}, '
                    f'数量: {order.executed.size}, 金额: {order.executed.value:.2f}, '
                    f'手续费: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单失败: {order.data._name}, 状态: {order.status}')


# ============================================================================
# 回测引擎
# ============================================================================

class BacktestEngine:
    """回测引擎。"""
    
    def __init__(self, config: Dict = None):
        """
        初始化回测引擎。
        
        参数:
            config: 配置字典，默认使用BACKTEST_CONFIG
        """
        self.config = config or BACKTEST_CONFIG
        
    def load_rebalance_data(self, input_path: str) -> pd.DataFrame:
        """加载调仓数据。"""
        df = pd.read_csv(input_path)
        
        # 标准化列名
        column_mapping = {
            'date': 'trade_date', 'asset': 'sec_code', 'stock': 'sec_code', 'code': 'sec_code',
            'w': 'weight', 'weights': 'weight'
        }
        df = df.rename(columns=column_mapping)
        
        required_cols = ['trade_date', 'sec_code', 'weight']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
            
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['sec_code'] = df['sec_code'].astype(str)
        
        self._validate_weights(df)
        return df
        
    def _validate_weights(self, df: pd.DataFrame):
        """验证权重数据。"""
        if (df['weight'] < 0).any():
            raise ValueError("权重不能为负数")
            
        weight_sums = df.groupby('trade_date')['weight'].sum()
        for date, total in weight_sums.items():
            if abs(total - 1.0) > 0.05:
                warnings.warn(f"调仓日 {date} 权重总和为 {total:.4f}")
                
    def prepare_stock_data(self, rebalance_df: pd.DataFrame) -> pd.DataFrame:
        """准备股票历史数据。"""
        min_date = rebalance_df['trade_date'].min()
        max_date = rebalance_df['trade_date'].max()
        
        start_date = (min_date - pd.Timedelta(days=self.config['pre_days'])).strftime('%Y%m%d')
        end_date = max_date.strftime('%Y%m%d')
        
        stocks = rebalance_df['sec_code'].unique()
        print(f"加载 {len(stocks)} 只股票数据，时间范围: {start_date} ~ {end_date}")
        
        all_data = []
        for stock in stocks:
            try:
                df = DataLoader.get_stock_data(start_date, end_date, stock)
                all_data.append(df)
            except Exception as e:
                print(f"警告: 加载 {stock} 数据失败: {e}")
                
        if not all_data:
            raise ValueError("没有加载到任何股票数据")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.rename(columns={'date': 'datetime', 'asset': 'sec_code'})
        combined_df['openinterest'] = 0
        
        return combined_df
        
    def run(self, input_path: str, save_path: Optional[str] = None) -> Dict:
        """
        运行回测。
        
        参数:
            input_path: 调仓数据CSV路径
            save_path: 结果保存路径
            
        返回:
            dict: 回测结果
        """
        # 1. 加载调仓数据
        print("="*60)
        print("步骤1: 加载调仓数据")
        print("="*60)
        rebalance_df = self.load_rebalance_data(input_path)
        print(f"加载完成: {len(rebalance_df)} 条调仓记录")
        print(f"调仓日期: {rebalance_df['trade_date'].nunique()} 个")
        print(f"股票数量: {rebalance_df['sec_code'].nunique()} 只")
        
        # 2. 准备股票数据
        print("\n" + "="*60)
        print("步骤2: 准备股票数据")
        print("="*60)
        stock_data = self.prepare_stock_data(rebalance_df)
        
        # 3. 设置Backtrader
        print("\n" + "="*60)
        print("步骤3: 设置回测引擎")
        print("="*60)
        cerebro = bt.Cerebro()
        
        cerebro.broker.setcash(self.config['initial_cash'])
        cerebro.broker.setcommission(commission=self.config['commission'])
        cerebro.broker.set_slippage_perc(perc=self.config['slippage'])
        
        print(f"初始资金: {self.config['initial_cash']:,.2f}")
        print(f"手续费率: {self.config['commission']:.4f}")
        print(f"滑点率: {self.config['slippage']:.4f}")
        
        # 4. 添加数据
        print("\n" + "="*60)
        print("步骤4: 添加股票数据")
        print("="*60)
        
        start_date = rebalance_df['trade_date'].min()
        end_date = rebalance_df['trade_date'].max() + pd.Timedelta(days=self.config['extend_days'])
        
        for stock in stock_data['sec_code'].unique():
            df = stock_data[stock_data['sec_code'] == stock].copy()
            df = df.set_index('datetime')
            df = df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]
            
            datafeed = bt.feeds.PandasData(
                dataname=df, fromdate=start_date, todate=end_date,
                timeframe=bt.TimeFrame.Days
            )
            cerebro.adddata(datafeed, name=stock)
            
        print(f"已添加 {len(stock_data['sec_code'].unique())} 只股票数据")
        
        # 5. 添加策略
        print("\n" + "="*60)
        print("步骤5: 添加策略")
        print("="*60)
        cerebro.addstrategy(
            RebalanceStrategy,
            rebalance_df=rebalance_df,
            reserve_cash=self.config['reserve_cash'],
            verbose=self.config['verbose'],
            max_stocks=self.config['max_stocks']
        )
        
        # 6. 添加分析器
        print("\n" + "="*60)
        print("步骤6: 添加分析器")
        print("="*60)
        
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', tann=252)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                           timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe_a')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        
        # 7. 运行回测
        print("\n" + "="*60)
        print("步骤7: 运行回测")
        print("="*60)
        
        initial_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        print(f"\n初始资产: {initial_value:,.2f}")
        print(f"最终资产: {final_value:,.2f}")
        print(f"总收益率: {(final_value/initial_value - 1)*100:.2f}%")
        
        # 8. 获取分析结果
        strat = results[0]
        analysis_results = self._extract_analysis(strat)
        
        # 9. 打印报告
        print("\n" + "="*60)
        print("回测结果报告")
        print("="*60)
        self._print_report(analysis_results)
        
        # 10. 保存结果
        if save_path:
            self._save_results(analysis_results, save_path)
            
        # 11. 绘制图表
        if self.config.get('plot_results', False):
            cerebro.plot(style='candlestick', barup='red', bardown='green')
            
        return analysis_results
        
    def _extract_analysis(self, strat) -> Dict:
        """提取分析结果。"""
        results = {}
        
        annual = strat.analyzers.annual_return.get_analysis()
        results['annual_returns'] = dict(annual)
        
        dd = strat.analyzers.drawdown.get_analysis()
        results['max_drawdown'] = dd.get('max', {}).get('drawdown', 0)
        results['max_drawdown_len'] = dd.get('max', {}).get('len', 0)
        
        returns = strat.analyzers.returns.get_analysis()
        results['total_return'] = returns.get('rtot', 0)
        results['avg_daily_return'] = returns.get('ravg', 0)
        results['annual_return'] = returns.get('rnorm', 0)
        results['annual_volatility'] = returns.get('rnorm100', 0) / 100 if returns.get('rnorm100') else 0
        
        sharpe = strat.analyzers.sharpe.get_analysis()
        results['sharpe_ratio'] = sharpe.get('sharperatio', 0)
        
        trade = strat.analyzers.trade_analyzer.get_analysis()
        if trade:
            results['total_trades'] = trade.get('total', {}).get('total', 0)
            results['win_trades'] = trade.get('won', {}).get('total', 0)
            results['loss_trades'] = trade.get('lost', {}).get('total', 0)
            results['win_rate'] = results['win_trades'] / results['total_trades'] if results['total_trades'] > 0 else 0
            
        time_ret = strat.analyzers.time_return.get_analysis()
        results['daily_returns'] = list(time_ret.values()) if time_ret else []
        
        return results
        
    def _print_report(self, results: Dict):
        """打印回测报告。"""
        print("\n【收益指标】")
        print(f"  总收益率:      {results.get('total_return', 0):.2%}")
        print(f"  年化收益率:    {results.get('annual_return', 0):.2%}")
        print(f"  日均收益率:    {results.get('avg_daily_return', 0):.4%}")
        
        print("\n【风险指标】")
        print(f"  最大回撤:      {results.get('max_drawdown', 0):.2%}")
        print(f"  回撤持续期:    {results.get('max_drawdown_len', 0)} 天")
        print(f"  年化波动率:    {results.get('annual_volatility', 0):.2%}")
        
        print("\n【风险调整指标】")
        sharpe = results.get('sharpe_ratio', 0)
        print(f"  夏普比率:      {sharpe:.4f}" if sharpe is not None else "  夏普比率:      N/A")
        
        print("\n【交易统计】")
        print(f"  总交易次数:    {results.get('total_trades', 0)}")
        print(f"  盈利次数:      {results.get('win_trades', 0)}")
        print(f"  亏损次数:      {results.get('loss_trades', 0)}")
        print(f"  胜率:          {results.get('win_rate', 0):.2%}")
        
        if results.get('annual_returns'):
            print("\n【年度收益】")
            for year, ret in sorted(results['annual_returns'].items()):
                print(f"  {year}: {ret:.2%}")
                
    def _save_results(self, results: Dict, save_path: str):
        """保存回测结果。"""
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
            
        results = convert(results)
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"\n结果已保存: {save_path}")


# ============================================================================
# 便捷函数
# ============================================================================

def run_backtest(input_path: str, save_path: Optional[str] = None, **kwargs) -> Dict:
    """
    便捷函数：运行回测。
    
    参数:
        input_path: 调仓数据CSV路径
        save_path: 结果保存路径
        **kwargs: 覆盖配置参数，如 initial_cash=500000
        
    返回:
        dict: 回测结果
    """
    config = BACKTEST_CONFIG.copy()
    config.update(kwargs)
    
    engine = BacktestEngine(config)
    return engine.run(input_path=input_path, save_path=save_path)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # 显示配置
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        show_config()
        sys.exit(0)
    
    # 默认运行示例
    print("="*60)
    print("量化回测系统")
    print("="*60)
    print("\n使用方法:")
    print("  python backtest_engine.py --config    # 显示配置")
    print("  python backtest_engine.py             # 运行默认回测")
    print("\n")
    
    # 检查是否存在输入文件
    if os.path.exists('input_template.csv'):
        print("运行回测: input_template.csv")
        results = run_backtest('input_template.csv', 'backtest_result.json')
    else:
        print("错误: 找不到 input_template.csv")
        print("请准备调仓数据文件，格式如下:")
        print("  trade_date,sec_code,weight")
        print("  2010-02-01,1,0.25")
        print("  2010-02-01,2,0.25")
