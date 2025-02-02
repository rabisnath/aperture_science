# Trading System Workflow Demo
# This notebook demonstrates the core functionality of both the 5-minute and daily loops

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb

# Import our modules
from data import DataLoader
from market_analysis import MarketAnalyzer
from strategy import MeanReversionStrategy, MomentumStrategy
from historian import Historian, MarketConditionsRecord
from portfolio_manager import PortfolioManager
from back_of_house import LiveTrader, RiskEngine

# Initialize core components
data_loader = DataLoader()
market_analyzer = MarketAnalyzer()
historian = Historian("data/")
portfolio_mgr = PortfolioManager()
risk_engine = RiskEngine()
live_trader = LiveTrader(risk_engine)

# Set up some sample strategies
strategies = {
    "mean_rev_1": MeanReversionStrategy("mean_rev_1", ["BTC-USDT"], entry_threshold=2.0),
    "mean_rev_2": MeanReversionStrategy("mean_rev_2", ["BTC-USDT"], entry_threshold=1.5),
    "momentum_1": MomentumStrategy("momentum_1", ["BTC-USDT"], lookback=14),
    "momentum_2": MomentumStrategy("momentum_2", ["BTC-USDT"], lookback=21)
}

#######################
# 5-Minute Loop Demo
#######################

def demo_five_minute_loop():
    print("Starting 5-minute loop demo...")
    
    # 1. Fetch latest market data
    market_data = data_loader.fetch_latest_data(symbols=["BTC-USDT"])
    
    # 2. Paper trade all strategies
    paper_trades = []
    for strategy in strategies.values():
        signals = strategy.generate_signals(market_data)
        if signals["BTC-USDT"] != 0:  # If we have a signal
            trade = {
                "strategy_id": strategy.name,
                "symbol": "BTC-USDT",
                "signal": signals["BTC-USDT"],
                "timestamp": datetime.now()
            }
            paper_trades.append(trade)
    
    # 3. Live trade selected strategies
    live_strategies = portfolio_mgr.get_live_strategies()
    for strategy_id in live_strategies:
        if strategy_id in strategies:
            strategy = strategies[strategy_id]
            signals = strategy.generate_signals(market_data)
            
            for symbol, signal in signals.items():
                if signal != 0 and risk_engine.check_trade_safety({"symbol": symbol, "size": signal}):
                    live_trader.execute_trade(symbol, signal)
    
    # 4. Record market conditions
    market_conditions = market_analyzer.analyze_market_conditions(
        np.array([df['close'].pct_change().dropna() for df in market_data.values()])
    )
    historian.save_market_conditions(MarketConditionsRecord(
        timestamp=datetime.now(),
        basic_stats=market_conditions.basic_stats,
        regression_stats=market_conditions.regression_stats,
        correlation_matrix=market_conditions.correlation_matrix,
        strategy_id="market"
    ))
    
    print("5-minute loop completed")

#######################
# Daily Loop Demo
#######################

def demo_daily_loop():
    print("Starting daily loop demo...")
    
    # 1. Load historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Load market conditions
    market_history = historian.load_market_conditions(start_date, end_date)
    
    # Load strategy performance
    strategy_performance = pd.DataFrame([
        {
            'strategy_id': name,
            'sharpe_ratio': strategy.performance_metrics.get('sharpe_ratio', 0),
            'win_rate': strategy.performance_metrics.get('win_rate', 0),
            'total_pnl': strategy.performance_metrics.get('total_pnl', 0)
        }
        for name, strategy in strategies.items()
    ])
    
    # 2. Train XGBoost model
    X = pd.DataFrame([
        {
            'volatility': stats.get('std_dev', 0),
            'skewness': stats.get('skewness', 0),
            'win_rate': perf.get('win_rate', 0)
        }
        for stats, perf in zip(market_history['basic_stats'], strategy_performance.to_dict('records'))
    ])
    
    y = strategy_performance['sharpe_ratio']
    
    model = xgb.XGBRegressor()
    model.fit(X, y)
    
    # 3. Update live trading roster
    latest_market_stats = historian.load_market_conditions(
        end_date - timedelta(days=1), 
        end_date
    )['basic_stats'].iloc[-1]
    
    # Prepare current market features
    current_features = pd.DataFrame([{
        'volatility': latest_market_stats.get('std_dev', 0),
        'skewness': latest_market_stats.get('skewness', 0),
        'win_rate': strategy_performance['win_rate'].mean()
    }])
    
    # Predict performance
    predictions = model.predict(current_features)
    
    # Select top 2 strategies
    strategy_scores = {
        name: pred for name, pred in zip(strategies.keys(), predictions)
    }
    top_strategies = sorted(
        strategy_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:2]
    
    # Update live roster
    portfolio_mgr.update_live_strategies([s[0] for s in top_strategies])
    
    print("Daily loop completed")

# Run demos
demo_five_minute_loop()
print("\n" + "="*50 + "\n")
demo_daily_loop()
