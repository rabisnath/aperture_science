# Algorithmic Trading Package

A comprehensive algorithmic trading package with support for multiple strategies, real-time market analysis, and paper trading.

## Quick Start - Paper Trading

Get started with paper trading in minutes:

1. Install the package:
```bash
pip install -r requirements.txt
```

2. Run the sample paper trading script:
```bash
python sample_usage.py
```

Or customize via environment variables:
```bash
export TRADING_SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT"  # Trading pairs
export INITIAL_CAPITAL="50000.0"                  # Starting capital in USDT
export MAX_POSITION_SIZE="0.05"                   # Maximum position size (5% of capital)
python sample_usage.py
```

## Interactive Demo

Explore the package features through our Jupyter notebook:
```bash
jupyter notebook demo.ipynb
```

The notebook provides a guided tour of:
- Data acquisition and preprocessing
- Market analysis tools
- Strategy implementation
- Portfolio management
- Live trading setup

## Features

- **Multiple Trading Strategies**:
  - PCA-based statistical arbitrage
  - Mean reversion
  - Momentum
  - Hybrid approaches

- **Real-time Analysis**:
  - Market condition monitoring
  - Risk management
  - Performance tracking

- **Production Ready**:
  - Paper trading support
  - Live trading capability
  - Comprehensive logging
  - Performance monitoring

## Project Structure

```
.
├── data/               # Data acquisition and preprocessing
├── strategy/           # Trading strategy implementations
├── portfolio_manager/  # Portfolio management and optimization
├── risk/              # Risk management tools
├── back_of_house/     # Trade execution and monitoring
├── scripts/           # Utility scripts
└── tests/             # Test suite
```

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
python -m unittest test_package.py
```

## Production Deployment

For production deployment, we provide scripts to set up:
- Systemd service
- Cron jobs
- Logging configuration

See `scripts/` directory for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details 