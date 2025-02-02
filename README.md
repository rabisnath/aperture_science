# Algorithmic Trading Package

A comprehensive algorithmic trading package for cryptocurrency trading using Alpaca's API.

## Quick Start - Paper Trading

Get started with paper trading in minutes:

1. Install the package:
```bash
pip install -r requirements.txt
```

2. Set up your API credentials:

   Option 1 - Environment Variables:
   ```bash
   # Add these to your shell profile (~/.bashrc, ~/.zshrc, etc.)
   export ALPACA_API_KEY="your_alpaca_key"
   export ALPACA_API_SECRET="your_alpaca_secret"
   ```

   Option 2 - .env File:
   ```bash
   # Create a .env file in the project root
   echo "ALPACA_API_KEY=your_alpaca_key" >> .env
   echo "ALPACA_API_SECRET=your_alpaca_secret" >> .env
   ```

   You can get your Alpaca API credentials by:
   1. Creating an account at https://app.alpaca.markets/signup
   2. Going to Paper Trading in your dashboard
   3. Generating API keys for paper trading

   Note: Make sure to keep your API credentials secure and never commit them to version control.

3. Run the sample paper trading script:
```bash
python demo.py
```

## Features

- Real-time cryptocurrency trading using Alpaca's API
- Multiple trading strategies (Mean Reversion, Momentum, PCA)
- Advanced market analysis and indicators
- Risk management and portfolio optimization
- Paper trading support for testing strategies
- Comprehensive test suite

## Trading Pairs

The package supports all cryptocurrency pairs available on Alpaca, including:
- BTC/USD
- ETH/USD
- And more...

Trading pairs use Alpaca's standard format (e.g., "BTC/USD" instead of "BTCUSD").

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

The test suite includes:
- Data acquisition and preprocessing tests
- Market analysis tests
- Strategy implementation tests
- Portfolio management tests
- End-to-end workflow tests

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
4. Run the test suite
5. Submit a pull request

## License

MIT License - see LICENSE file for details 
