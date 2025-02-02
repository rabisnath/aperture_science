# Algorithmic Trading Package Implementation Guide

## Overview

This guide explains how to set up automated paper trading and live trading using our algorithmic trading package. The system consists of two main automated components:

1. **Five Minute Loop**: Handles real-time paper trading and live trading
2. **Daily Loop**: Manages strategy evaluation and portfolio optimization

## Prerequisites

- Python 3.8+
- pip
- git
- systemd (Linux) or cron (macOS/Linux)
- Exchange API credentials

## Directory Structure

After installation, your package should look like this:
```
trading_package/
├── README.md
├── IMPLEMENTATION.md
├── setup.py
├── requirements.txt
├── config/
│   ├── default_config.yaml
│   ├── credentials.yaml.template
│   └── logging.yaml
├── scripts/
│   ├── install.sh
│   ├── setup_systemd.sh
│   └── setup_cron.sh
├── systemd/
│   ├── trading-five-min.service
│   └── trading-daily.service
└── trading/
    └── [all module files]
```

## Installation Steps

1. Clone the repository:
```bash
git clone <repository>
cd trading_package
```

2. Run the installation script:
```bash
bash scripts/install.sh
```

3. Configure your credentials:
```bash
cp config/credentials.yaml.template config/credentials.yaml
# Edit credentials.yaml with your API keys
```

4. Choose your automation method:

For Linux (systemd):
```bash
sudo bash scripts/setup_systemd.sh
```

For macOS/Linux (cron):
```bash
bash scripts/setup_cron.sh
```

## Strategy Zoo Setup

The Strategy Zoo allows you to quickly initialize many variants of our basic strategies. Here's how to set it up:

```python
from trading.strategy_zoo import StrategyZoo
from trading.portfolio_manager import PortfolioManager
from trading.strategy import StrategyFactory

# Get your trading symbols
symbols = ["BTCUSDT", "ETHUSDT"]  # Example symbols

# Create strategy variants
mean_rev_strategies = StrategyZoo.create_mean_reversion_variants(symbols)
momentum_strategies = StrategyZoo.create_momentum_variants(symbols)
pca_strategies = StrategyZoo.create_pca_variants(symbols)

# Register with portfolio manager
portfolio_mgr = PortfolioManager()
for strategy in [*mean_rev_strategies, *momentum_strategies, *pca_strategies]:
    portfolio_mgr.add_strategy(StrategyFactory.create(strategy))
```

## Service Management

### Systemd (Linux)

Check service status:
```bash
sudo systemctl status trading-five-min
sudo systemctl status trading-daily
```

Start/stop services:
```bash
sudo systemctl start trading-five-min
sudo systemctl stop trading-five-min
sudo systemctl start trading-daily
sudo systemctl stop trading-daily
```

View logs:
```bash
journalctl -u trading-five-min -f
journalctl -u trading-daily -f
```

### Cron (macOS/Linux)

View cron jobs:
```bash
crontab -l
```

Edit cron jobs:
```bash
crontab -e
```

View logs:
```bash
tail -f logs/trading.log
```

## Monitoring and Reporting

1. View real-time portfolio status:
```bash
python -m trading.reporting --view portfolio
```

2. Generate performance report:
```bash
python -m trading.reporting --report daily
```

3. Monitor strategy performance:
```bash
python -m trading.reporting --view strategies
```

## Configuration Files

### 1. default_config.yaml
```yaml
trading:
  initial_capital: 100000.0
  max_position_size: 0.1
  risk_free_rate: 0.02
  rebalancing_frequency: daily
  max_strategies: 10
  max_strategy_allocation: 0.2

data:
  default_interval: 5m
  lookback_periods: 100
  high_volume_pairs: 5

risk:
  max_drawdown: 0.2
  var_confidence: 0.95
  position_limit: 0.1
```

### 2. logging.yaml
```yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: default
    filename: logs/trading.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
loggers:
  trading:
    level: INFO
    handlers: [console, file]
    propagate: no
root:
  level: INFO
  handlers: [console]
```

## Troubleshooting

1. Service fails to start:
   - Check logs: `journalctl -u trading-five-min -n 50`
   - Verify Python path: `echo $PYTHONPATH`
   - Check permissions: `ls -l /etc/systemd/system/trading-*.service`

2. Strategy not trading:
   - Check strategy allocation: `python -m trading.reporting --view allocations`
   - Verify market data: `python -m trading.data --check-feed`
   - Check risk limits: `python -m trading.risk --check-limits`

3. Performance issues:
   - Monitor CPU/Memory: `top -u trading`
   - Check disk space: `df -h`
   - Monitor API rate limits: `python -m trading.data --rate-limits`

## Security Considerations

1. API Key Management:
   - Never commit credentials to git
   - Use environment variables when possible
   - Regularly rotate API keys

2. Permission Management:
   - Run services with minimal required permissions
   - Use separate user for trading processes
   - Restrict config file permissions

3. Network Security:
   - Use SSL/TLS for API connections
   - Consider using VPN for production
   - Monitor for unusual activity

## Backup and Recovery

1. Regular Backups:
```bash
# Backup trading data
python -m trading.historian --backup

# Backup configurations
cp -r config/ backups/config_$(date +%Y%m%d)/
```

2. Recovery Process:
```bash
# Restore from backup
python -m trading.historian --restore <backup_date>

# Verify integrity
python -m trading.historian --verify
```

## Updates and Maintenance

1. Update package:
```bash
git pull
pip install -e .
```

2. Restart services:
```bash
sudo systemctl restart trading-five-min
sudo systemctl restart trading-daily
```

3. Verify update:
```bash
python -m trading.reporting --version
python -m trading.reporting --health-check
``` 