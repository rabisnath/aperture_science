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
python demo.py
```
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
