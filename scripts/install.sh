#!/bin/bash

# Exit on error
set -e

echo "Installing Algorithmic Trading Package..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install package and dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .

# Create necessary directories
echo "Creating directory structure..."
mkdir -p logs
mkdir -p data
mkdir -p backups

# Set up configuration
echo "Setting up configuration..."
if [ ! -d "config" ]; then
    mkdir config
fi

# Copy template files if they don't exist
if [ ! -f "config/credentials.yaml" ]; then
    cp config/credentials.yaml.template config/credentials.yaml
    echo "Please edit config/credentials.yaml with your API credentials"
fi

# Set correct permissions
echo "Setting permissions..."
chmod 600 config/credentials.yaml
chmod 755 scripts/*.sh

# Create logs directory with correct permissions
mkdir -p logs
chmod 755 logs

echo "Installation complete!"
echo "Next steps:"
echo "1. Edit config/credentials.yaml with your API keys"
echo "2. Run setup_systemd.sh (Linux) or setup_cron.sh (macOS) to configure automation" 