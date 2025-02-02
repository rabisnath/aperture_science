#!/bin/bash

# Exit on error
set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo)"
    exit 1
fi

echo "Setting up systemd services..."

# Get absolute paths
INSTALL_DIR=$(pwd)
PYTHON_PATH=$(which python3)
USER=$(logname)

# Create systemd service files
cat > /etc/systemd/system/trading-five-min.service << EOF
[Unit]
Description=Trading Five Minute Loop
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python -m trading.five_min_loop
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/trading-daily.service << EOF
[Unit]
Description=Trading Daily Loop
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python -m trading.daily_loop
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Set correct permissions
chmod 644 /etc/systemd/system/trading-*.service

# Reload systemd
systemctl daemon-reload

# Enable and start services
systemctl enable trading-five-min
systemctl enable trading-daily
systemctl start trading-five-min
systemctl start trading-daily

echo "Systemd services installed and started!"
echo "Check status with: systemctl status trading-five-min" 