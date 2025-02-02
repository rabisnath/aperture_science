#!/bin/bash

# Exit on error
set -e

echo "Setting up cron jobs..."

# Get absolute paths
INSTALL_DIR=$(pwd)
PYTHON_PATH="$INSTALL_DIR/venv/bin/python"

# Create temporary cron file
TEMP_CRON=$(mktemp)

# Export current crontab
crontab -l > "$TEMP_CRON" 2>/dev/null || true

# Add our jobs if they don't exist
if ! grep -q "trading.five_min_loop" "$TEMP_CRON"; then
    echo "*/5 * * * * cd $INSTALL_DIR && $PYTHON_PATH -m trading.five_min_loop >> $INSTALL_DIR/logs/five_min.log 2>&1" >> "$TEMP_CRON"
fi

if ! grep -q "trading.daily_loop" "$TEMP_CRON"; then
    echo "0 0 * * * cd $INSTALL_DIR && $PYTHON_PATH -m trading.daily_loop >> $INSTALL_DIR/logs/daily.log 2>&1" >> "$TEMP_CRON"
fi

# Install new cron file
crontab "$TEMP_CRON"

# Clean up
rm "$TEMP_CRON"

# Create log files with correct permissions
touch "$INSTALL_DIR/logs/five_min.log"
touch "$INSTALL_DIR/logs/daily.log"
chmod 644 "$INSTALL_DIR/logs/"*.log

echo "Cron jobs installed!"
echo "Check cron jobs with: crontab -l"
echo "Monitor logs with: tail -f logs/five_min.log" 