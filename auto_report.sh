#!/bin/bash
# Auto Report Script - Runs every 5 minutes
cd /Users/tianhuihuang/Desktop/cs224_porject
while true; do
    ./generate_status_report.sh
    echo ""
    echo "--- Next report in 5 minutes ---"
    sleep 300  # 5 minutes
done
