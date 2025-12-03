#!/bin/bash
# Quick status check for pipeline execution

PROJECT_DIR="/Users/tianhuihuang/Desktop/cs224_porject"
OUTPUT_LOG="$PROJECT_DIR/output.log"

echo "=========================================="
echo "📊 Pipeline Status Check"
echo "=========================================="
echo ""

# Check if log file exists
if [ ! -f "$OUTPUT_LOG" ]; then
    echo "❌ Log file not found: $OUTPUT_LOG"
    exit 1
fi

# Check process status
if ps aux | grep -E "run_complete_pipeline|run_full_pipeline" | grep -v grep > /dev/null; then
    echo "✅ Pipeline is running"
    PID=$(ps aux | grep -E "run_complete_pipeline|run_full_pipeline" | grep -v grep | awk '{print $2}' | head -1)
    echo "   PID: $PID"
else
    echo "⚠️  No pipeline process found"
fi

echo ""

# Log file info
if [ -f "$OUTPUT_LOG" ]; then
    SIZE=$(ls -lh "$OUTPUT_LOG" | awk '{print $5}')
    LINES=$(wc -l < "$OUTPUT_LOG" | tr -d ' ')
    echo "📄 Log File:"
    echo "   Size: $SIZE"
    echo "   Lines: $LINES"
    echo ""
fi

# Recent activity
echo "📝 Recent Activity (last 10 lines):"
echo "----------------------------------------"
tail -10 "$OUTPUT_LOG" 2>/dev/null | sed 's/^/   /'
echo ""

# Check for errors
ERROR_COUNT=$(grep -ic "error\|exception\|traceback\|failed" "$OUTPUT_LOG" 2>/dev/null | head -1)
if [ -z "$ERROR_COUNT" ]; then
    ERROR_COUNT=0
fi
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Found $ERROR_COUNT error/warning messages in log"
    echo "   Recent errors:"
    grep -i "error\|exception\|traceback" "$OUTPUT_LOG" 2>/dev/null | tail -3 | sed 's/^/   - /'
    echo ""
fi

# Phase completion status
echo "📊 Phase Status:"
grep -E "✅.*Phase|❌.*Phase|🔄.*Phase" "$OUTPUT_LOG" 2>/dev/null | tail -8 | sed 's/^/   /'

echo ""
echo "=========================================="

