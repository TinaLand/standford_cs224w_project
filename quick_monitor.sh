#!/bin/bash
# Quick Pipeline Monitor

PROJECT_DIR="/Users/tianhuihuang/Desktop/cs224_porject"
OUTPUT_LOG="$PROJECT_DIR/output.log"

cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo " Pipeline Monitor Report"
echo "=========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check process status
PID=$(ps aux | grep "run_full_pipeline.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    echo " Process Status: Running (PID: $PID)"
else
    echo "  Process Status: No running process detected"
fi
echo ""

# Graph construction progress
GRAPH_COUNT=$(find data/graphs -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
TOTAL_GRAPHS=2317
if [ "$GRAPH_COUNT" -gt 0 ]; then
    PERCENTAGE=$(echo "scale=1; $GRAPH_COUNT * 100 / $TOTAL_GRAPHS" | bc)
    echo " Graph Construction Progress: $GRAPH_COUNT / $TOTAL_GRAPHS ($PERCENTAGE%)"
else
    echo " Graph Construction Progress: 0 / $TOTAL_GRAPHS (0%)"
fi
echo ""

# Current phase
echo "Current Phase:"
if grep -q "Phase 1" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 2" "$OUTPUT_LOG" 2>/dev/null; then
    echo "   Phase 1: Data Collection/Feature Engineering"
elif grep -q "Phase 2" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 3" "$OUTPUT_LOG" 2>/dev/null; then
    echo "   Phase 2: Graph Construction"
    echo "    Progress: $GRAPH_COUNT / $TOTAL_GRAPHS graphs"
elif grep -q "Phase 3" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 4" "$OUTPUT_LOG" 2>/dev/null; then
    EPOCH=$(grep -o "Epoch [0-9]*" "$OUTPUT_LOG" 2>/dev/null | tail -1)
    echo "   Phase 3: Baseline Training $EPOCH"
elif grep -q "Phase 4" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 5" "$OUTPUT_LOG" 2>/dev/null; then
    EPOCH=$(grep -o "Epoch [0-9]*" "$OUTPUT_LOG" 2>/dev/null | tail -1)
    echo "   Phase 4: Transformer Training $EPOCH"
elif grep -q "Phase 5" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 6" "$OUTPUT_LOG" 2>/dev/null; then
    echo "   Phase 5: RL Integration"
elif grep -q "Phase 6" "$OUTPUT_LOG" 2>/dev/null; then
    echo "   Phase 6: Evaluation"
elif grep -q "Pipeline Summary\|Full pipeline execution complete" "$OUTPUT_LOG" 2>/dev/null; then
    echo "   Pipeline Complete!"
else
    echo "  ⏳ Starting..."
fi
echo ""

# Latest logs
echo "Latest Logs (last 3 lines):"
tail -3 "$OUTPUT_LOG" 2>/dev/null | sed 's/^/  /'
echo ""

# Error check
ERRORS=$(grep -i "error\|exception\|traceback" "$OUTPUT_LOG" 2>/dev/null | tail -3)
if [ -n "$ERRORS" ]; then
    echo "  Recent Errors:"
    echo "$ERRORS" | sed 's/^/  /'
    echo ""
fi

# Completed phases
echo "Completed Phases:"
grep -E ".*Phase|.*Phase" "$OUTPUT_LOG" 2>/dev/null | tail -6 | sed 's/^/  /' || echo "  None"
echo ""

# File statistics
echo "Generated File Statistics:"
echo "  - Graph files: $GRAPH_COUNT files"
echo "  - Model files: $(find models -name "*.pt" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  - Result files: $(find results -type f 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

# Log file information
if [ -f "$OUTPUT_LOG" ]; then
    LOG_SIZE=$(ls -lh "$OUTPUT_LOG" | awk '{print $5}')
    LOG_LINES=$(wc -l < "$OUTPUT_LOG" | tr -d ' ')
    echo "Log file: $LOG_LINES lines, $LOG_SIZE"
fi

echo "=========================================="

