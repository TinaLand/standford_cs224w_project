#!/bin/bash
# Pipeline Monitor Script
# Automatically monitors the full pipeline execution and reports progress

PROJECT_DIR="/Users/tianhuihuang/Desktop/cs224_porject"
OUTPUT_LOG="$PROJECT_DIR/output.log"
CHECK_INTERVAL=30  # Check every 30 seconds

cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "🚀 Pipeline Monitor Started"
echo "=========================================="
echo "Monitoring: $OUTPUT_LOG"
echo "Check interval: ${CHECK_INTERVAL} seconds"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to check pipeline status
check_status() {
    local phase=""
    local progress=""
    local last_line=$(tail -1 "$OUTPUT_LOG" 2>/dev/null)
    
    # Detect current phase
    if grep -q "Phase 1.*Data Collection" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 1.*Feature Engineering" "$OUTPUT_LOG" 2>/dev/null; then
        phase="Phase 1: Data Collection"
    elif grep -q "Phase 1.*Feature Engineering" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 2" "$OUTPUT_LOG" 2>/dev/null; then
        phase="Phase 1: Feature Engineering"
    elif grep -q "Phase 2.*Graph Construction" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 3" "$OUTPUT_LOG" 2>/dev/null; then
        phase="Phase 2: Graph Construction"
        # Count graphs
        local graph_count=$(find data/graphs -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
        progress=" (Graphs: $graph_count)"
    elif grep -q "Phase 3.*Baseline" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 4" "$OUTPUT_LOG" 2>/dev/null; then
        phase="Phase 3: Baseline Training"
        # Check for epoch progress
        local epoch=$(grep -o "Epoch [0-9]*" "$OUTPUT_LOG" 2>/dev/null | tail -1)
        if [ -n "$epoch" ]; then
            progress=" ($epoch)"
        fi
    elif grep -q "Phase 4.*Transformer" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 5" "$OUTPUT_LOG" 2>/dev/null; then
        phase="Phase 4: Transformer Training"
        # Check for epoch progress
        local epoch=$(grep -o "Epoch [0-9]*" "$OUTPUT_LOG" 2>/dev/null | tail -1)
        if [ -n "$epoch" ]; then
            progress=" ($epoch)"
        fi
    elif grep -q "Phase 5.*RL" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 6" "$OUTPUT_LOG" 2>/dev/null; then
        phase="Phase 5: RL Integration"
    elif grep -q "Phase 6.*Evaluation" "$OUTPUT_LOG" 2>/dev/null; then
        phase="Phase 6: Evaluation"
    elif grep -q "Pipeline Summary\|Full pipeline execution complete" "$OUTPUT_LOG" 2>/dev/null; then
        phase="✅ COMPLETED"
    else
        phase="Starting..."
    fi
    
    # Check if process is running
    local pid=$(ps aux | grep "run_full_pipeline.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -z "$pid" ]; then
        phase="⚠️  Process not running"
    fi
    
    echo "[$(date '+%H:%M:%S')] $phase$progress"
    
    # Show last meaningful line
    if [ -n "$last_line" ] && [ ${#last_line} -lt 200 ]; then
        echo "   Last: ${last_line:0:100}..."
    fi
}

# Function to generate summary
generate_summary() {
    echo ""
    echo "=========================================="
    echo "📊 Pipeline Summary Report"
    echo "=========================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # File counts
    echo "Generated Files:"
    echo "  - Graphs: $(find data/graphs -name "*.pt" 2>/dev/null | wc -l | tr -d ' ') files"
    echo "  - Models: $(find models -name "*.pt" 2>/dev/null | wc -l | tr -d ' ') files"
    echo "  - Results: $(find results -name "*.csv" -o -name "*.json" 2>/dev/null | wc -l | tr -d ' ') files"
    echo ""
    
    # Log file info
    if [ -f "$OUTPUT_LOG" ]; then
        echo "Log File:"
        echo "  - Size: $(ls -lh "$OUTPUT_LOG" | awk '{print $5}')"
        echo "  - Lines: $(wc -l < "$OUTPUT_LOG" | tr -d ' ') lines"
        echo ""
    fi
    
    # Process status
    local pid=$(ps aux | grep "run_full_pipeline.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$pid" ]; then
        echo "Process Status: ✅ Running (PID: $pid)"
    else
        echo "Process Status: ❌ Not running"
    fi
    echo ""
    
    # Recent errors
    local errors=$(grep -i "error\|exception\|traceback\|failed" "$OUTPUT_LOG" 2>/dev/null | tail -3)
    if [ -n "$errors" ]; then
        echo "Recent Errors/Warnings:"
        echo "$errors" | sed 's/^/  - /'
        echo ""
    fi
    
    # Phase completion status
    echo "Phase Status:"
    grep -E "✅.*Phase|❌.*Phase" "$OUTPUT_LOG" 2>/dev/null | tail -6 | sed 's/^/  /'
    echo ""
}

# Main monitoring loop
iteration=0
while true; do
    iteration=$((iteration + 1))
    
    if [ $((iteration % 10)) -eq 0 ]; then
        # Every 10 iterations (5 minutes), generate full summary
        generate_summary
    else
        # Regular status check
        check_status
    fi
    
    sleep $CHECK_INTERVAL
done

