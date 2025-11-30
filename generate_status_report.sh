#!/bin/bash
# Status Report Generator
# Generates detailed status reports for the pipeline

PROJECT_DIR="/Users/tianhuihuang/Desktop/cs224_porject"
OUTPUT_LOG="$PROJECT_DIR/output.log"
REPORT_FILE="$PROJECT_DIR/pipeline_status_report.txt"

cd "$PROJECT_DIR" || exit 1

generate_report() {
    {
        echo "=========================================="
        echo "📊 Pipeline Status Report"
        echo "=========================================="
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        # Process Status
        echo "=== Process Status ==="
        local pid=$(ps aux | grep "run_full_pipeline.py" | grep -v grep | awk '{print $2}' | head -1)
        if [ -n "$pid" ]; then
            echo "✅ Pipeline Running (PID: $pid)"
            local cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ')
            local mem=$(ps -p "$pid" -o %mem= 2>/dev/null | tr -d ' ')
            echo "   CPU: ${cpu}% | Memory: ${mem}%"
        else
            echo "❌ Pipeline Not Running"
        fi
        echo ""
        
        # Current Phase Detection
        echo "=== Current Phase ==="
        if grep -q "Phase 1.*Data Collection.*complete" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 1.*Feature Engineering" "$OUTPUT_LOG" 2>/dev/null; then
            echo "🔄 Phase 1: Data Collection (Completed)"
        elif grep -q "Phase 1.*Feature Engineering" "$OUTPUT_LOG" 2>/dev/null; then
            if grep -q "Phase 1.*Feature Engineering.*Complete" "$OUTPUT_LOG" 2>/dev/null; then
                echo "✅ Phase 1: Feature Engineering (Completed)"
            else
                echo "🔄 Phase 1: Feature Engineering (In Progress)"
            fi
        elif grep -q "Phase 2.*Graph Construction" "$OUTPUT_LOG" 2>/dev/null; then
            if grep -q "Phase 2.*Graph Construction.*Complete" "$OUTPUT_LOG" 2>/dev/null; then
                echo "✅ Phase 2: Graph Construction (Completed)"
            else
                local graph_count=$(find data/graphs -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
                echo "🔄 Phase 2: Graph Construction (In Progress)"
                echo "   Graphs constructed: $graph_count"
            fi
        elif grep -q "Phase 3.*Baseline" "$OUTPUT_LOG" 2>/dev/null; then
            if grep -q "Phase 3.*complete\|Final Test Results" "$OUTPUT_LOG" 2>/dev/null; then
                echo "✅ Phase 3: Baseline Training (Completed)"
            else
                local epoch=$(grep -o "Epoch [0-9]*" "$OUTPUT_LOG" 2>/dev/null | tail -1)
                echo "🔄 Phase 3: Baseline Training (In Progress)"
                [ -n "$epoch" ] && echo "   $epoch"
            fi
        elif grep -q "Phase 4.*Transformer\|Starting Phase 4" "$OUTPUT_LOG" 2>/dev/null; then
            if grep -q "Final Test Results\|Testing.*Complete" "$OUTPUT_LOG" 2>/dev/null; then
                echo "✅ Phase 4: Transformer Training (Completed)"
            else
                local epoch=$(grep -o "Epoch [0-9]*/40" "$OUTPUT_LOG" 2>/dev/null | tail -1)
                echo "🔄 Phase 4: Transformer Training (In Progress)"
                [ -n "$epoch" ] && echo "   $epoch"
            fi
        elif grep -q "Phase 5.*RL" "$OUTPUT_LOG" 2>/dev/null; then
            if grep -q "Phase 5.*complete\|RL.*complete" "$OUTPUT_LOG" 2>/dev/null; then
                echo "✅ Phase 5: RL Integration (Completed)"
            else
                echo "🔄 Phase 5: RL Integration (In Progress)"
            fi
        elif grep -q "Phase 6.*Evaluation" "$OUTPUT_LOG" 2>/dev/null; then
            if grep -q "Phase 6.*complete\|Evaluation.*complete" "$OUTPUT_LOG" 2>/dev/null; then
                echo "✅ Phase 6: Evaluation (Completed)"
            else
                echo "🔄 Phase 6: Evaluation (In Progress)"
            fi
        elif grep -q "Pipeline Summary\|Full pipeline execution complete" "$OUTPUT_LOG" 2>/dev/null; then
            echo "✅ ALL PHASES COMPLETED"
        else
            echo "🔄 Starting..."
        fi
        echo ""
        
        # File Statistics
        echo "=== Generated Files ==="
        echo "Graphs: $(find data/graphs -name "*.pt" 2>/dev/null | wc -l | tr -d ' ') files"
        echo "Models: $(find models -name "*.pt" 2>/dev/null | wc -l | tr -d ' ') files"
        echo "Results: $(find results -name "*.csv" -o -name "*.json" 2>/dev/null | wc -l | tr -d ' ') files"
        echo "Logs: $(find logs -type f 2>/dev/null | wc -l | tr -d ' ') files"
        echo ""
        
        # Log File Info
        if [ -f "$OUTPUT_LOG" ]; then
            echo "=== Log File Info ==="
            echo "Size: $(ls -lh "$OUTPUT_LOG" | awk '{print $5}')"
            echo "Lines: $(wc -l < "$OUTPUT_LOG" | tr -d ' ')"
            echo ""
        fi
        
        # Recent Activity (last 5 meaningful lines)
        echo "=== Recent Activity ==="
        grep -E "✅|❌|Phase|Epoch|Progress|Complete|Error" "$OUTPUT_LOG" 2>/dev/null | tail -5 | sed 's/^/  /'
        echo ""
        
        # Error Summary
        local error_count=$(grep -ic "error\|exception\|traceback\|failed" "$OUTPUT_LOG" 2>/dev/null || echo "0")
        if [ "$error_count" -gt 0 ]; then
            echo "=== Error Summary ==="
            echo "Total errors/warnings: $error_count"
            echo "Recent errors:"
            grep -i "error\|exception\|traceback" "$OUTPUT_LOG" 2>/dev/null | tail -3 | sed 's/^/  - /'
            echo ""
        fi
        
        # Phase Completion Summary
        echo "=== Phase Completion Status ==="
        grep -E "✅.*Phase|❌.*Phase" "$OUTPUT_LOG" 2>/dev/null | tail -8 | sed 's/^/  /'
        echo ""
        
        echo "=========================================="
        echo "Report generated at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
    } > "$REPORT_FILE"
    
    cat "$REPORT_FILE"
}

# Generate report
generate_report

