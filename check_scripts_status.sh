#!/bin/bash
# Script Status Monitor
# 

PROJECT_DIR="/Users/tianhuihuang/Desktop/cs224_porject"
cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo " "
echo "=========================================="
echo ": $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 
echo " :"
RUNNING=$(ps aux | grep -E "python.*scripts|lookahead|sparsification|improved_ablation" | grep -v grep)
if [ -n "$RUNNING" ]; then
    echo "$RUNNING" | awk '{print "  - PID " $2 ": " $11 " " $12 " " $13 " " $14 " " $15}'
    COUNT=$(echo "$RUNNING" | wc -l | tr -d ' ')
    echo "  : $COUNT "
else
    echo "   "
fi
echo ""

# 
echo " :"
echo ""
for file in "lookahead_horizon_results" "graph_sparsification_results" "ablation_results"; do
    result=$(find results -name "*${file}*" 2>/dev/null | head -1)
    if [ -n "$result" ]; then
        size=$(ls -lh "$result" 2>/dev/null | awk '{print $5}')
        mtime=$(ls -l "$result" 2>/dev/null | awk '{print $6, $7, $8}')
        echo "   $file"
        echo "     : $(basename $result)"
        echo "     : $size"
        echo "     : $mtime"
    else
        echo "  ⏳ $file: "
    fi
    echo ""
done

# 
echo "  / (50):"
tail -50 output.log | grep -iE "error|traceback|failed|exception" | tail -3 || echo "   "
echo ""

# 
echo "  (5):"
tail -5 output.log | sed 's/^/  /'
echo ""

echo "=========================================="

