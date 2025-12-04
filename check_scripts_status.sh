#!/bin/bash
# Script Status Monitor
# 定期检查脚本运行状态

PROJECT_DIR="/Users/tianhuihuang/Desktop/cs224_porject"
cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "📊 脚本运行状态监控"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查正在运行的脚本
echo "🔄 正在运行的脚本:"
RUNNING=$(ps aux | grep -E "python.*scripts|lookahead|sparsification|improved_ablation" | grep -v grep)
if [ -n "$RUNNING" ]; then
    echo "$RUNNING" | awk '{print "  - PID " $2 ": " $11 " " $12 " " $13 " " $14 " " $15}'
    COUNT=$(echo "$RUNNING" | wc -l | tr -d ' ')
    echo "  总计: $COUNT 个脚本正在运行"
else
    echo "  ✅ 没有脚本在运行（可能已完成）"
fi
echo ""

# 检查结果文件
echo "📁 结果文件状态:"
echo ""
for file in "lookahead_horizon_results" "graph_sparsification_results" "ablation_results"; do
    result=$(find results -name "*${file}*" 2>/dev/null | head -1)
    if [ -n "$result" ]; then
        size=$(ls -lh "$result" 2>/dev/null | awk '{print $5}')
        mtime=$(ls -l "$result" 2>/dev/null | awk '{print $6, $7, $8}')
        echo "  ✅ $file"
        echo "     文件: $(basename $result)"
        echo "     大小: $size"
        echo "     修改时间: $mtime"
    else
        echo "  ⏳ $file: 尚未生成"
    fi
    echo ""
done

# 检查日志中的错误
echo "⚠️  最近的错误/警告 (最后50行):"
tail -50 output.log | grep -iE "error|traceback|failed|exception" | tail -3 || echo "  ✅ 未发现错误"
echo ""

# 检查最新进度
echo "📈 最新进度 (最后5行):"
tail -5 output.log | sed 's/^/  /'
echo ""

echo "=========================================="

