#!/bin/bash
# Quick Pipeline Monitor - 快速监控脚本

PROJECT_DIR="/Users/tianhuihuang/Desktop/cs224_porject"
OUTPUT_LOG="$PROJECT_DIR/output.log"

cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "📊 Pipeline 监控报告"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查进程状态
PID=$(ps aux | grep "run_full_pipeline.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    echo "✅ 进程状态: 运行中 (PID: $PID)"
else
    echo "⚠️  进程状态: 未检测到运行进程"
fi
echo ""

# 图构建进度
GRAPH_COUNT=$(find data/graphs -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
TOTAL_GRAPHS=2317
if [ "$GRAPH_COUNT" -gt 0 ]; then
    PERCENTAGE=$(echo "scale=1; $GRAPH_COUNT * 100 / $TOTAL_GRAPHS" | bc)
    echo "📈 图构建进度: $GRAPH_COUNT / $TOTAL_GRAPHS ($PERCENTAGE%)"
else
    echo "📈 图构建进度: 0 / $TOTAL_GRAPHS (0%)"
fi
echo ""

# 当前阶段
echo "当前阶段:"
if grep -q "Phase 1" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 2" "$OUTPUT_LOG" 2>/dev/null; then
    echo "  🔄 Phase 1: 数据收集/特征工程"
elif grep -q "Phase 2" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 3" "$OUTPUT_LOG" 2>/dev/null; then
    echo "  🔄 Phase 2: 图构建"
    echo "    进度: $GRAPH_COUNT / $TOTAL_GRAPHS 图"
elif grep -q "Phase 3" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 4" "$OUTPUT_LOG" 2>/dev/null; then
    EPOCH=$(grep -o "Epoch [0-9]*" "$OUTPUT_LOG" 2>/dev/null | tail -1)
    echo "  🔄 Phase 3: Baseline 训练 $EPOCH"
elif grep -q "Phase 4" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 5" "$OUTPUT_LOG" 2>/dev/null; then
    EPOCH=$(grep -o "Epoch [0-9]*" "$OUTPUT_LOG" 2>/dev/null | tail -1)
    echo "  🔄 Phase 4: Transformer 训练 $EPOCH"
elif grep -q "Phase 5" "$OUTPUT_LOG" 2>/dev/null && ! grep -q "Phase 6" "$OUTPUT_LOG" 2>/dev/null; then
    echo "  🔄 Phase 5: RL 集成"
elif grep -q "Phase 6" "$OUTPUT_LOG" 2>/dev/null; then
    echo "  🔄 Phase 6: 评估"
elif grep -q "Pipeline Summary\|Full pipeline execution complete" "$OUTPUT_LOG" 2>/dev/null; then
    echo "  ✅ Pipeline 已完成!"
else
    echo "  ⏳ 启动中..."
fi
echo ""

# 最新日志
echo "最新日志 (最后3行):"
tail -3 "$OUTPUT_LOG" 2>/dev/null | sed 's/^/  /'
echo ""

# 错误检查
ERRORS=$(grep -i "error\|exception\|traceback" "$OUTPUT_LOG" 2>/dev/null | tail -3)
if [ -n "$ERRORS" ]; then
    echo "⚠️  最近错误:"
    echo "$ERRORS" | sed 's/^/  /'
    echo ""
fi

# 完成阶段
echo "已完成阶段:"
grep -E "✅.*Phase|❌.*Phase" "$OUTPUT_LOG" 2>/dev/null | tail -6 | sed 's/^/  /' || echo "  暂无"
echo ""

# 文件统计
echo "生成文件统计:"
echo "  - 图文件: $GRAPH_COUNT 个"
echo "  - 模型文件: $(find models -name "*.pt" 2>/dev/null | wc -l | tr -d ' ') 个"
echo "  - 结果文件: $(find results -type f 2>/dev/null | wc -l | tr -d ' ') 个"
echo ""

# 日志文件信息
if [ -f "$OUTPUT_LOG" ]; then
    LOG_SIZE=$(ls -lh "$OUTPUT_LOG" | awk '{print $5}')
    LOG_LINES=$(wc -l < "$OUTPUT_LOG" | tr -d ' ')
    echo "日志文件: $LOG_LINES 行, $LOG_SIZE"
fi

echo "=========================================="

