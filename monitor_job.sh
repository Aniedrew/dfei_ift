#!/bin/bash
#
# 监控作业 9410780 的完成状态
# 每 30 秒检查一次，并显示最新的日志输出
#

JOB_ID="9410780"
LOG_DIR="/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs"
OUT_LOG="${LOG_DIR}/full_job.out"
ERR_LOG="${LOG_DIR}/full_job.err"

echo "============================================"
echo "监控作业 $JOB_ID"
echo "开始时间: $(date)"
echo "输出日志: $OUT_LOG"
echo "每 30 秒检查一次，按 Ctrl+C 退出"
echo "============================================"
echo ""

PREV_OUT_SIZE=0
while true; do
    # 检查作业状态
    JOB_INFO=$(hep_q -u guoqingxiang 2>/dev/null | grep "$JOB_ID")
    
    if [ -z "$JOB_INFO" ]; then
        echo ""
        echo "============================================"
        echo "作业 $JOB_ID 已不在队列中（可能已完成或已删除）"
        echo "时间: $(date)"
        echo "============================================"
        echo ""
        echo "=== 最终输出日志 ==="
        cat "$OUT_LOG" 2>/dev/null
        echo ""
        echo "=== 错误日志 ==="
        cat "$ERR_LOG" 2>/dev/null
        echo ""
        echo "=== 输出目录文件列表 ==="
        ls -lh /lzufs/user/guoqingxiang/DFEI_data/converted/00342442_inclusive/ 2>/dev/null
        exit 0
    fi
    
    # 提取状态和运行时间
    STATUS=$(echo "$JOB_INFO" | awk '{print $5}')
    RUNTIME=$(echo "$JOB_INFO" | awk '{print $4}')
    
    # 显示最新的输出
    echo "[$(date +%H:%M:%S)] 状态: $STATUS | 运行: $RUNTIME"
    
    # 如果状态已变化，显示日志增量
    if [ -f "$OUT_LOG" ]; then
        CURRENT_SIZE=$(wc -c < "$OUT_LOG")
        if [ "$CURRENT_SIZE" -gt "$PREV_OUT_SIZE" ]; then
            echo "--- 新日志 ---"
            tail -5 "$OUT_LOG"
            echo "---"
            PREV_OUT_SIZE=$CURRENT_SIZE
        fi
    fi
    
    if [ "$STATUS" == "H" ]; then
        echo "⚠️  作业被挂起(Hold)，查看原因:"
        hep_q -u guoqingxiang -hold 2>/dev/null | grep "$JOB_ID"
        echo "尝试释放: hep_release $JOB_ID"
        exit 1
    fi
    
    if [ "$STATUS" == "C" ]; then
        echo ""
        echo "✅ 作业已完成 (Completed)!"
        echo "时间: $(date)"
        cat "$OUT_LOG" 2>/dev/null | tail -20
        exit 0
    fi
    
    sleep 30
done
