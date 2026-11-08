#!/bin/bash
#
# watch_and_compare.sh - 监控 3 个 GPU 任务进度, F 评估完成后自动触发 LCA 阈值对比
#
# 用法: nohup bash watch_and_compare.sh > logs/watch_and_compare.log 2>&1 &
#
# 监控对象:
#   v32  CERN 续训训练 (进度: metrics.csv 最新 epoch)
#   基线 v31 评估 (进度: signal_reco_df...v31_paper_thr02.csv)
#   F    v31+方案F 评估 (进度: signal_reco_df...v31_f_k12_h3_s05.csv)
# 触发: F CSV 生成 -> 运行 compare_chain_lca_thr.py --submit (多阈值对比) -> 微信通知
#

BASE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
V32_METRICS=$BASE/LHCb_logs/DFEI/version_32/metrics.csv
BASE_CSV=$BASE/LHCb_logs/DFEI/version_31/signal_reco_df_inclusive_00342442__v31_paper_thr02.csv
F_CSV=$BASE/LHCb_logs/DFEI/version_31/signal_reco_df_inclusive_00342442__v31_f_k12_h3_s05.csv
CHECK_INTERVAL=60
SEND_KEY="SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv"

notify() {
  local title="$1" desc="$2"
  curl -s --connect-timeout 10 -X POST "https://sctapi.ftqq.com/${SEND_KEY}.send" \
    -d "title=${title}" -d "desp=${desc}" > /dev/null 2>&1
}

echo "[$(date)] watch_and_compare 启动: 监控 v32训练 + 基线评估 + F评估, F完成后自动对比"
notify "[DFEI] 👀 已开始监控" "## 监控已启动
- v32 CERN 续训
- v31 基线评估
- v31 方案F评估
F 评估完成后将自动触发 chain_lca_filter 多阈值对比。"

prev_ep=""
while true; do
  sleep "$CHECK_INTERVAL"

  # v32 训练进度 (metrics.csv 最新 epoch)
  ep=$(tail -1 "$V32_METRICS" 2>/dev/null | cut -d, -f1)
  [ -z "$ep" ] && ep="-"
  # 作业状态
  n_jobs=$(hep_q -u guoqingxiang 2>/dev/null | grep -c guoqingxiang || echo 0)
  # 评估 CSV 状态
  base_ok="no"; f_ok="no"
  [ -f "$BASE_CSV" ] && base_ok="YES"
  [ -f "$F_CSV" ] && f_ok="YES"

  if [ "$ep" != "$prev_ep" ]; then
    echo "[$(date '+%H:%M:%S')] v32_epoch=$ep | 作业数=$n_jobs | 基线CSV=$base_ok | F_CSV=$f_ok"
    prev_ep="$ep"
  fi

  # F 评估完成 -> 触发对比
  if [ -f "$F_CSV" ]; then
    echo "[$(date)] ✅ F 评估完成 (CSV 生成), 触发 chain_lca 多阈值对比..."
    notify "[DFEI] ✅ F评估完成, 开始LCA阈值对比" "## 方案F评估已完成
- v32 epoch: $ep
- 基线CSV: $base_ok
- F CSV: ✅
开始自动运行: compare_chain_lca_thr.py --submit"
    cd "$BASE" || exit 1
    python3 -u compare_chain_lca_thr.py --submit \
      >> "$BASE/logs/compare_chain_lca_thr.log" 2>&1
    RC=$?
    echo "[$(date)] 对比脚本结束 rc=$RC, 结果: $BASE/LHCb_logs/DFEI/version_31/chain_lca_thr_compare.md"
    if [ $RC -eq 0 ]; then
      summary=$(tail -8 "$BASE/LHCb_logs/DFEI/version_31/chain_lca_thr_compare.md" 2>/dev/null | tr '\n' ' ')
      notify "[DFEI] ✅ LCA阈值对比完成" "## chain_lca_filter 多阈值对比完成
| 项目 | 值 |
|------|-----|
| **v32 epoch** | $ep |
| **结果** | version_31/chain_lca_thr_compare.md |
| **退出码** | $RC |

\`\`\`
$(tail -8 "$BASE/LHCb_logs/DFEI/version_31/chain_lca_thr_compare.md" 2>/dev/null)
\`\`\`"
    else
      notify "[DFEI] ❌ LCA阈值对比失败" "## 对比脚本失败 (rc=$RC)
日志: logs/compare_chain_lca_thr.log"
    fi
    exit 0
  fi
done
