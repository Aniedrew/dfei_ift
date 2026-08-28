#!/bin/bash
#
# 多路评估看护 (CPU 作业): 同时向多个节点提交相同评估, 任一实例真正开始即清理其余。
#
# 背景: 调度器不检查 GPU 空闲就分配 (e2ac1338 反复被占), 单实例 PREFLIGHT 拦截后
#       自动重提, 但作业内 hep_sub 重提的作业常被调度器静默丢弃 -> 重试链断裂。
# 本脚本: 分散到 gpu03-10 (排除总出问题的 gpu02), 总有一个实例能快速拿到好卡;
#        检测到任一开始评估 (日志含 Loading from checkpoint) 后, kill 队列中
#        所有该 config 的排队实例, 直到评估完成。
#
# 用法:
#   hep_sub watchdog_eval.sh -argu "<config.yaml> [nodes]" -g ghigh -cpu 1 -m 4000 -wt long \
#       -o logs/watchdog.out -e logs/watchdog.err
#   $1 = eval config (config_files/ 下)
#   $2 = (可选) 节点列表, 默认 gpu03 gpu04 gpu05 gpu06 gpu07 gpu08 gpu09 gpu10
#
# 注意: 各实例使用独立日志 logs/watch_<config>_<node>.{out,err}。

CONFIG="${1:?usage: watchdog_eval.sh <config.yaml> [nodes]}"
NODES="${2:-gpu03 gpu04 gpu05 gpu06 gpu07 gpu08 gpu09 gpu10}"
BASE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn
cd "$BASE"

source ~/.bashrc
export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH

CONFIG_BASE="${CONFIG%.yaml}"
MAX_ROUND=3          # 整个评估最多启动轮次 (防 started 实例中途崩)
POLL_S=60
POLLS_PER_ROUND=240  # 每轮最长 4h

echo "[watchdog] $(date) 看护启动: $CONFIG -> $NODES"

for round in $(seq 1 $MAX_ROUND); do
  echo "[watchdog] ==== 第 $round 轮: 提交实例 ===="
  for node in $NODES; do
    # 若该节点已有此 config 作业在队列/运行, 跳过 (避免重复)
    if hep_q 2>/dev/null | grep "$CONFIG" | grep -qE "^\s*[0-9]+\.\d+\s+guoqingxiang"; then
      echo "[watchdog] $node 已有实例, 跳过"
      continue
    fi
    out_log=logs/watch_${CONFIG_BASE}_${node}.out
    err_log=logs/watch_${CONFIG_BASE}_${node}.err
    rm -f "$out_log" "$err_log"
    hep_sub submit_eval.sh -argu "$CONFIG" "$node" -g ghigh -gpu 1 -cpu 4 -m 32000 \
      -wt mid -o "$out_log" -e "$err_log" -wn "$node" >/dev/null 2>&1
    echo "[watchdog] $node 已提交"
    sleep 5
  done

  started=""
  for poll in $(seq 1 $POLLS_PER_ROUND); do
    sleep $POLL_S
    # 1. 检测任一实例真正开始评估
    for node in $NODES; do
      out=logs/watch_${CONFIG_BASE}_${node}.out
      if [ -f "$out" ] && grep -q "Loading from checkpoint" "$out" 2>/dev/null; then
        if [ -z "$started" ]; then
          started="$node"
          echo "[watchdog] $(date) ★ 实例 $node 已开始评估, 清理其余排队实例"
        fi
      fi
    done
    # 2. 清理: kill 该 config 的排队(I)实例
    jobs=$(hep_q 2>/dev/null | grep "$CONFIG" | grep -E "\bI\b" | awk '{print $1}')
    for jid in $jobs; do
      hep_rm "$jid" >/dev/null 2>&1 && echo "[watchdog] kill 排队作业 $jid"
    done
    # 3. 若已开始: 等待其完成 (日志出现 EXIT CODE)
    if [ -n "$started" ]; then
      out=logs/watch_${CONFIG_BASE}_${started}.out
      if [ -f "$out" ] && grep -q "EXIT CODE" "$out" 2>/dev/null; then
        code=$(grep -oE "EXIT CODE\s*:\s*[0-9]+" "$out" | grep -oE "[0-9]+$")
        echo "[watchdog] $(date) 评估结束 ($started, exit=$code)"
        # kill 所有残留该 config 作业
        for jid in $(hep_q 2>/dev/null | grep "$CONFIG" | awk '{print $1}'); do
          hep_rm "$jid" >/dev/null 2>&1
        done
        if [ "$code" = "0" ]; then
          echo "[watchdog] ✅ 成功完成: $CONFIG"
          echo "  结果: $(grep -E 'perfect_reco|all_particles|none_iso' \
            $BASE/LHCb_logs/DFEI/version_*/info_*${CONFIG_BASE}*_reco.txt 2>/dev/null | head -3)"
        else
          echo "[watchdog] ❌ 实例 $started 失败 (exit=$code), 进入下一轮"
        fi
        exit 0
      fi
    fi
  done
  echo "[watchdog] 第 $round 轮超时, 进入下一轮"
done

echo "[watchdog] $(date) 所有轮次结束 (可能仍未成功), 检查日志: logs/watch_${CONFIG_BASE}_*.out"
exit 1
