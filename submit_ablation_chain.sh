#!/bin/bash
#
# DFEI 逐项消融链 (ab01..ab09): 从 v31 出发, PPT 顺序每次只加一个优化
#
# 协议 (每步):
#   - 起点 = "当前已接受基线" 的 best checkpoint (不是盲目上一步)
#   - 20 epoch 微调 (lr 3e-5) + 自动 thr0.9 评估 (trainer.py)
#   - 评估用重建指标 (AllParticles, 同 S2 口径); 若本步 AllParticles >= 基线
#     -> 接受, 更新基线 (下一步从这里继续); 否则 -> 回退 (下一步仍从旧基线出发,
#     本步结果保留作为"该优化无效"的证据)
#
# 提交方式:
#   hep_sub submit_ablation_chain.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
#       -o logs/ablation_chain.out -e logs/ablation_chain.err
#

source ~/.bashrc

export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
export PYTHONPATH=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn:$PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn

# === 重提前压缩环境 (作业内 env 常 >64KB, hep_sub 提交会失败) ===
strip_env() {
  local keep="HOME PATH USER LOGNAME HOSTNAME SHELL PWD OMP_NUM_THREADS PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES PYTHONPATH _CONDOR_IHEP_JOB_ID"
  for v in $(env | cut -d= -f1); do
    echo " $keep " | grep -q " $v " || unset "$v" 2>/dev/null
  done
}

STEP_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_chain.step
BASE_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_chain.base   # "ver all perf"
CONFIGS=(ab01_rebal ab02_b2 ab03_cl2w ab04_hinge ab05_ce ab06_source ab07_mass ab08_struct ab09_mom)
N_STEPS=${#CONFIGS[@]}
V0=500
LOG_DIR=LHCb_logs/DFEI

# === 初始化基线: v31 (文档 thr0.9: All 43.42 / Perfect 23.93) ===
if [ ! -f "$BASE_FILE" ]; then
  echo "31 43.42 23.93" > "$BASE_FILE"
fi
read BASE_VER BASE_ALL BASE_PERF < "$BASE_FILE"
echo "[BASE] 当前已接受基线: version_${BASE_VER}  (AllParticles ${BASE_ALL}% / Perfect ${BASE_PERF}%)"

# === 读取当前步 ===
STEP=0
[ -f "$STEP_FILE" ] && STEP=$(cat "$STEP_FILE")
if [ "$STEP" -ge "$N_STEPS" ]; then
  echo "[DONE] 消融链 9 步全部完成 (step=$STEP/$N_STEPS)"
  exit 0
fi
NAME=${CONFIGS[$STEP]}
CFG=config_files/train_CERN_${NAME}.yaml
VER=$((V0 + STEP))

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "STEP        : $((STEP+1))/$N_STEPS  ($NAME) -> version_${VER}"
echo "起点(基线)  : version_${BASE_VER}"
echo "CONFIG      : $CFG"
echo "========================================"

# === 把本步 cpt 改写为当前基线的 best checkpoint ===
BASE_BEST=$(ls "$LOG_DIR"/version_${BASE_VER}/checkpoints/*best-epoch*.ckpt 2>/dev/null | head -1)
if [ -z "$BASE_BEST" ]; then
  echo "[FAIL] 基线 version_${BASE_VER} 无 best checkpoint, 中止"
  exit 1
fi
sed -i "s|^  cpt:.*|  cpt: ${BASE_VER}|" "$CFG"
echo "[PRE] $CFG 的 cpt -> ${BASE_VER}  (best: $(basename "$BASE_BEST"))"

# === GPU 预检 (失败自动重排) ===
echo "[PREFLIGHT] GPU check at $(date), device=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 | head -3
python3 -u -c "
import torch
if not torch.cuda.is_available():
    print('[PREFLIGHT] FAIL: torch.cuda.is_available()=False')
    raise SystemExit(77)
p = torch.cuda.get_device_properties(0)
print(f'[PREFLIGHT] OK: {p.name} (mem {p.total_memory/1024**3:.1f} GB)')
a = torch.randn(500,500,device='cuda')
b = (a @ a).sum().item()
print('[PREFLIGHT] matmul OK, sum=%.3f' % b)
"
PREFLIGHT_RC=$?
if [ $PREFLIGHT_RC -ne 0 ]; then
  echo "[PREFLIGHT] FAIL (rc=$PREFLIGHT_RC): 分配的GPU不可用"
  RETRY_COUNT_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_chain.retry_count
  M=0
  [ -f "$RETRY_COUNT_FILE" ] && M=$(cat "$RETRY_COUNT_FILE")
  M=$((M+1)); echo "$M" > "$RETRY_COUNT_FILE"
  echo "[RETRY] 第 $M 次, sleep 60s 后重排..."
  sleep 60
  strip_env
  hep_sub submit_ablation_chain.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
      -o logs/ablation_chain.out -e logs/ablation_chain.err
  echo "[RETRY] 已重提, 本次退出"
  exit 0
fi

# === 训练本步 (trainer.py 会自动评估 thr0.9) ===
python3 -u wmpgnn/analysis/trainer.py --config "$CFG"
EXIT_CODE=$?
echo "========================================"
echo "EXIT CODE   : $EXIT_CODE"
echo "END TIME    : $(date)"
echo "========================================"

# === 本步产物检查 ===
BEST_CNT=$(ls "$LOG_DIR"/version_${VER}/checkpoints/*best-epoch*.ckpt 2>/dev/null | wc -l)
INFO_FILE=$(ls "$LOG_DIR"/version_${VER}/info_inclusive_00342442__*_reco.txt 2>/dev/null | head -1)
if [ "$EXIT_CODE" -ne 0 ] || [ "$BEST_CNT" -eq 0 ]; then
  echo "[FAIL] 本步 ($NAME) 未产出 best checkpoint (exit=$EXIT_CODE, best=$BEST_CNT). 停止链"
  exit 1
fi
if [ -z "$INFO_FILE" ]; then
  echo "[FAIL] 本步 ($NAME) 无评估输出, 无法判定接受/回退. 停止链 (人工检查 version_${VER})"
  exit 1
fi

# === 解析本步重建指标 ===
STEP_ALL=$(awk '/^all_particles/ {print $3; exit}' "$INFO_FILE")
STEP_PERF=$(awk '/^perfect_reco/ {print $3; exit}' "$INFO_FILE")
if [ -z "$STEP_ALL" ] || [ -z "$STEP_PERF" ]; then
  echo "[FAIL] 无法从 $INFO_FILE 解析重建指标 (all='$STEP_ALL' perf='$STEP_PERF'). 停止链"
  exit 1
fi
echo "[EVAL] $NAME: AllParticles ${STEP_ALL}% / Perfect ${STEP_PERF}%  (基线: ${BASE_ALL}% / ${BASE_PERF}%)"

# === 接受 / 回退判定 (AllParticles 单调非降即接受) ===
JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
if awk -v s="$STEP_ALL" -v b="$BASE_ALL" 'BEGIN{exit !(s >= b)}'; then
  echo "$VER $STEP_ALL $STEP_PERF" > "$BASE_FILE"
  echo "[ACCEPT] $NAME 有效 (${STEP_ALL} >= ${BASE_ALL}), 基线更新为 version_${VER}"
  MSG="✅ ACCEPT $NAME: All ${BASE_ALL}->${STEP_ALL} (Perfect ${BASE_PERF}->${STEP_PERF})"
else
  echo "[REJECT] $NAME 无效 (${STEP_ALL} < ${BASE_ALL}), 回退: 下一步仍从 version_${BASE_VER} 出发"
  MSG="⏸ REJECT $NAME: All ${BASE_ALL}->${STEP_ALL} (Perfect ${BASE_PERF}->${STEP_PERF}) — 已回退, 保留作无效证据"
fi

# === 推进到下一步并链式重提交 ===
STEP=$((STEP+1))
echo "$STEP" > "$STEP_FILE"
read BASE_VER BASE_ALL BASE_PERF < "$BASE_FILE"
if [ "$STEP" -lt "$N_STEPS" ]; then
  echo "[CHAIN] 下一步: ${CONFIGS[$STEP]} (从 version_${BASE_VER} 出发, $STEP/$N_STEPS), 重提交"
  strip_env
  hep_sub submit_ablation_chain.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
      -o logs/ablation_chain.out -e logs/ablation_chain.err
else
  echo "[DONE] 消融链全部 $N_STEPS 步完成, 最终基线 version_${BASE_VER}"
  MSG="$MSG | 🏁 全部完成, 最终模型 version_${BASE_VER}"
fi

# === Server酱 通知 ===
curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
  -d "title=[DFEI] 消融 $MSG" \
  -d "desp=## 消融链 step $STEP/$N_STEPS
| 步骤 | $NAME (version_$VER) |
| 基线 | version_${BASE_VER} (All ${BASE_ALL} / Perfect ${BASE_PERF}) |
| 时间 | $(date) |
| 下一步 | $( [ "$STEP" -lt "$N_STEPS" ] && echo ${CONFIGS[$STEP]} || echo 全部完成 ) |

\`\`\`bash
tail -40 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_chain.out
\`\`\`" > /dev/null 2>&1

exit 0
