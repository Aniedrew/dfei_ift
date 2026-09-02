#!/bin/bash
#
# DFEI 逐项消融链 (ab01..ab09): 从 v31 出发, PPT 顺序每次只加一个优化
# 每段 = 一个 ab 步骤 (20 epoch 微调 + 自动 thr0.9 评估), 完成后自动推进下一步并重提交自己
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

STEP_FILE=/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_chain.step
CONFIGS=(ab01_rebal ab02_b2 ab03_cl2w ab04_hinge ab05_ce ab06_source ab07_mass ab08_struct ab09_mom)
N_STEPS=${#CONFIGS[@]}
V0=500          # ab01 的版本号 (log_version)
LOG_DIR=LHCb_logs/DFEI

# === 读取当前步 (0..8), 默认从头开始 ===
STEP=0
[ -f "$STEP_FILE" ] && STEP=$(cat "$STEP_FILE")
if [ "$STEP" -ge "$N_STEPS" ]; then
  echo "[DONE] 消融链 9 步全部完成 (step=$STEP/$N_STEPS)"
  exit 0
fi
NAME=${CONFIGS[$STEP]}
CFG=config_files/train_CERN_${NAME}.yaml

echo "========================================"
echo "JOB ID      : $_CONDOR_IHEP_JOB_ID"
echo "HOST        : $(hostname)"
echo "START TIME  : $(date)"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "STEP        : $((STEP+1))/$N_STEPS  ($NAME)"
echo "CONFIG      : $CFG"
echo "========================================"

# === 前置检查: 上一步 best ckpt 必须已存在 (cpt=int 依赖它) ===
if [ "$STEP" -gt 0 ]; then
  PREV_VER=$((V0 + STEP - 1))
  PREV_BEST=$(ls "$LOG_DIR"/version_${PREV_VER}/checkpoints/*best-epoch*.ckpt 2>/dev/null | head -1)
  if [ -z "$PREV_BEST" ]; then
    echo "[FAIL] 上一步 version_${PREV_VER} 尚无 best checkpoint, 中止 (避免死循环)"
    exit 1
  fi
  echo "[PRE] 上一步 best: $(basename "$PREV_BEST")"
else
  echo "[PRE] step ab01: 起点 = v31 best (cpt=31)"
fi

# === GPU 预检 (失败自动重排) ===
echo "[PREFLIGHT] GPU check at $(date), device=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>&1 | head -3
python3 -u -c "
import torch
if not torch.cuda.is_available():
    print('[PREFLIGHT] FAIL: torch.cuda.is_available()=False')
    raise SystemExit(77)
p = torch.cuda.get_device_properties(0)
print(f'[PREFLIGHT] OK: {p.name} (cap {p.major}.{p.minor}, mem {p.total_memory/1024**3:.1f} GB)')
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

# === 本步是否真的产出了 best? ===
VER=$((V0 + STEP))
BEST_CNT=$(ls "$LOG_DIR"/version_${VER}/checkpoints/*best-epoch*.ckpt 2>/dev/null | wc -l)
if [ "$EXIT_CODE" -ne 0 ] || [ "$BEST_CNT" -eq 0 ]; then
  echo "[FAIL] 本步 ($NAME) 未产出 best checkpoint (exit=$EXIT_CODE, best=$BEST_CNT). 停止链, 避免死循环"
  JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
  curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
    -d "title=[DFEI] ❌ 消融链中断 at $NAME" \
    -d "desp=## 消融链 step $((STEP+1))/$N_STEPS ($NAME) 失败
| 作业ID | ${JOB_ID} |
| exit | ${EXIT_CODE} |
| best ckpt | ${BEST_CNT} |
| 时间 | $(date) |" > /dev/null 2>&1
  exit 1
fi

# === 推进到下一步并链式重提交 ===
STEP=$((STEP+1))
echo "$STEP" > "$STEP_FILE"
if [ "$STEP" -lt "$N_STEPS" ]; then
  echo "[CHAIN] 本步 $NAME 完成, 推进到下一步 ${CONFIGS[$STEP]} ($STEP/$N_STEPS), 重提交"
  hep_sub submit_ablation_chain.sh -g ghigh -gpu 1 -cpu 4 -m 64000 -wt long \
      -o logs/ablation_chain.out -e logs/ablation_chain.err
else
  echo "[DONE] 消融链全部 9 步完成"
fi

# === 每步完成通知 ===
JOB_ID="${_CONDOR_IHEP_JOB_ID:-unknown}"
curl -s --connect-timeout 10 -X POST https://sctapi.ftqq.com/SCT387631TDiuLj6UNUsFTaDRjkaSWcdPv.send \
  -d "title=[DFEI] ✅ 消融 step $((STEP))/$N_STEPS ($NAME) 完成" \
  -d "desp=## 消融链进度 $STEP/$N_STEPS
| 作业ID | ${JOB_ID} |
| 步骤 | ${NAME} (version_${VER}) |
| 时间 | $(date) |
| 下一步 | $( [ "$STEP" -lt "$N_STEPS" ] && echo ${CONFIGS[$STEP]} || echo 全部完成 ) |

\`\`\`bash
tail -30 /lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/logs/ablation_chain.out
\`\`\`" > /dev/null 2>&1

exit 0
