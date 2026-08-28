# DFEI 训练成果报告

> 更新日期：2026-08-09
> 论文：*Scalable Multi-Task Learning for Particle Collision Event Reconstruction with HGNNs*（arXiv:2504.21844, MLST 6 045060）

## 1. 数据来源与格式

| 数据 | 来源 | 格式 | 粒子级 truth | 用途 |
|---|---|---|---|---|
| `converted_LHCbcollision` | **论文数据**（组内 PYTHIA+EvtGen 模拟，Zenodo 15584745）转换 | zst 图（10 维特征） | ❌ 转换时丢失 | v23 训练 |
| `converted_LHCb_truth` | 同上（重新转换，保留 truth） | zst 图（含完整粒子 truth） | ✅ 完整 | v23 评估 |
| `MC_normed` | **CERN 官方 MC**（DFEI_IFT_20260702） | zst 图（新格式，含 part_keys/sig_keys） | 部分（新格式） | v25 训练+评估 |
| `CERN_data_LHCb` | CERN 旧数据 | zst 图（旧格式） | ✅ 完整 | 早期评估（v8） |

**关键点**：论文数据 = 组内模拟（PYTHIA+EvtGen 复现 Run 3 条件），不是 CERN 官方 MC。原始 npy 含完整粒子 truth，本地转换脚本曾丢失，已修复（§6）。

## 2. 训练成果总览

| Version | 数据 | 精度 | Epochs | Best val | 状态 |
|---|---|---|---|---|---|
| v22 | 公开数据 | bf16 | 15 | NaN | ❌ 失败（bf16 溢出） |
| v23 | 论文数据 | 32 | 76（中断） | **0.742** @60 | ⚠️ test 阶段崩溃中断 |
| v24 | — | — | 2 | 0.856 | 早期测试 |
| v25 | **CERN 官方 MC** | 32 | **100 完整** | **0.807** @96 | ✅ 完成 |
| v27/v28 | 续训实验 | 32 | 22/0 | 0.744 | 已停（held/LR 问题） |

## 3. CERN 官方 MC 训练（v25）——训练完成 ✅

- 数据：`MC_normed`（inclusive_00342442），FP32，batch=1，gacc=16
- 训练：100/100 epochs 完成，val_combined_loss 最低 **0.807**（epoch 96），checkpoint `best-epoch=96-val_combined_loss=0.807.ckpt`
- 多任务损失：LCA + 节点剪枝 + 33×边剪枝 + PV 关联

### v25 评估（阈值 0.2，13519 信号链）

| 指标 | v25 | 论文 WHGNN |
|---|---|---|
| PerfectReco | **27.68%** | 21.5% |
| AllParticles | 55.57% | 40.8%（Complete+Perfect） |
| NoneIso | 6.03% | 45.8% |
| PartReco | 15.90% | 13.5% |

- 单 B：28.32% / 双 B：27.13% / 多 B：16.54%
- PV 关联：HGNN miss 0.65%（≈正确率 99.35%），minIP miss 28.2%

> ⚠️ 口径：v25 评估基于 MC_normed 新格式 truth（从 tt.y + sig_keys 构建），与论文完整粒子 truth 口径不同；LCAG（Table1）当时未落盘，可补跑。

## 4. 论文数据训练（v23）——训练中断，评估完成 ✅

- 数据：`converted_LHCbcollision`（论文数据转换），FP32，batch=1，gacc=16
- 训练：跑到 epoch 75/99，best val_combined_loss **0.742**（epoch 60），随后自动 test 崩溃（已修复）
- checkpoint：`best-epoch=60-val_combined_loss=0.742.ckpt`

### Table1 LCAG 分类准确率（12806 链 / 9035 事件）

| Class | 样本数 | pred0 | pred1 | pred2 | pred3 | 论文 |
|---|---|---|---|---|---|---|
| y0（背景） | 4.24 亿 | **99.99%** | 0% | 0.01% | 0% | 99.3% |
| y1（直接共享母） | 10.4 万 | 17.73% | **0.00%** | 82.27% | 0% | 75.9% |
| y2（一代中间态） | 31.2 万 | 33.00% | 0% | **67.00%** | 0% | 61.3% |
| y3（两代） | 5.8 万 | 45.83% | 0% | 54.17% | **54.17%** | 84.0% |

**关键问题：y1 类准确率 0%（全部误判为 y2），y2/y3 也偏低——LCAG 分类弱是重建性能低的直接原因。**

### Table2 重建分类

| 指标 | v23 | 论文 WHGNN |
|---|---|---|
| PerfectReco | **16.50%** | 21.5% |
| AllParticles | 48.62% | 40.8% |
| NoneIso | 10.63% | 45.8% |
| PartReco | 25.51% | 13.5% |
| NotFound | 15.24% | — |

- 单 B：**21.05%**（≈ 论文 21.5%）/ 双 B：13.47% / 多 B：8.23%
- 事件级 PerfectEventReconstruction：**12.54%**
- 信号链分布：1B 5448 / 2B 3442 / 3B 108 / 4B 35 / 5B 2；reco 链数==truth 链数 67.58%

### Table4 PV 关联

track 级 **100%**（论文数据 trpv 图为每 track 单候选，属确定性关联，非判别性指标）

## 5. 论文对比总结

| 指标 | 论文 | v23（论文数据） | v25（CERN 官方 MC） |
|---|---|---|---|
| PerfectReco | 21.5% | 16.50%（单B 21.05%） | 27.68% |
| LCAG y0/y1/y2/y3 | 99.3/75.9/61.3/84.0 | 99.99/0/67/54.2 | 未测 |
| PV 关联 | ~99.8% | 100%（单候选） | 99.35%（HGNN） |

- v25（CERN 官方 MC）重建性能反超论文，但 truth 口径不同，谨慎解读。
- v23（论文数据）是严格 apples-to-apples 对比：单 B 与论文持平，总体偏低源于 **LCAG y1/y2/y3 分类弱**。

## 6. 已修复的关键问题

1. **pv_filter 稀疏边崩溃**：论文数据 track-PV 图稀疏，原 `view(ntracks, npvs)` 稠密化必崩 → 按 track 分组聚合。
2. **论文数据 truth 丢失**：转换脚本未保留粒子 truth → 修复并重新转换（`converted_LHCb_truth`）。
3. **part_ids 容错**：`true_node_pruning` 对无顶层 part_ids 数据崩溃 → 加 hasattr 保护。
4. **续训 LR 重置**：续训用 1e-3 覆盖了 checkpoint 的 6.25e-5，val 从 0.742 恶化到 1.768 → 改匹配小 LR。
5. **内存 held**：32GB 上限触发 HTCondor held → 提交 `-m 64000`。
6. **坏 GPU 调度**：GPU-a2db080d 损坏仍被调度 → GPU 预检 + 自动重提看护。

## 7. 进行中的工作

| 任务 | 状态 | 说明 |
|---|---|---|
| 续训 v3（job 9573398） | 🔄 运行中 | 从 v23 best（epoch 60）续训 40 epoch，LR=6.25e-5，64GB 内存 |
| 论文数据完整评估 | ✅ 已完成 | Table1/2/4 全部产出 |
| 非贪心全局重建方案 | ⏳ 待验证 | 用户建议：不做硬剪枝，用全部 LCA 预测 + 软权重走重建 |

## 8. 结论

- **CERN 官方 MC（v25）训练完整跑完**（100 epoch，best 0.807），重建 perfect 27.7%，但 truth 口径与论文不一致，作参考。
- **论文数据（v23）训练到 75/100 中断**，best val 0.742；严格论文口径评估 perfect 16.5%（单 B 21%），**瓶颈是 LCAG y1 类完全失效**。
- 续训进行中，若 val 突破 0.742 将补一次完整评估更新本报告。
