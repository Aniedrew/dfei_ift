# DFEI 版本演进记录（CERN 训练线）

> 用途：快速区分每个 version_XX 是干什么的、有什么改动、结果如何。
> 数据口径：除注明外，均为 CERN 官方 MC（DFEI_IFT_20260702），thr0.9 同口径评估，20 测试文件。
> 更新：2026-08-18（v38 训练中）

---

## 一、版本总览（速查）

| 版本 | 一句话 | 起点 | 关键改动 | best val | thr0.9 Perfect |
|---|---|---|---|---|---|
| v23 | 早期：public paper 复现 | 从头 | 原始训练流程 | — | 19.8%（paper 数据，不同口径）|
| v25 | 早期：CERN 首次训练 | 从头 | 同上 | — | ~25.8%（smoke 口径）|
| v30 | 早期：CERN（bug 前） | 从头 | 同上 | — | — |
| **v31** | **基线（修 class weight bug 后）** | 从头 | LCA__weights→LCA_weights | 36.221 | **23.93%** |
| v32 | 微调 | v31 ep56 | lr 1e-3→1e-4 | 36.152 | — |
| v33 | B2 代码 smoke 验证 | — | 2 epochs 试跑 | — | — |
| v34 | 微调（失败，被 v35 取代） | v32 ep61 | 同 v35 配置 | — | — |
| v35 | 微调（成功） | v32 ep61 | resume 重跑 | 35.955 | — |
| v36 | **B2 可微剪枝 + 源检测头** | v35 ep68 | b2_cut 0.5 + source_head | 35.577 | **26.28%** |
| v37 | **方案4/5/6：class2加权+chain_lca_loss+b2_cut0.7** | v36 ep79 | lca_class2_weight 3.0 | 35.562 | **27.25%** |
| v38 | **456 改良：b2_cut0.85+cl2加权2.0+chain_lca_ce** | v37 ep90 | 见下文 | 35.860 | **29.26%** |
| v39 | PV 分簇（truth 硬切，训练侧试验） | v38 | truth 分簇 + EarlyStopping 提前结束 | — | — |
| v40 | PV 分簇：可训练 cluster 头（训练侧切子图） | v38 ep101 | 见下文（发散失败） | 84.9（发散） | 28.79%（发散权重） |
| v41 | PV 分簇（课程过渡修复版，200files） | v38 ep101 | 3.2h/epoch + val 不降被停 | 125.8（不降） | — |
| v42 | PV 分簇：小规模快速验证（50files/48ep） | v38 ep101 | 见下文 | 118.2 | **26.28%（伤模型）** |

---

## 二、每个版本详细说明

### 早期版本（v23 / v25 / v30）—— 基线探索期
- 统一配置：lr 1e-3，100 epochs，从头训练，**class weight bug 未修复**。
- **v23**：public paper 数据复现，Perfect 19.8%（paper_thr02 口径，NoneIso 12.6% 是因为评估阈值/口径与后续不同，**不可直接对比**）。
- **v25**：CERN 官方 MC 首次训练，Perfect ~25.8%（chain_smoke 口径，样本小）。
- **v30**：CERN 早期训练（bug 前），无有效评估结果（评估曾因 GPU 问题重试）。

### v31 —— ⭐ 修 bug 后的真正基线
- **改动**：class weight 配置 bug 修复（`LCA__weights` → `LCA_weights`，之前拼写错误导致 LCAG class1 准确率掉到 0%）。
- 从头训练，lr 1e-3 / 100 epochs / best ep56（val 36.221）。
- **结果（thr0.9 基线）**：Perfect 23.93%，AllParticles 43.42%，NoneIso 56.58%。
- 后续所有对比都以它为准。

### v32 —— 微调
- resume v31 ep56，lr 降到 1e-4，150 epochs。best ep61（val 36.152）。

### v33 —— B2 代码 smoke 验证
- 仅 2 epochs 试跑（chain_train_smoke），验证 B2 可微剪枝代码能跑通，非正式训练。

### v34 —— 微调（失败版）
- resume v32 ep61，配置与 v35 相同。训练异常/被取代，无有效结果。

### v35 —— 微调（成功版）
- resume v32 ep61，v34 的同配置重跑（v34 失败）。best ep68（val 35.955）。
- 评估阶段曾遇 GPU 驱动错误，训练本身成功。

### v36 —— ⭐ B2 可微剪枝 + 源检测头
- **resume v35 ep68**，lr 5e-5。
- **B2 软掩码**：训练时对 node/edge weight 施加 `σ((w-cut)/τ)` 软掩码 + 温度退火（tau 1.0→0.1），模拟推理剪枝，消除 train-inference gap。b2_cut 0.5。
- **源检测头（第6头）**：Rumor Centrality 训练化——监督 GNN 预测每条 truth 链的根节点，对齐推理侧 RC 找根。
- best ep79（val 35.577）。
- **结果（thr0.9）**：Perfect 26.28%（+2.35pp），All 49.24%，NoneIso 50.76%。
- 首个明确有效的优化版本（比 v31 大提升）。

### v37 —— ⭐ 方案4/5/6：class2 加权 + 链级 LCA 一致性 + 更狠剪枝
- **resume v36 ep79**，lr 5e-5。
- **方案4**：`lca_class2_weight: 3.0` —— class2（同B边）专项加权，针对结构瓶颈。
- **方案5**：`chain_lca_loss` —— 链内 LCA 一致性辅助损失（hinge，鼓励链内边高置信，chain_lca_filter 训练化）。
- **方案6**：`b2_cut: 0.5 → 0.7` —— 训练时剪更狠，更接近推理硬阈值。
- best ep90（val 35.562）。
- **结果（thr0.9）**：Perfect 27.25%（+0.97pp vs v36），All 50.60%，NoneIso 49.40%（**首次破 50%**）。
- 注意：class2 准确率反而略降（v36 47.7%→v37 44.2%），class1 大幅提升（64.5%→69.1%）→ 提示 3.0 加权过度。

### v38 —— 456 改良版
- **resume v37 ep90**，lr 3e-5（更保守）。
- **方案4 改良**：`lca_class2_weight: 3.0 → 2.0`（v37 实验证明 3.0 过度）。
- **方案6 改良**：`b2_cut: 0.7 → 0.85`（对齐推理 thr0.9，收窄 train-inference gap）。
- **方案5 升级**：`chain_lca_ce: true` —— 在"高置信"hinge 基础上，新增链内边**类别正确性 CE**（只监督 y>0 的结构边，对抗 class0 对主 LCA loss 的稀释）。
- **结果（thr0.9，ep101）**：Perfect **29.26%**（+2.0pp vs v37），LCAG class1 76.8%、class2 47.9%。

### v39 —— PV 分簇（truth 硬切，训练侧试验，未达预期）
- 方向：事件内 tracks 按 PV 分簇 → 每簇独立链重建 → 汇总。单图节点从 91-139 降到每簇 20-30，缓解高连通图跨链干扰。
- 训练侧先用 **truth PV 硬切**（`_split_by_pv` 用 y==1 的 track-pv 边分组），`pv_cluster_assign: pred`（推理侧用 pv_asso 头）。
- **结果**：EarlyStopping 提前结束（15 epoch 无改善）——预期行为，因为当时 cluster 只是推理侧改动、训练侧分簇器不可训练，模型学不到与推理一致的分簇结构。

### v40 —— PV 分簇：可训练 cluster 头（训练侧切子图，失败）
- **resume v38 ep101**，lr 3e-5。分簇器 = `model.pv_cluster_head`（独立 MLP，BCE 监督），温度退火 Gumbel 分配切子图；推理时 reconstruction 用同一头分簇（`pv_cluster_assign: cluster_head`）。
- **失败原因**（3 个叠加）：
  1. **train/val 不对称**：只对 train 切子图，val 仍全图 → 模型在子图学的结构在全图退化，val_combined_loss 从 35.9 爆到 84.9→114→114→155（发散）
  2. **随机 cluster 头 + tau 已退火完**：resume 时 current_epoch=102 使 tau 直接=0.1，随机头确定性输出垃圾分簇，直接扰动 v38 权重
  3. **EarlyStopping 继承计数**：v37/v38 的 wait_count 继承（best 35.562 15 轮未破）→ 只训 4 epoch（102-105）就停
- **~2h/epoch**：batch 从 8 个全图炸成 ~50 子图，150 epoch ≈ 12.5 天不可行。
- 测试用了最终发散权重：Perfect 28.79%（< v38 29.26%）、LCAG class1 46.47%（v38 76.8%）。
- checkpoint 落在 version_41 目录（首次被杀的空跑占了 version_40）。

### v41 —— PV 分簇：cluster 头 + 退火 + 课程过渡 + val 对齐（被停）
- **resume v38 ep101**，lr 3e-5。v40 三件套修复：
  - **① val 也切子图**（确定性分配，无 Gumbel）→ 消除 train/val 图结构 gap（val 不再发散）
  - **② truth→cluster 课程式过渡**：alpha 1.0（truth 分簇稳定热启动，v38 权重不被随机头扰动）→ 0.0（cluster 头），30 epoch 线性，**相对本次 run 起点计**（tau 退火同样改相对 run 起点，修复 v40 tau=0 的问题）
  - **③ 训练侧子图数上限**（`pv_cluster_max_subgraphs: 24`，随机保留提速；val/test 全量）
- **EarlyStopping 重置**：子图训练 val 口径变化，on_train_start 重置 best/wait_count，只跟本次 run 比（防止继承 v37/v38 计数误停）。
- 推理分簇仍用同一 cluster 头（`pv_cluster_assign: cluster_head`），单 B 处理路径完全不变。
- **结果**：代码正确（0 WARN、课程过渡生效、EarlyStopping 已重置），但跑在 gpu02（数据加载受限）+ **3.2h/epoch**（150 epoch ≈ 20 天不可行），且 val 震荡不降（131→121→134→126）。epoch 102-105 后手动停止。checkpoint 落 version_42 目录。

### v42 —— PV 分簇：小规模快速验证（结论：子图训练伤模型）
- **resume v38 ep101**，lr 3e-5。与 v41 同代码，仅缩规模：nfiles 200→50、epochs 150→50、tau 退火 100→40（匹配 50 epoch）。
- 目的：快速回答两个问题——①子图训练能否让 val 下降 ②推理分簇（cluster 头）能否提升 Perfect（vs v38 29.26%）。
- **结果（version_44）**：48 epoch（102-149），val 130.9→118.2（课程过渡+val 对齐生效，不再发散，速度 ~50min/epoch）；但测试 Perfect **26.28%**（vs v38 29.26%，**-2.98pp**），LCAG class1 **56.4%**（v38 76.8%，严重退化）。
- **结论：训练侧切子图伤模型**——测试时 GNN 前向仍跑全图（重建流程：全图前向→剪枝→分簇→簇内重建），子图训练让全图前向能力退化（class1 掉 20pp）。train/infer 的 gap 在于「训练切子图 vs 测试全图前向」，不是 val 对齐能解决的。
- 后续：**推理分簇单独验证完成（2026-08-24）**：v38 ckpt + pv_asso 头分簇，20 文件 10038 事件、0 回退，Perfect **28.70%**（vs v38 基线 29.26%，无显著差异，-0.56pp）。1B 34.18%、2B 20.98%，part_reco 异常低（3.29%，簇间链合并问题）。
- **最终结论：PV 分簇路线整体关闭**（训练侧伤模型 + 推理侧无效）。v38 全图方案（29.26%）为当前最优，后续优化另寻方向。

---

## 三、关键指标对比（thr0.9 同口径）

| 指标 | v31 基线 | v36 | v37 | v37−v31 |
|---|---|---|---|---|
| PerfectReco | 23.93% | 26.28% | **27.25%** | **+3.32pp** |
| AllParticles | 43.42% | 49.24% | 50.60% | +7.18pp |
| NoneIso | 56.58% | 50.76% | 49.40% | −7.18pp |
| LCAG class1 | 67.78% | 64.51% | 69.14% | +1.36pp |
| LCAG class2 | 41.26% | 47.66% | 44.22% | +2.96pp |
| LCAG class3 | 75.43% | 78.32% | 77.17% | +1.74pp |
| 2B 事件 Perfect | 15.22% | — | 20.24% | **+5.02pp** |

---

## 四、相关实验 / 评估（非正式版本）

- **方案 F（seed-expand 连通性保留剪枝）**：从种子节点扩展 top-k 边。在 CERN 高连通图上失控（9→73 节点，约等于不剪枝），Perfect 仅 ~0.9%。**方法不成立，已弃用**（程序逻辑正确）。
- **chain_lca_record 评估（v36）**：记录每条链的 LCA 置信度（不过滤）。阈值扫描结论：**chain_lca_conf / class2_frac 单判据和联合判据均无效**——真链/假链置信度分布重叠大（Perfect 链 med 0.635 vs 失败链 0.542），任何阈值都同步误杀真链。待 v37/v38（训练了 chain_lca_loss/ce）的 record 数据再验证。
- **edge top-k / 选择 MLP（方案E）**：per-track top-k 边选择 + 候选链打分（CandidateScorer），代码已实现但 scorer 未正式训练启用。

---

## 五、训练速度备注

| 版本 | 速度 | epoch 时间（2080Ti） |
|---|---|---|
| v31 | 11.78 it/s | ~45 min |
| v37 | 10.43 it/s | ~51 min（+13%，额外 loss 头代价）|

GPU 分配注意：调度器会把多个作业塞同一物理卡（GPU_NOTE.md 有详查）；提交脚本已带 PREFLIGHT matmul 检测 + 失败自动重排（不指定节点）。
