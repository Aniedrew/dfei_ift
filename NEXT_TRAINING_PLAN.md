# 下一次训练计划：B2 可微剪枝 + 源检测头 + 链级 LCA 物理判据

> 更新：2026-08-15（v2：剔除方案 F——CERN 评估证明其扩展失控；B2 参数不再依赖 F，改为确定默认；新增"避免过硬剪枝"的定量分析与组合方案；Rumor Centrality 训练化（源检测头）确认加入）
> 前置：v32（修复版 CERN 续训）训练中（9671366）；F thr0.9 评估已完成并判定失败
> 目标：把"推理侧修补"（F/RC/链级LCA）中**可训练的部分前移进训练**，消除 train-inference gap

---

## 1. 现状梳理（含 F 失败结论）

### 1.1 已确认的事实

| 项目 | 结果 |
|---|---|
| v27（公开修复版） | Perfect **38.26%**（thr0.9），NoneIso 7.7% |
| v31（CERN 修复版，thr0.9） | Perfect **23.93%**，NoneIso **56.6%**（CERN 事件更大更难） |
| 方案 F（v31 thr0.9 + seed-expand） | Perfect **0.94%**，NoneIso **96.9%**（**失败**） |
| loss 分解（v32） | tt_edges 占 **90.9%**，t_nodes 5.8%，LCA 1.5%，pv 1.7% |
| 训练时 | **不剪枝**（全图学习）→ train-inference gap |
| node_weight 分布（CERN 实测） | mean 0.14，>0.5 占 10.1%，>0.7 占 6.4%，>0.9 占 3.3% |

### 1.2 F 失败的根因（2026-08-15 诊断结论）

在 v31 模型上实测 224 个 CERN 事件：

| 量 | 数值 |
|---|---|
| 全图节点 N | 91（中位 86） |
| 全图边 E | 10047（中位 7225） |
| 种子节点 (node_weight>0.5) | 9.2 |
| **F 扩展后保留节点** | **73.5（占全图 80%）** |
| F 保留边数 | 709（硬剪枝仅 9.6） |
| 基线硬剪枝保留节点 (>0.9) | 3.0 |

**结论**：
1. 程序无崩溃 bug，逻辑按设计执行；但 `k0=12 / hop=3 / edge_thr=0` 在 CERN 高连通图（每节点 ~100 边）上把 9 个种子扩散成 73 个节点，**等于不剪枝** → NoneIso 爆炸。
2. 方法层面：seed-expand 用**连通性**做保留标准，而 CERN 噪声图的连通性 ≈ 噪声本身。这验证了"应关心**最物理**而非最像树"的判断。
3. **弃用 F**。其可取的教训是：多保留低置信节点只会引入背景，剪枝必须伴随物理判据（链级 LCA）。

### 1.3 关于"避免过硬剪枝"的定量结论（是否只靠下调 thr？）

- thr 从 0.9 → 0.7 → 0.5，保留节点 3 → 6 → 9 个（分布见上）。**单纯下调 thr 提升有限**：多保留的节点绝大多数是背景（node_weight 分布长尾），NoneIso 不降反升（F 已证明）。
- 因此**正确组合**是"**适度放宽剪枝（thr 0.7）+ 链级 LCA 物理过滤**"：
  - 放宽剪枝 → 少误杀真链节点（降低 chain 断裂）；
  - 链级 LCA 过滤 → 剔除由低置信 LCA 边构成的噪声链（压 NoneIso）。
  - 两者互补：前者管"别剪掉真的"，后者管"别留下假的"。

---

## 2. 本次训练加入的改进（按优先级）

### 2.1 B2：可微剪枝训练（消除 gap，核心）

**做法**：训练时对 node/edge weight 施加**可微软掩码**（阈值式，不用 top-k 扩展——F 已证明 top-k 在 CERN 上失控），让消息传递在"被剪的图"上进行，梯度穿过掩码流回主干。

```
训练时每个 GN block:
  ① 算 edge_weights = σ(MLP_infer(edges))      （现有）
  ② 软掩码: mask_e = σ((edge_w - cut) / τ)      ← 阈值式软掩码 (弃用 top-k)
  ③ 节点聚合用 mask_e 加权消息传递               ← 模拟剪枝后的图
  ④ 节点特征更新 → 下一 block
  τ 随训练退火 (epoch 0: 1.0 → epoch 100: 0.1)
```

- **可微**：mask 是连续函数，梯度经 mask 流回 weight 头与主干。
- **参数（确定默认，不再依赖 F）**：
  - `b2_cut = 0.5`：与 node_weight>0.5（保留 ~10% 节点）对齐的"宽松起步"；
  - `b2_k = 0`：**弃用 top-k 语义**（F 教训），纯阈值软掩码；
  - `b2_tau_start = 1.0, b2_tau_end = 0.1`（近硬，与推理硬阈值对齐）。
- 训练观察：若 node 头过拟合（acc 高但重建差），可微调 `b2_cut` 0.3–0.7。

### 2.2 源检测头（Rumor Centrality 训练化）—— 确认加入

用户确认：RC 位于重建**最后一步**，面对的是相对干净的信息（链已生成），因此其"找根"能力可作为监督信号进训练。

**做法**：加第 6 个头，监督 GNN 预测每条 truth 链的"根节点"（B 介子候选）。

```
truth 链 → 根节点 = 链内 rumor centrality 最大的节点（truth 图结构算）
head: 节点级二分类 "该节点是否为某链的根"
loss: BCE，权重 w_source
```

- 让主干显式学"衰变链的根-叶结构"，直接服务 LCA 结构分类；
- 与推理侧 RC（找根）对齐：训练时学找根，推理时 RC 用根（干净链上判根更准）；
- 根标签生成复用 `chain_center_score`（truth 图上算，无噪声）。

### 2.3 链级 LCA 物理判据（推理侧，配合 2.1）

"最物理"判据已实现（`filter_chains_by_lca` + `chain_lca_record`）。训练后评估矩阵中作为主过滤手段：

- `chain_lca_conf_thr`：链内边平均非背景 LCA 置信度下限（建议先扫 0.0/0.3/0.5）；
- `chain_lca_class2_thr`：链内 class2（同 B）边占比下限（可空）。
- 训练时无需此模块（纯推理后处理），但**可与 B2 叠加验证**：B2 提升打分 → 链内 LCA 置信度更可信 → 过滤更有效。

### 2.4 Loss 再平衡（node/LCA 梯度欠投入）

```
combined = LCA + w_node·t_nodes + 33·tt_edges + w_pv·pv_asso + w_source·source_loss
初值: w_node = 10, w_lca = 10, w_source = 5
     (把 node/LCA/source 的梯度占比提到 ~20-30% 量级)
```

### 2.5 LCA weights clip（修复 inf 隐患）

CERN 数据 LCA class 2/3 样本为 0 → `transform_pos_weight` 产出 **inf 权重** → CrossEntropyLoss inf（FT 已修，LCA 未修）。修复：对 LCA 权重 `clip(max=1e3)`，inf→1.0 兜底（与 FT 一致）。

---

## 3. 训练起点与流程

```
起点: v32 best checkpoint (修复版, 待 v32 训练完成后取其最终 best)
流程:
  1. 实现 B2 软掩码 (GN block 改造, 阈值式) + 源检测头 + loss 再平衡 + LCA clip   ← 本次计划主任务
  2. 提交 train_CERN_next.yaml (resume_ckpt = v32 final best, lr 5e-5 起步)
  3. 训练后评估矩阵 (见 §4), 重点: 基线 thr0.9 / thr0.7+chain_lca_filter / +B2 输出对比
```

### 3.1 实现前置清单

| 文件 | 改动 |
|---|---|
| `wmpgnn/model/gnn/hetero_graph_network.py` | forward 中节点聚合前施加**阈值式**可微软掩码（`b2` 开关 + 温度退火，`b2_cut` 参数） |
| `wmpgnn/lightning_module/dfei_lightning_module.py` | 源检测头（truth 链根监督）+ `shared_step` 加入 `source_loss`；loss 权重参数 |
| `wmpgnn/data_loader/weights_calculator.py` | LCA 权重 `clip(max=1e3)`，inf→1.0 |
| `wmpgnn/reconstruction/topk_selection.py` | 复用 `chain_center_score` 生成 truth 链根标签（训练时调用） |

---

## 4. 评估计划（训练后）

```
对比矩阵 (同一训练好的模型, 只改推理配置):
  A. 基线 (thr 0.9)                          ← 对照
  B. 放宽剪枝 (thr 0.7)                       ← 降低真链误杀
  C. B + chain_lca_filter (conf 0.3/0.5 扫描)  ← 链级物理判据 (主过滤)
  D. + B2 训练 (新模型)                        ← 训练侧改进 (打分更可信 → 过滤更有效)
  E. D + source_head 叠加验证
指标: per-chain Perfect/NoneIso, per-event Perfect, 候选边数
```

---

## 5. 下一步行动顺序

1. **实现代码**（§3.1 清单，4 个文件）
2. **v32 训练完成后**取最终 best → 填 resume_ckpt
3. **提交 train_CERN_next.yaml**（B2 + 源检测头 + loss 再平衡 + LCA clip）
4. **训练后评估矩阵**（§4），主验证点：B2 是否提升打分 → chain_lca_filter 是否更有效
