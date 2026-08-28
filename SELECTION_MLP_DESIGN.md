# 候选衰变链选择 MLP 设计文档

> 状态：设计定稿（2026-08-12）｜实现：`wmpgnn/reconstruction/topk_selection.py` + `train_selection_mlp.py`
> 定位：DFEI 优化方案 E（edge top-k）之后的下游选择器，与主干多任务**联合训练**

---

## 1. 动机

贪心重建（`reconstruct_decay`）无回溯：软剪枝（top-k）会产生多个候选边集，重建出多条**结构不同**的候选衰变链。现有管线把重建出的所有链都当作结果，导致背景混入（NoneIso）和错误链未被剔除。

需要一个**选择器**：对每条完整重建的衰变链打分（该链的"合理性"），选出总 likelihood 最高的链。

**与"边选择"的本质区别**：选择对象是**完整衰变链**（一棵树），而不是边集或点集。一条链的分数聚合其内部所有边/点的信息。

---

## 2. 模型结构（`CandidateScorer`）

```
输入: 单条候选衰变链
  node_feats [N, d_node]   N = 链内节点数 (随链变化)
    每行 = [node_weight] ⊕ tracks.x 物理特征
  edge_feats [M, d_edge]   M = 链内边数 (随链变化)
    每行 = [edge_weight] ⊕ LCA概率(4) ⊕ edges 物理特征
```

### 2.1 维度

| 项 | 值 |
|---|---|
| `d_node` | 1 + dim(tracks.x) = **17**（CERN use_pid 时 tracks.x 为 16 维） |
| `d_edge` | 1 + 4 + dim(edges) = **9**（edges 为 4 维） |
| 隐藏层 | 64 |

> 注意：`tracks.x` 的维度是**模型内部**维度（DFEI forward 拼接 PID 后为 16），不是原始 8 维。特征提取必须用模型输出图的特征，保证训练/推理一致。

### 2.2 前向（变长 → 定长 → 标量）

```
① node_enc (共享 MLP, 2层64):  [N, 17] → [N, 64]   每条径迹独立编码
   edge_enc (共享 MLP, 2层64):  [M, 9]  → [M, 64]   每条边独立编码

② set pooling (置换不变):
   n_mean = mean(nh), n_max = max(nh), n_sum = sum(nh)   # 各 [64]
   e_mean = mean(eh), e_max = max(eh), e_sum = sum(eh)   # 各 [64]

③ 拼接定长向量:
   z = [n_mean, n_max, n_sum, e_mean, e_max, e_sum,
        log1p(N), log1p(M)]                              # [64*6+2 = 386]

④ head (2层64→1):  z → score (标量, 该链总 likelihood)
```

**为什么用 mean+max+sum 三件套**：
- **max**：链内最"强"的节点/边（是否包含高置信信号）
- **mean**：整体置信水平
- **sum**：链的规模（大链 vs 小链）
- 三者组合对节点/边顺序不变（置换不变），因此天然支持任意 N、M

---

## 3. 训练：与主干多任务**联合训练**

### 3.1 训练样本（方案 A：truth 链 + 假链）

选择器**不单独训练**，作为 DFEI 的第 5 个监督头，loss 并入总 loss：

```
combined_loss = LCA + t_nodes + 33·tt_edges + pv_asso + λ·chain_select
```

每个训练事件构造链级样本：

| 样本 | 构造 | 标签 |
|---|---|---|
| 正样本 | 用 **truth LCA** 重建出真链（`reconstruct_decay(truth_LCA, sig_keys)`），取链内节点/边特征 | 1 |
| 负样本 | 随机抽取节点集合组成假链（避开与任何真链 node_keys 完全一致），取链内特征 | 0 |

**梯度回传路径**：
- 链成员由 truth 决定（**固定、不可微**）——这没问题，因为监督信号是"真链内的 node_weight/edge_weight/LCA 概率应高"
- 链内特征（node_weight、edge_weight、LCA 概率）来自主干输出 → **可微** → 梯度经 scorer 流回主干
- scorer 自身参数也更新

**训练时特征提取**（`shared_step` train 分支）：
```
outputs = model(batch)                       # 主干前向
node_w  = block.node_weights["tracks"]       # [N_all]
edge_w  = block.edge_weights[tt]             # [E_all]
lca     = outputs[tt].edges                  # [E_all, 4]
# 对每条 truth 链 / 假链:
chain_features(graph, node_w, edge_w, lca, edge_mask=None,
               chain_node_keys, particle_keys)   # 与推理同一函数
score = scorer(node_feats, edge_feats)       # 打分包
loss_chain = BCEWithLogits(score, label)
```

### 3.2 λ 的确定

| 任务 | 量级（v31 权重修复后） |
|---|---|
| LCA / t_nodes / tt_edges / pv_asso | ~40（逆频率权重放大） |
| chain_select（BCE） | ~0.7（未加权） |

建议 `λ = 10` 起（把 chain_select 放大到与其他任务同量级），或按 val loss 校准。**待定，可扫参**。

### 3.3 与现有任务的互补关系

- truth 链的正边 = LCA class 1/2/3 的边 → chain_select 强化"结构完整"监督，而非逐边独立
- 假链 = 随机组合 → 惩罚"把无关节点连成链"的倾向，直接对应 NoneIso 的降低
- 该任务**不改变** 4 个现有头的定义，只是新增监督信号

---

## 4. 推理集成

训练完成后 scorer 成为模型的一部分，评估时：

```
① 剪枝段 (reconstruct_heavyhadrons):
   edge_topk>0 → topk_edge_selbool 生成候选边集 (保留更多低置信真边)
   挂载剪枝后权重到图: _chain_node_w / _chain_edge_w / _chain_lca

② 重建 (reconstruct_single_evt):
   rc_dict = reconstruct_decay(reco_LCA, particle_keys)   # 多条候选链

③ 链级打分 (score_chains):
   对 rc_dict 每条链: chain_features → scorer → score
   过滤: 保留 score ≥ max(score) - selection_mlp_margin 的链
```

配置开关（config `inference` 段）：
```yaml
edge_topk: 15                  # >0: per-track top-k 边选择
selection_mlp: "scorer.ckpt"   # 非空: 启用链级选择 (或为模型内嵌时置 "builtin")
selection_mlp_margin: 5.0      # 过滤 margin (score 单位)
```

---

## 5. 数据流总览

```
训练 (联合):
  batch → DFEI_HGNN → {node_w, edge_w, lca, tracks.x}
          ├→ LCA 分类 loss
          ├→ 节点剪枝 loss
          ├→ 边剪枝 loss
          ├→ PV 关联 loss
          └→ [truth 链 + 假链] → CandidateScorer → BCE loss
  combined = 4 个原 loss + λ·chain_select → 反传

推理:
  outputs → 节点剪枝 → edge top-k → 候选边集
          → reconstruct_decay → 多条候选链
          → 每条链 chain_features → CandidateScorer → score
          → 保留高分链 → 匹配 truth 评估
```

---

## 6. 关键实现文件

| 文件 | 内容 |
|---|---|
| `wmpgnn/reconstruction/topk_selection.py` | `topk_edge_selbool`、`CandidateScorer`、`chain_features`、`score_chains`、`load_scorer` |
| `train_selection_mlp.py` | `collect`（生成链级样本+标签）、`train`（训练 scorer） |
| `wmpgnn/reconstruction/reconstruction.py` | 剪枝段 top-k 开关 + `reconstruct_single_evt` 链级打分过滤 |
| `wmpgnn/lightning_module/dfei_lightning_module.py` | **已改**：`shared_step` 增加 chain_select loss（train 分支） |

---

## 7. 待办（并入下次重训）

- [x] 1. `CandidateScorer` 挂载为 `DFEI_HGNN` 子模块（`model.chain_scorer`），随主干一起 `state_dict` 保存/加载。`DFEILightningModule` 提供只读 property `chain_scorer` 代理访问（避免重复注册）。兼容旧 checkpoint 续训：`on_load_checkpoint` 重置 optimizer/lr_scheduler 状态，`load_state_dict` 允许新头参数缺失（随机初始化）。
- [x] 2. `shared_step` train 分支：truth 链（正）+ 随机假链（负）→ scorer → BCE → 并入 combined_loss（`chain_select_loss_weight`，默认 10）。已在真实训练循环验证：loss 正常产出、梯度流向 scorer 与主干。
- [ ] 3. 推理时直接从模型取 scorer（`selection_mlp: "builtin"`），无需独立 ckpt
- [ ] 4. λ 与假链生成策略扫参；验证 CERN 数据（class 2/3 样本少，truth 链多为 1 链）

> 续训配置示例：`config_files/train_CERN_chain_select.yaml`（从 v32 best checkpoint 续训，开启第5头）
