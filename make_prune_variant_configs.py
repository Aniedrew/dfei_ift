"""生成"剪枝优化"训练配置 (都从 v601 权重出发, 0904 生产), 并可直接提交。

注意: epochs/early_stop_patience/log_version 在 `settings` 段 (不是 `train`; 配置里没有 train 段),
      新加的损失开关在 `inference` 段 (lightning module 读的是 configs["inference"])。

v610-v613 (2026-09-22, 已提交): 边/点剪枝的权重与阈值再平衡
  v610 = 只把 epochs 20 -> 60 (欠训练假设)
  v611 = 20ep + OHEM 最难负边 2%
  v612 = 60ep + OHEM 2% + pos_weight ×3 (决策边界偏 precision)
  v613 = 全上: 60ep + OHEM 2% + pos_weight ×3 + edge_prune_weight 100 + 评测 thr0.95 + edge_topk16

v614/v615 (2026-09-22 晚, 基于当天预检的修正): pairwise ranking
  预检结论 (v601/0904, 60 事件):
    - 结构先验 support(i,j)=max_k min(s_ik,s_kj) 作为正则: 真边 vs fake_inter AUC 0.53 (噪声) -> 放弃
    - 换分数来源 (LCA 头 / 双头 AND 门) 也无增益 -> 放弃
    - 真问题: 在真正的决策 population (两端都过点剪枝 thr0.9) 上, 边头 thr0.9 时保留 ~90% 的边、
      precision 只有 0.295 (= 基频); 要走到 R90 工作点 (thr≈0.996) 才有 precision 0.73。
      该 population 的假阳性 76% 至少一端是背景径迹, 24% 是跨链 —— 缺的都是"信号样"边之间的排序。
      根因: BCE 的 pos_weight≈700 把梯度几乎全投在"推高真边", 对负边之间的次序约束很弱。
  故 v614/v615 加 edge_rank_loss (每事件 top-N 最难负边 vs 全部真边的 pairwise hinge):
    v614 = 20ep + rank(w=10, nneg=64, margin=1.0)     <- 与 v601 (20ep, 无) 单变量对照
    v615 = 60ep + rank + OHEM 2%                      <- "都用上"版, 与 v613 对照
  注意: 排序损失只学相对次序 -> 评测阈值必须按 ROC 重扫, 不能沿用 0.9。
"""
import argparse
import copy

import yaml

BASE = "config_files/train_CERN_0904_v601.yaml"
INIT = ("/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/LHCb_logs/DFEI/"
        "version_601/checkpoints/best-epoch=19-val_combined_loss=130.272.ckpt")

VARIANTS = {
    "v610": dict(epochs=60, early_stop_patience=8, tag="v610_long60"),
    "v611": dict(epochs=20, early_stop_patience=5, tag="v611_ohem", edge_ohem_frac=0.02),
    "v612": dict(epochs=60, early_stop_patience=8, tag="v612_ohem_pw3",
                 edge_ohem_frac=0.02, edge_pos_weight_scale=3.0),
    "v613": dict(epochs=60, early_stop_patience=8, tag="v613_allin",
                 edge_ohem_frac=0.02, edge_pos_weight_scale=3.0, edge_prune_weight=100.0,
                 thr=0.95, edge_topk=16),
    "v614": dict(epochs=20, early_stop_patience=5, tag="v614_edge_rank",
                 edge_rank_weight=10.0, edge_rank_nneg=64, edge_rank_margin=1.0),
    "v615": dict(epochs=60, early_stop_patience=8, tag="v615_rank_ohem60",
                 edge_rank_weight=10.0, edge_rank_nneg=64, edge_rank_margin=1.0,
                 edge_ohem_frac=0.02),
    # ---- 2026-09-23: delta_z0 方向 (leak 修复 / 可测量定向 / 剪枝 MLP 方向头) ----
    # 反事实结论: 7 月模型边头实质是"delta_z0 符号的单边判据"(抹掉符号 0.9996->0.77~0.86),
    # 而那个符号来自 np.sort(ParticleIndex) 这个真值相关的人工顺序; 0904 训的模型只掉 0.04~0.09。
    # 四个变体分别试: 随机化方向 / 用可测量 minIP 定向 / 剪枝 MLP 多学一个方向头 / 全上。
    "v616": dict(epochs=20, early_stop_patience=5, tag="v616_dzflip", dz_dict=True,
                 edge_dz_flip_prob=1.0),
    "v617": dict(epochs=20, early_stop_patience=5, tag="v617_ipcanon", dz_dict=True,
                 edge_dz_ip_canon=True),
    "v618": dict(epochs=20, early_stop_patience=5, tag="v618_dirhead", dz_dict=True,
                 edge_dz_flip_prob=1.0, dir_head_weight=10.0, gn=dict(MLP_infer_dir_head=True)),
    "v619": dict(epochs=20, early_stop_patience=5, tag="v619_dz_allin", dz_dict=True,
                 edge_dz_ip_canon=True, dir_head_weight=10.0, gn=dict(MLP_infer_dir_head=True)),
    # ---- 2026-09-23 晚: 把两个"真赢家"叠起来 (v614 的 ranking + v617 的可测量定向) ----
    # 依据 (同 20 文件 / thr 0.9, 固定分母计数 A=All%*N/100, 基线 v601=1915.9):
    #   v614 ranking          +12.7%   |  v617 minIP 定向 +1.4%   |  v616 方向随机化 -11.5%
    #   v611/612/615 OHEM     -12~-16% (该线放弃)
    "v620": dict(epochs=20, early_stop_patience=5, tag="v620_rank_canon", dz_dict=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64),
    "v621": dict(epochs=60, early_stop_patience=8, tag="v621_rank_canon60", dz_dict=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64),
    # ---- 2026-09-23 深夜: 剪枝阶段的三条线 (输入 / 头与目标 / 传递性) ----
    # v622 = "1+3 组合": 重心转到点 (方案A 上下文头 + 节点侧 ranking) + 边侧 ranking 保留。
    # v623 = 派生输入 (节点 7 维 / 边 7 维物理派生量), 经零初始化适配器注入剪枝 MLP。
    # v624 = 派生输入 + 三角传递性 (边 9 维) + 打开 chain_lca_filter。
    "v622": dict(epochs=20, early_stop_patience=5, tag="v622_point_rank", dz_dict=True,
                 node_rank_weight=10.0, node_rank_nneg=64,
                 edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(context_prune=dict(mode="attn", heads=4, hidden=32, refine_edge=True))),
    "v623": dict(epochs=20, early_stop_patience=5, tag="v623_derived", dz_dict=True,
                 derived_prune=True, gn=dict(extra_node_dim=7, extra_edge_dim=7)),
    "v624": dict(epochs=20, early_stop_patience=5, tag="v624_derived_tri", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 gn=dict(extra_node_dim=7, extra_edge_dim=9)),
    # ---- 2026-09-24: 定向方式的第三个代理量 + "不用方向"的诚实基线 ----
    # v625 = yukai 的 zPV 排序定向 (minIP 关联 PV 的 z 更小者在前); 与 v617 (minIP 大小) 对照。
    # v626 = 只保留 |Δz| 不给方向 (对应他的 delta_z_mode="abs"); 用来证明"不用方向也追得上"。
    "v625": dict(epochs=20, early_stop_patience=5, tag="v625_dz_pvz", dz_dict=True,
                 edge_dz_pvz_canon=True),
    "v626": dict(epochs=20, early_stop_patience=5, tag="v626_dz_abs", dz_dict=True,
                 edge_dz_abs=True),
    # ---- 2026-09-24: 四条"正交维度"的方向 (分) + 一条全开 (总) ----
    # 判据: 之前所有手段 (全局阈值 / node 阈值 / per-track topk / delta_z 定向 / ranking) 都在**同一条
    #       precision-recall 曲线**上滑, 所以互相可替代、叠加收益有限。这四条换的是
    #       **信息 / 目标函数 / 决策自由度**, 目标是抬高曲线本身。
    # 依据: per_event_thr_oracle.py (v601, 300 事件): 2-B 最优 thr=0.99 vs 1-B 0.95~0.96;
    #       事件自适应阈值 oracle 上界 +1.4~+3.4pp (All_fix) -> 事件级信息确有增量, 需要模型自己学。
    # v627 = A2: 把 PV 关联头的"软同 PV 重叠"喂进 tt 剪枝 (打通"跨 PV 的边"这条通道)
    # v628 = C1: 事件级自适应剪枝偏置 (端到端学, 无标签)
    # v629 = B1: 链级对比损失 (在嵌入空间按真值链拉近/推远 -> 换目标函数)
    # v630 = B2a: 事件级链数辅助头 (迫使事件级表征编码"这事件有几条链")
    # v631 = 总: 四条一起 + 已验证的 ranking(10) + minIP 定向
    "v627": dict(epochs=20, early_stop_patience=5, tag="v627_pvoverlap",
                 gn=dict(pv_overlap_inject=True)),
    "v628": dict(epochs=20, early_stop_patience=5, tag="v628_evtbias",
                 gn=dict(event_bias=True)),
    "v629": dict(epochs=20, early_stop_patience=5, tag="v629_chaincon",
                 chain_contrastive_weight=0.5, chain_contrastive_tau=0.1),
    "v630": dict(epochs=20, early_stop_patience=5, tag="v630_evtcount",
                 gn=dict(event_count_head=True), event_count_weight=2.0),
    "v631": dict(epochs=20, early_stop_patience=5, tag="v631_allin_new", dz_dict=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(pv_overlap_inject=True, event_bias=True, event_count_head=True),
                 chain_contrastive_weight=0.5, event_count_weight=2.0),
    # ---- 2026-09-24: 「当前最好版本」总装 + 长训 (v632) ----
    # 挑选依据 (0904 官方 20 文件, 固定分母 13255):
    #   有效 (纳入):
    #     edge ranking (v614):  thr0.9 14.45->16.29 (+12.7%); thr0.95 -> 20.45 (全口径最好)
    #     minIP 可测量定向 (v617/v620): +1.4% 单独, 与 ranking 叠加 +16.9%
    #     node_prune_thr 0.95~0.97: v601 14.45->17.90->18.46 (最大单一旋钮, 评测侧; 本配置自带 test 也用它)
    #     训练长度 (v610 60ep vs v601 20ep): +2% -> 本配置给 80ep/patience 15
    #   无效/有害 (排除):
    #     edge_topk: 同阈值下比无 topk 低 1.1pp (k8/16/32 在 thr0.9 完全一样) -> 去掉
    #     OHEM / edge_pos_weight_scale: -12~-16% -> 去掉
    #   正交但尚在验证中 (用户要求"都加一点", 各按温和权重纳入, 全部零初始化/默认关则等价旧模型):
    #     pv_overlap_inject (跨 PV 信息) / event_bias (事件自适应) / event_count_head (链数辅助)
    #     / chain_contrastive (链级目标)
    "v632": dict(epochs=80, early_stop_patience=15, tag="v632_best", dz_dict=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(pv_overlap_inject=True, event_bias=True, event_count_head=True),
                 chain_contrastive_weight=0.5, event_count_weight=2.0,
                 thr=0.95),
    # ---- 2026-09-25 层2: v633 = 只留有效项 + 唯一被证明抬高曲线的成分 ----
    # v631(19.77) 中 event_count_head (v630 单测 -19%) / pv_overlap_inject (v627 单测 -2.6%) 已证负 -> 剔除;
    # 加入 v624 的派生输入 + 三角传递 (benchmark 唯一在所有指标上同时抬高: edge AP +67%, p@r90 x2.2, node AP +16%)
    # 及 chain_lca_filter; 保留 ranking + minIP 定向 + event_bias + 链对比 (v628/v629 微正)。
    "v633": dict(epochs=60, early_stop_patience=15, tag="v633_clean_stack", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9),
                 chain_contrastive_weight=0.5, thr=0.95),
    # ---- 层3a: v637 = v633 + line-graph 边-边注意力 (唯一能结构性攻 fake_inter 的手段) ----
    "v637": dict(epochs=60, early_stop_patience=15, tag="v637_linegraph", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9,
                         line_graph_attn=True, line_graph_rounds=1, line_graph_heads=4,
                         line_graph_hidden=32),
                 chain_contrastive_weight=0.5, thr=0.95),
    # ---- 层3b/3c: v635 = v633 + listwise(InfoNCE) 顶部排序; v636 = v633 + 次级顶点一致性特征 ----
    # 依据: 边 AUC ~0.97 但 AP 仅 0.29 -> 瓶颈在顶部难负例之间的次序, listwise 把梯度集中到候选集;
    #       fake_inter(跨链) 实为"同 PV、不同次级顶点" -> 用两径迹公共垂足 z / 飞行距离 / 共线性替代 IP-PV 家族。
    "v635": dict(epochs=60, early_stop_patience=15, tag="v635_listwise", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 edge_rank_mode="infonce",
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9),
                 chain_contrastive_weight=0.5, thr=0.95),
    "v636": dict(epochs=60, early_stop_patience=15, tag="v636_vertex", dz_dict=True,
                 derived_prune=True, derived_triangle=True, derived_vertex=True,
                 chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=12),
                 chain_contrastive_weight=0.5, thr=0.95),
    # ---- 2026-09-26 v639 = v633 配方 + 四处 bugfix (单变量: 只带修复, 不动结构/超参) ----
    # 修复内容: (1) weights_calculator 里 pv_asso 的 pos_weight 用 `=` 覆盖 -> 改成 `+=` (PV 关联头类别平衡);
    #           (2) chain_contrast 损失落日志 (此前不可见); (3) 派生特征算失败在 train 模式 raise (不再静默降级);
    #           (4) 训练自带 test 改用 best ckpt (与下游 eval 口径统一; 注意: 自报数字因此变成 best-ckpt 口径)。
    "v639": dict(epochs=60, early_stop_patience=15, tag="v639_fix_pvasso", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9),
                 chain_contrastive_weight=0.5, thr=0.95),
    # ---- 2026-09-29 v640 = v633 配方, 唯一区别: best ckpt / 早停的监控指标换成剪枝 AP ----
    # 动机: val_combined_loss 是 8 个任务的加权和 (edge 项占 65-70%), 与"剪枝 AP"的排序不一致,
    #       选 ckpt 用的其实是另一把尺子 (v614: min-val ckpt 13.81 vs 末轮 16.32, 白丢 2.5pp)。
    #       把监控指标切到验收判据 (边剪枝 AP) 上, 与 v633 构成单变量对照。
    "v640": dict(epochs=60, early_stop_patience=15, tag="v640_apmon", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9),
                 chain_contrastive_weight=0.5, thr=0.95,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=200)),
    # ---- 2026-09-29 v641 = v640 + 链级 min-pooling recall 损失 (AND / 指数语义) ----
    # 动机: "一个环节掉下去整条链就断" 是乘性 (长 n 链存活 = p^n), 逐边 BCE 优化的是平均
    #       正确率, 与链级目标不一致。chain_recall_loss 直接罚每条真值链里**最弱**的点/边,
    #       是"指数语义"在 loss 侧的对应物 (此前 chain_recall_weight=0, 从未启用过)。
    # 参数: thr 取 0.9 与推理剪枝阈值对齐 (该 loss 默认 0.5, 与决策点不匹配); tau=0.1 保持锐利;
    #       权重 5.0 为探索值 (该项量级 ~0.7-7, 相对 combined ~125 属"有意义但不主导")。
    "v641": dict(epochs=60, early_stop_patience=15, tag="v641_chainrecall", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9),
                 chain_contrastive_weight=0.5, thr=0.95,
                 chain_recall_weight=5.0, chain_recall_edge_weight=5.0,
                 chain_recall_thr=0.9, chain_recall_tau=0.1,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=200)),
    # ---- 2026-09-30 v642 = v641 的公平重跑: 加早停 dead-band ----
    # v641 停在 ep15 不是因为收敛, 而是难池 AP 在 ep0 抽到 0.777 的高点后一直没超过 ->
    # patience 15 触发 (per-epoch 抖动 ~±0.03)。这里 patience 20 + min_delta 0.005,
    # 让"不改善"必须是真的往下走, 而不是噪声。
    "v642": dict(epochs=60, early_stop_patience=20, tag="v642_chainrecall_db", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9),
                 chain_contrastive_weight=0.5, thr=0.95,
                 chain_recall_weight=5.0, chain_recall_edge_weight=5.0,
                 chain_recall_thr=0.9, chain_recall_tau=0.1,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    # ==== 2026-09-30 层4: 依据"特征上界分析"(docs/feature_ceiling_analysis.md) 的三条平行臂 ====
    # 该分析(GDBT, 按事件留出)结论: 现有 5+9+3 维边特征对"真边 vs 跨链假边"判别力 = AUC 0.502
    # (随机); 而 (a) 两端生产顶点 3D 距离 |Δr| 单标量 = 0.704, (b) 正确 DOCA 在节点特征之上
    # 再 +0.06 AUC (=0.738); 训练好的 GNN 同一任务只有 0.63 (=模型没用足节点信息)。
    # 三条臂都以 v633 为底, 均把 ckpt/早停监控切到验收判据 val_prune_ap (与 v642 同口径,
    # 保证四臂之间互比干净), 而不与 v633 的 val_combined_loss 口径混比。
    # ---- v643 (A1): 顶点一致性**几何绝对量** [doca/100, log(doca), |Δ起点|/100] (+3 维) ----
    # 旧 derived_vertex 只给比值 d_perp/(|Δz|+1) -> 绝对量级被除掉, 实测贡献≈0; 这里给绝对量。
    # 用标准最小二乘解**现算**, 绕开上游 calculate_doca 的 t1 符号 bug(那列 log_DOCA_reco 是噪声)。
    "v643": dict(epochs=60, early_stop_patience=20, tag="v643_vgeom", dz_dict=True,
                 derived_prune=True, derived_triangle=True, derived_vertex_geom=True,
                 chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=12),
                 chain_contrastive_weight=0.5, thr=0.95,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    # ---- v645 (A3): 两端节点特征**对称**直连到边 [x_i+x_j, |x_i-x_j|, x_i*x_j] (+24 维) ----
    # 动机: 判别信息几乎全在节点侧, 而 GNN 没用足; 给边头一条直接的端点通道。
    # 对称性: 用和/差/积 (天然交换不变), **不用** [x_i, x_j] 顺序拼接 (要靠人为定向, IP 相等时
    #   退化为按下标排序 -> 会像 0702 的 delta_z0 那样带进顺序信息)。
    "v645": dict(epochs=60, early_stop_patience=20, tag="v645_pairsym", dz_dict=True,
                 derived_prune=True, derived_triangle=True, derived_pair_sym=True,
                 chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=33),
                 chain_contrastive_weight=0.5, thr=0.95,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    # ---- v646 (A4): **只训剪枝** (纯配置) -> 检验"多任务干扰"假说 ----
    # 关掉全部非剪枝任务损失: LCA 分类 / 链 LCA / 链对比 / PV 关联; 保留 node+edge 剪枝 + ranking
    # (ranking 属于剪枝头)。若剪枝曲线因此显著抬高, 说明此前是被其它 6 个任务拖住的。
    "v646": dict(epochs=60, early_stop_patience=20, tag="v646_pruneonly", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9),
                 chain_contrastive_weight=0.0, thr=0.95,
                 lca_weight=0.0, chain_lca_loss_weight=0.0, pv_asso_weight=0.0,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    # ---- v647 (总): A1 + A3 同时上 (doca/logdoca/|Δ起点| + 端点对称直连, +27 维) ----
    # ---- v648 (P1): v643(修正 DOCA 几何) + 竞争/排他上下文 (+6 维, 共 18 维) ----
    # 依据: 大样本探针里 deg_s(+0.0235) 是仅次于 doca(+0.1066) 的特征 -> 先把这条手工版本
    #   拿到主模型上验证, 作为"注意力/匹配"结构方案的**下界对照**。
    "v648": dict(epochs=60, early_stop_patience=20, tag="v648_geo_comp", dz_dict=True,
                 derived_prune=True, derived_triangle=True,
                 derived_vertex_geom=True, derived_comp=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=18),
                 chain_contrastive_weight=0.5, thr=0.95,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    # ---- v652 (A4a) = v648 + derived_vertex (次级顶点一致性 z_assoc/zcpa/flight/collin, +3 维 -> 21) ----
    # 依据 (2026-10-09): 修掉分析脚本的 D1 索引错位后, **多 B 子集**的 GBDT 探针里
    #   `flight` 排第 3 (+0.0476, 仅次于 doca +0.1663 / trdist +0.0727) ->
    #   值得在"当前最优底子"上把这条分支重新验证一次。
    # 背景/风险: 早期 200 事件的混合池里 derived_vertex 贡献≈0 (v636 难池 AP 0.6423 < v633 0.6647),
    #   当时的解释是它给的是比值量 d_perp/(|Δz|+1), 绝对量级被除掉; 但 v643 的绝对量版本
    #   (derived_vertex_geom) 在 500 事件上也是 +0.000。所以本臂是**在多 B 判据下**的复检,
    #   而不是"新发现"。单变量: 与 v648 只差 derived_vertex。
    "v652": dict(epochs=60, early_stop_patience=20, tag="v652_geo_comp_vertex", dz_dict=True,
                 derived_prune=True, derived_triangle=True,
                 derived_vertex_geom=True, derived_comp=True, derived_vertex=True,
                 chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=21),
                 chain_contrastive_weight=0.5, thr=0.95,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    "v647": dict(epochs=60, early_stop_patience=20, tag="v647_vgeom_pairsym", dz_dict=True,
                 derived_prune=True, derived_triangle=True,
                 derived_vertex_geom=True, derived_pair_sym=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=36),
                 chain_contrastive_weight=0.5, thr=0.95,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    # ==== 2026-10-05 层5: 把"几何量直接耦合到注意力"搬进主模型 (v637 的单变量补丁) ====
    # 依据: (1) bench 200 事件里 v637 (line-graph 边-边注意力) 难池 AP 0.6553 < v633 0.6647
    #           -> 纯结构注意力没帮上忙; (2) 小模型探针里唯一值得搬的差异是 Graphormer 式的
    #           **几何 pair-bias** (把几何量直接加到 attention logit 上, 而不是当第 N 个输入列)。
    # ---- v649 = v637 + 几何 pair-bias (严格单变量: 除 bias 外与 v637 逐键一致) ----
    # bias 列 = der_edges 的列 0/4/6 = dR(两端生产顶点 3D 距离) / ΔIP / dzp(沿合动量的纵向分离),
    #   这是 v633 那 9 维里真实存在的几何量 (doca 只在 derived_vertex_geom 的 v643/v648 里, 本臂不带);
    #   已核对这 9 列在**端点交换下全对称** (dpT/dzp 的符号被 up 抵消) -> bias 不会引入顺序泄漏。
    # 单变量理由: 与 v637 构成"只多一个 bias"的干净对照; 与 v633 构成"line-graph + bias"的对照。
    "v649": dict(epochs=60, early_stop_patience=15, tag="v649_lg_bias", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9,
                         line_graph_attn=True, line_graph_rounds=1, line_graph_heads=4,
                         line_graph_hidden=32, line_graph_bias_cols=[0, 4, 6]),
                 chain_contrastive_weight=0.5, thr=0.95),
    # ---- v650 = v649 + 容量 (rounds 3 / hidden 64 / 邻居上限 32->64 / bias 隐层 64) ----
    # 动机: v637 的注意力容量很小 (1 轮 / hidden 32 / 邻居 32), "没用"可能只是容量不够。
    # 邻居上限取 64 而**非不限**: build_line_graph 在不裁剪时代价是 Σ_v deg(v)^2, 真实每事件
    #   ~95 径迹 -> Σ d² ~ 8e5 对/事件, 多事件 batch 下有显存风险; 64 已比 v637 翻倍且可控。
    "v650": dict(epochs=60, early_stop_patience=15, tag="v650_lg_bias_cap", dz_dict=True,
                 derived_prune=True, derived_triangle=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=9,
                         line_graph_attn=True, line_graph_rounds=3, line_graph_heads=4,
                         line_graph_hidden=64, line_graph_max_neighbors=64,
                         line_graph_bias_cols=[0, 4, 6], line_graph_bias_hidden=64),
                 chain_contrastive_weight=0.5, thr=0.95),
    # ---- v651 = v648 + line-graph 注意力 + 几何 pair-bias (2026-10-06) ----
    # 动机: 500 事件 bench (n=5964, SE≈0.006) 把底子排名**重排**了 —— v623/v624/v648 属第一
    #   梯队 (0.6524~0.6534), 而 v633 "clean stack" 反而低 0.016:
    #     v633 = v624 + (chain_lca_filter / edge_dz_ip_canon / ranking / 链对比 / event_bias) -> -0.016
    #     v643 = v633 + 顶点几何 -> +0.000;  v648 = v643 + 竞争上下文 (derived_comp) -> +0.015
    #   即"整套堆叠"轻微有害, 而唯一有效的新特征是把 deg(竞争度) 类上下文加进去。
    #   所以把"几何 bias 能不能救 line-graph 注意力"放到**第一梯队**的底子上再问一遍 (与 v649 并行)。
    # bias 列 = der_edges 的 9/10/11 = doca/100, log(doca+1e-5), |Δ起点|/100 (derived_vertex_geom),
    #   **与探针里最强的 bias 列完全一致** (探针 permutation importance: doca +0.1066 排第一)。
    # 单变量: 除 line_graph_* 外与 v648 逐键一致。
    "v651": dict(epochs=60, early_stop_patience=20, tag="v651_geo_bias", dz_dict=True,
                 derived_prune=True, derived_triangle=True,
                 derived_vertex_geom=True, derived_comp=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=18,
                         line_graph_attn=True, line_graph_rounds=1, line_graph_heads=4,
                         line_graph_hidden=32, line_graph_bias_cols=[9, 10, 11]),
                 chain_contrastive_weight=0.5, thr=0.95,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
}

DZ_DICT_0904 = ("/lzufs/user/guoqingxiang/DFEI_IFT_20260904/dfei_repo/preprocessing/"
                "normalization_dict.pt")   # 0904 生产用的归一化字典 (delta_z0 的 center/scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=list(VARIANTS),
                    help="只写指定的版本 (默认全写)")
    a = ap.parse_args()
    base = yaml.safe_load(open(BASE))
    for ver in a.versions:
        v = VARIANTS[ver]
        c = copy.deepcopy(base)
        num = int(ver[1:])
        c["settings"]["model"] = num
        c["DFEI"]["cpt"] = INIT                     # 从 v601 权重出发 (不恢复 optimizer)
        c["settings"]["epochs"] = v["epochs"]
        c["settings"]["early_stop_patience"] = v["early_stop_patience"]
        c["settings"]["log_version"] = num
        c["evaluate"]["over_write"] = v["tag"]
        for k, key in [("edge_ohem_frac", "edge_ohem_frac"),
                       ("edge_pos_weight_scale", "edge_pos_weight_scale"),
                       ("edge_prune_weight", "edge_prune_weight"),
                       ("edge_rank_weight", "edge_rank_weight"),
                       ("edge_rank_nneg", "edge_rank_nneg"),
                       ("edge_rank_margin", "edge_rank_margin"),
                       ("edge_dz_flip_prob", "edge_dz_flip_prob"),
                       ("edge_dz_ip_canon", "edge_dz_ip_canon"),
                       ("edge_dz_pvz_canon", "edge_dz_pvz_canon"),
                       ("edge_dz_abs", "edge_dz_abs"),
                       ("dir_head_weight", "dir_head_weight"),
                       ("node_rank_weight", "node_rank_weight"),
                       ("node_rank_nneg", "node_rank_nneg"),
                       ("derived_prune", "derived_prune"),
                       ("derived_triangle", "derived_triangle"),
                       ("chain_lca_filter", "chain_lca_filter"),
                       ("event_count_weight", "event_count_weight"),
                       ("chain_contrastive_weight", "chain_contrastive_weight"),
                       ("chain_contrastive_tau", "chain_contrastive_tau"),
                       # [2026-09-26] 层3 新键: listwise 排序模式 / 次级顶点一致性特征
                       ("edge_rank_mode", "edge_rank_mode"),
                       ("derived_vertex", "derived_vertex"),
                       ("edge_dz_abs", "edge_dz_abs"),
                       ("edge_dz_pvz_canon", "edge_dz_pvz_canon"),
                       # [2026-09-29] 链级 min-pooling recall loss (AND / 最弱环节语义)
                       ("chain_recall_weight", "chain_recall_weight"),
                       ("chain_recall_edge_weight", "chain_recall_edge_weight"),
                       ("chain_recall_thr", "chain_recall_thr"),
                       ("chain_recall_tau", "chain_recall_tau"),
                       # [2026-09-30] 层4: 顶点几何绝对量 / 端点对称直连 / 只训剪枝
                       ("derived_vertex_geom", "derived_vertex_geom"),
                       ("derived_pair_sym", "derived_pair_sym"),
                       ("lca_weight", "lca_weight"),
                       ("chain_lca_loss_weight", "chain_lca_loss_weight"),
                       ("pv_asso_weight", "pv_asso_weight"),
                       # [2026-10-01] 竞争/排他性上下文特征
                       ("derived_comp", "derived_comp")]:
            if k in v:
                c["inference"][key] = v[k]
        if v.get("dz_dict"):
            c["inference"]["dz_norm_dict"] = DZ_DICT_0904
        for k, val in (v.get("gn") or {}).items():      # GN blocks 级开关 (模型结构)
            c["DFEI"]["GNblocks"][k] = val
        # [2026-09-29] settings 段扩展 (监控指标 / val 剪枝指标等), 避免"新键被吞"
        for k, val in (v.get("settings_extra") or {}).items():
            c["settings"][k] = val
        if "thr" in v:
            c["inference"]["node_prune_thr"] = v["thr"]
            c["inference"]["edge_prune_thr"] = v["thr"]
        if "edge_topk" in v:
            c["inference"]["edge_topk"] = v["edge_topk"]
        p = f"config_files/train_CERN_0904_{ver}.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
        print("写出", p, f"epochs={v['epochs']}",
              {k: v[k] for k in v if k not in ("epochs", "tag")})


if __name__ == "__main__":
    main()
