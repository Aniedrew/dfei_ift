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
    # ---- v653 (A2 上主模型) = v648 + 最弱环链级存活损失 (chain_weakest_w=1.0) ----
    # 依据 (2026-10-09): 纯 CPU 小探针上首次出现**大而稳**的多 B 增益 —— "链存活@边精度90%"
    #   24.2% -> 32.6% (3/3 种子全为正, 平均 +8.4pp), 且边级 AP/AUC 同时变好 (0.8393->0.8533 /
    #   0.8650->0.8732)。关键对照: 换成"乘积/几何平均"形式 (≈逐边 BCE) 只有 28.0% ->
    #   **有效的是罚"链内最弱环"的非线性形式, 而不是"换成乘积"**。
    # 物理动机: 判据是"整条真值链每个环节都要对"(AND, 一失毁全链), 而训练一直是逐边 BCE。
    # 单变量: 与 v648 只差 chain_weakest_w (默认 0.0 = 逐位无影响, 已验证)。
    "v653": dict(epochs=60, early_stop_patience=20, tag="v653_geo_comp_weakest", dz_dict=True,
                 derived_prune=True, derived_triangle=True,
                 derived_vertex_geom=True, derived_comp=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=18),
                 chain_contrastive_weight=0.5, thr=0.95,
                 chain_weakest_w=1.0, chain_weakest_gamma=10.0,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    # ==== 2026-10-09 重做链级损失臂 (依据 500 事件 bench 的物理判据) ====
    # 依据: 500 事件表里**链存活与难池 AP 排序几乎无关**; v641/v642 (v633 + chain_recall)
    #   难池 AP 只在中游 (0.6389/0.6380, 8-9th), 但**链存活 59.2%/61.6%, 比 AP 冠军 v623 的 47.1%
    #   高 14pp**, 且 per_link_p_2 (2 环节链的单环节存活率) 0.80 vs v623 0.71 / v633 0.67 / v637 0.65。
    #   => 之前的结论"链级损失有害"是判据问题 (200 事件 + 只看 AP), 不是机制问题。
    # 因此把 chain_recall 搬到**当前最优底子 v648** 上重做 (v641/v642 只差早停参数, 且 v641 的
    #   validate_prune_events=200 是已知 bug, 故只取一份公平设置)。与已在跑的 v653
    #   (v648 + 最弱环 softmin 链级损失) 构成"同一想法的两种写法"对照。
    # ---- v654 = v648 + chain_recall (v642 的原参数: w=5/5, thr=0.9, tau=0.1) ----
    "v654": dict(epochs=60, early_stop_patience=20, tag="v654_geo_comp_chainrecall", dz_dict=True,
                 derived_prune=True, derived_triangle=True,
                 derived_vertex_geom=True, derived_comp=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=18),
                 chain_contrastive_weight=0.5, thr=0.95,
                 chain_recall_weight=5.0, chain_recall_edge_weight=5.0,
                 chain_recall_thr=0.9, chain_recall_tau=0.1,
                 settings_extra=dict(monitor_metric="val_prune_ap",
                                     validate_prune_metric=True,
                                     validate_prune_events=1000,
                                     early_stop_min_delta=0.005)),
    # ---- v655 = v648 + chain_recall (权重加强 5->15): 物理判据是"链存活", 想确认权重是否是限制项 ----
    "v655": dict(epochs=60, early_stop_patience=20, tag="v655_geo_comp_chainrecall_w15", dz_dict=True,
                 derived_prune=True, derived_triangle=True,
                 derived_vertex_geom=True, derived_comp=True, chain_lca_filter=True,
                 edge_dz_ip_canon=True, edge_rank_weight=10.0, edge_rank_nneg=64,
                 gn=dict(event_bias=True, extra_node_dim=7, extra_edge_dim=18),
                 chain_contrastive_weight=0.5, thr=0.95,
                 chain_recall_weight=15.0, chain_recall_edge_weight=15.0,
                 chain_recall_thr=0.9, chain_recall_tau=0.1,
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

# ============================================================================
# 2026-10-13 层6: 以 **v654 为固定底子** 的一次性大批量铺开 (v656..v675, 共 20 个臂)。
#
# 底子为什么是 v654: v654 = v648 + 链级 chain_recall 最弱环损失 (w=5/5, thr=0.9, tau=0.1),
#   3000 事件基准上把主判据"整链完整恢复比例 chain_surv@0.9"从 35.8% 抬到 52.3% ——
#   是本项目当前最好底子。本批**一律以 v654 为底, 只改动下游/损失侧**。
#
# 实现方式: 全部由 _V654 = VARIANTS["v654"] 派生, 用 _v654_plus(**over) 只覆盖列出的键 ->
#   保证每个臂除自己那 1~2 个键外与 v654 **逐键一致** (便于干净的单/双变量对照,
#   也避免重复粘贴 v654 那一大段 dict 时手抖改错键)。生成后逐一 diff 复核。
#
# 覆盖方向 (对应任务里的 6 组):
#   组1 链级损失族 (已验证方向的精调)          : v656..v661
#   组2 辅助头权重 (改变"重建什么"的侧重)      : v662..v666
#   组3 边打分器训练方式                       : v667..v669
#   组4 下游结构/注意力                        : v670..v671
#   组5 特征家族补齐 (顶点+端点对称直连)       : v672..v673
#   组6 监控/选择                              : v674..v675
#
# 因总量上限 20 而**未纳入**的旋钮及原因 (见返回值):
#   chain_recall_thr=0.8 / chain_recall_tau=0.5 (各留一个代表值探方向即可, 权重扫描优先);
#   event_count_weight (v630 实测 -19%, 且需新开头, 信息量低);
#   edge_pos_weight_scale=2.0 (v612/v613 实测 -12~-16%, 已知有害)。
# ============================================================================
_V654 = VARIANTS["v654"]


def _v654_plus(**over):
    """以 v654 全部键为基, 仅覆盖 over 中列出的键 -> 保证单/双变量干净。"""
    d = dict(_V654)
    d.update(over)
    return d


# ---- 组1: 链级损失族 (v654 已验证: chain_recall w=5/5, thr=0.9, tau=0.1) ----
# 预期: 找 chain_recall 的最优权重 / 阈值 / 锐度, 并试更强的最弱环(softmin)组合。
VARIANTS["v656"] = _v654_plus(chain_recall_weight=2.0, chain_recall_edge_weight=2.0,
                              tag="v656_cr_w2")        # 权重减半: 确认 5 是否过强 / 边AP是否被压
VARIANTS["v657"] = _v654_plus(chain_recall_weight=10.0, chain_recall_edge_weight=10.0,
                              tag="v657_cr_w10")       # 权重翻倍: chain_surv 与难池AP 的折中点
VARIANTS["v658"] = _v654_plus(chain_recall_weight=20.0, chain_recall_edge_weight=20.0,
                              tag="v658_cr_w20")       # 权重×4: 上限 / 是否开始伤边级AP
VARIANTS["v659"] = _v654_plus(chain_recall_thr=0.7, tag="v659_cr_thr07")
                              # 阈值降到 0.7 (推理剪枝在 0.95): 目标更松 -> 是否反而更稳
VARIANTS["v660"] = _v654_plus(chain_recall_tau=0.3, tag="v660_cr_tau03")
                              # 温度 0.1->0.3 (更软): 更平滑的链级梯度是否更易优化
VARIANTS["v661"] = _v654_plus(chain_weakest_w=3.0, chain_weakest_gamma=30.0,
                              tag="v661_weakest_w3g30")
                              # 与 chain_recall 互补的 softmin 最弱环损失更强版本 (v653 只 w=1,γ=10)

# ---- 组2: 辅助头权重 (单变量: 只改一个下游损失权重) ----
# 预期: 看"把梯度更多/更少投给其它头"是否通过共享主干间接改善剪枝链存活。
VARIANTS["v662"] = _v654_plus(lca_weight=20.0, tag="v662_lca_x2")     # LCA 权重 10->20
VARIANTS["v663"] = _v654_plus(lca_weight=5.0, tag="v663_lca_x05")     # LCA 权重 10->5
VARIANTS["v664"] = _v654_plus(pv_asso_weight=2.0, tag="v664_pvasso_x2")   # PV关联头权重 1->2
VARIANTS["v665"] = _v654_plus(pv_asso_weight=0.5, tag="v665_pvasso_x05")  # PV关联头权重 1->0.5
VARIANTS["v666"] = _v654_plus(dir_head_weight=10.0,
                              gn=dict(_V654["gn"], MLP_infer_dir_head=True),
                              tag="v666_dirhead_w10")
                              # 打开剪枝MLP的"方向头"(v654 里该头是关的 -> 权重无意义), 用 v618 的 w=10

# ---- 组3: 边打分器的训练方式 (单变量) ----
VARIANTS["v667"] = _v654_plus(edge_rank_weight=20.0, tag="v667_rank_x2")   # pairwise ranking 权重 10->20
VARIANTS["v668"] = _v654_plus(edge_rank_mode="infonce", tag="v668_rank_infonce")
                              # ranking 从 hinge 换成 listwise(InfoNCE): 直接约束顶部难负例
VARIANTS["v669"] = _v654_plus(edge_ohem_frac=0.25, tag="v669_ohem025")
                              # OHEM 保留最难 25% 负边 (已知在旧底子上 -12~-16%, 在链级损失底子上复检)

# ---- 组4: 下游结构/注意力 (单变量, 挂最后一个 GN block) ----
VARIANTS["v670"] = _v654_plus(
    gn=dict(_V654["gn"], context_prune=dict(mode="attn", heads=4, hidden=32, refine_edge=True)),
    tag="v670_ctxprune")
                              # 上下文感知剪枝头: 让点/边裁决看到邻域分数 (CRF/SAGPool 式)
VARIANTS["v671"] = _v654_plus(
    gn=dict(_V654["gn"], line_graph_attn=True, line_graph_rounds=2,
            line_graph_heads=4, line_graph_hidden=64),
    tag="v671_linegraph")
                              # tt边图(LCAG)边-边注意力 2 轮: 结构性攻"跨链假边" (传递性)

# ---- 组5: 特征家族补齐 (必须同步 extra_edge_dim, 靠冒烟卡维度自洽) ----
# v654 边派生维 = 7(基础)+2(三角)+3(顶点几何)+6(竞争) = 18。
VARIANTS["v672"] = _v654_plus(derived_vertex=True, derived_pair_sym=True,
                              gn=dict(_V654["gn"], extra_edge_dim=45),
                              tag="v672_vertex_psym")
                              # +3(次级顶点一致性 zcpa/flight/collin) +24(端点对称直连) = 45 维
VARIANTS["v673"] = _v654_plus(derived_pair_sym=True,
                              gn=dict(_V654["gn"], extra_edge_dim=42),
                              tag="v673_psym")
                              # 只加端点对称直连 [x_i+x_j,|x_i-x_j|,x_i*x_j] (24 维) = 42 维

# ---- 组6: 监控 / 选择 (改变选 ckpt 的稳定性与训练长度) ----
VARIANTS["v674"] = _v654_plus(
    settings_extra=dict(_V654["settings_extra"], validate_prune_events=3000),
    tag="v674_val3000")
                              # val 剪枝事件 1000->3000: ckpt 选择的 val_prune_ap 更稳
VARIANTS["v675"] = _v654_plus(epochs=120, tag="v675_ep120")
                              # 训练 60->120 轮 (patience 仍 20): 是否欠训练

# ============================================================================
# 2026-10-13 方向1: 全局划分损失 (partition loss) —— 传递性 / 排他性 / 集合F1
#
# 底子 = v654 (同上, 全部键逐键继承)。已定结论: 把"整条真值链一失毁全"的 AND 语义写进
#   损失 (chain_recall 的链内最弱环) 是目前唯一大幅有效的改动 (3000 事件: 整链完整恢复
#   35.8% -> 52.3%)。但它**只罚链内最弱环节**, 缺另一半: 现有架构是"逐边二分类 + 取连通
#   分量", 没有任何东西保证**划分一致性** (一条径迹可同时属于两条链 / 跨链假边可两个都判
#   高分)。本方向把"划分一致性"写进损失, 三项 (都实现在 dfei_lightning_module.py):
#     part_trans_w (1a) 传递性: 真值同链共用一个节点的两条边 -> 第三边(闭合环)也必须高分;
#        罚 relu(min(p_e1,p_e2)-p_e3)。
#     part_excl_w  (1b) 排他性: 一条径迹的存活质量不应分散到多条不同真值链;
#        罚 Σ_v(1-Σ_g m_g²), 非真值伙伴归 -1 组。
#     part_f1_w    (1c) 集合 F1: 事件级 prec/rec 的调和平均 (尺度无关的集合重叠, 取代逐边平均);
#        罚 1-f1。
#   缺省三 w 全 0 -> 与 v654 逐位一致 (已在单测 + 真实 CPU 冒烟验证)。
#
# 命名: 本方向统一用 part 前缀 (不用 v6xx, 避免与既有版本号视觉混淆); log_version 从 701 起。
# 实现: 由 _part(**over) 从 _v654_plus 派生 -> 除新开的 1~2 个键外与 v654 逐键一致;
#       log_version/tag 也在此注入。
# ============================================================================
_PART_BASE = 700


def _part(slug, log_version, **over):
    """以 v654 全部键为基, 追加本方向的新开关 (part_*) 与 tag/log_version。

    slug/log_version 生成描述性 tag (如 part_trans_701) 与 settings.log_version;
    除 over 中列出的键外与 v654 逐键一致 -> 干净的单/双变量对照。
    """
    d = _v654_plus(**over)
    d["tag"] = f"part_{slug}_{log_version}"
    d["log_version"] = log_version
    return d


# ---- 单项臂: 三个损失各自单独打开 (w=1 起, 与 chain_recall 5 同量级但属独立信号) ----
# 预期测: "划分一致性"的哪一半 (传递 / 排他 / 集合重叠) 单独就有效。
VARIANTS["part_trans"] = _part("trans", 701, part_trans_w=1.0)
VARIANTS["part_excl"] = _part("excl", 702, part_excl_w=1.0)
VARIANTS["part_f1"] = _part("f1", 703, part_f1_w=1.0)
# ---- 组合臂: 三项同开 (w=1 起), 检验互补性 ----
# 预期测: 三个信号叠加是否 > 任一单项 (它们罚的是不同的划分失配)。
VARIANTS["part_all"] = _part("all", 704, part_trans_w=1.0, part_excl_w=1.0, part_f1_w=1.0)
# ---- 权重档: 三项同开 w=3 ----
# 预期测: w=1 是否过弱 (对照 chain_recall 的 5 量级), 或叠加后是否开始伤边级 AP。
VARIANTS["part_all_w3"] = _part("allw3", 705, part_trans_w=3.0, part_excl_w=3.0, part_f1_w=3.0)
# ---- 排他性温度档: 只开排他性, tau=0.3 (更锐: 归一化前概率更接近 0/1) ----
# 预期测: 排他性的"质量集中度"对温度是否敏感 (tau<1 更接近硬分配)。
VARIANTS["part_excl_tau03"] = _part("excltau03", 706, part_excl_w=1.0, part_excl_tau=0.3)

# ============================================================================
# 2026-10-13 方向4: 几何候选生成 (geometry-based candidate generation)
#
# 问题: 现在 recall 由**学出来的**边二分类头决定 (漏一条真值边整条链报废), 给不出召回保证;
#   且下游要在 O(N^2) 里学占比极低的真值边。本方向把"生成候选"从学习里剥离: 用**不学习**的
#   纯几何量 (dR / DOCA / |Δ起点|, 由 derive_pruning_features 现算, 绕开上游 log_DOCA_reco 的
#   bug) 定义候选窗口 -> 召回由物理保证, 学习只在干净子集上打分。
# 实现 (dfei_lightning_module.py):
#   geomcand_train_mask: 训练时把边剪枝/LCAG 的**损失统计**限制在候选集内, 并打印"被窗口漏掉的
#       真值边比例";
#   geomcand_gate: 推理(test)时把候选作为**乘性门**接到学习分数上 (非候选边分数置 0 -> 不可能选中)。
#   两开关独立, 缺省 False -> 与 v654 逐位一致 (已在单测 + 真实 CPU 冒烟验证)。
#
# 窗口阈值**来自第一步的离线标定** (analyze_geomcand_windows.py, 200 事件真实 MC inclusive_00342451,
#   946508 边 / 1112 真值结构边 / 147 真值链; 表见 report_figs/geomcand_study.json):
#     loose (dR<=0.30, DOCA<=5.0mm, |Δ起点|<=200mm): 真值边 recall=0.801 / 候选压缩 1.55x / 链恢复上界 0.667
#     mid   (dR<=0.25, DOCA<=0.5mm, |Δ起点|<=200mm): recall=0.557 / 压缩 3.63x / 上界 0.381
#     tight (dR<=0.20, DOCA<=0.5mm, |Δ起点|<=50mm):  recall=0.315 / 压缩 11.86x / 上界 0.177
#   关键发现: 几何窗口**无法同时**给出高召回与强压缩 (recall>0.9 时压缩<1.5x) -> 这正是"学习在
#   窗口内打分"要补的那一段; loose 档的链恢复上界 0.667 已高于 v654 的 chain_surv 基准 0.523。
#
# 命名: 本方向统一用 geomcand 前缀; log_version 从 1001 起。生成用 _geomcand(**over) 从
#   _v654_plus 派生 -> 除新开的 1~2 个键外与 v654 逐键一致 (干净的单/双变量对照)。
# ============================================================================
_GEO_LOOSE = dict(geomcand_max_r=0.30, geomcand_max_doca=5.0, geomcand_max_dz=200.0)
_GEO_MID = dict(geomcand_max_r=0.25, geomcand_max_doca=0.5, geomcand_max_dz=200.0)
_GEO_TIGHT = dict(geomcand_max_r=0.20, geomcand_max_doca=0.5, geomcand_max_dz=50.0)


def _geomcand(slug, log_version, **over):
    """以 v654 全部键为基, 追加本方向的新开关 (geomcand_*) 与 tag/log_version。

    除 over 中列出的键外与 v654 逐键一致 -> 干净的单/双变量对照; 生成后逐一 diff 复核。
    """
    d = _v654_plus(**over)
    d["tag"] = f"geomcand_{slug}_{log_version}"
    d["log_version"] = log_version
    return d


# ---- 三档窗口 (只开 geomcand_train_mask): 用第一步标定的 loose/mid/tight 阈值 ----
VARIANTS["geomcand_loose"] = _geomcand("loose", 1001, geomcand_train_mask=True, **_GEO_LOOSE)
VARIANTS["geomcand_mid"] = _geomcand("mid", 1002, geomcand_train_mask=True, **_GEO_MID)
VARIANTS["geomcand_tight"] = _geomcand("tight", 1003, geomcand_train_mask=True, **_GEO_TIGHT)
# ---- 只开 gate (推理门控), 用中间档窗口 ----
VARIANTS["geomcand_gate"] = _geomcand("gate", 1004, geomcand_gate=True, **_GEO_MID)
# ---- mask + gate 同开 (训练在候选集上学 + 推理把候选作为乘性门) ----
VARIANTS["geomcand_both"] = _geomcand("both", 1005, geomcand_train_mask=True, geomcand_gate=True,
                                      **_GEO_MID)

# ============================================================================
# 2026-10-13 方向3: 可学习顺序重组合解码器 (anti-kT 式层次聚类), 取代"取连通分量"
#
# 动机: 重建最后一步 "取连通分量" (`reco_helper.reconstruct_decay` 的 nx.connected_components)
#   不保排他性、不显式决定链数。换成 sequential recombination: 反复合并"最近"的一对簇,
#   天然独占 (每条径迹只用一次)、不需预设簇数。距离度量可学习:
#     d = -ln(p_e + eps) + lam * d_geom   (p_e=模型边存活概率; d_geom=DOCA/100, 现算)
#   离线评测 (eval_seqrec_offline.py, v654 ckpt, 600 事件真实数据, 同一候选边+同一真值):
#     连通分量基线   整链完整恢复 7.79% / 链级 P 33.5% / R 37.7%
#     seqrec-learned 整链恢复 17.3% / P 50.3% / R 42.0%   (cut=0.02; 约 2.2x 整链恢复)
#     seqrec-geom(无学习) 整链恢复 ~0% -> 证明**学出来的度量**是关键。
#
# 命名: 本方向统一用 seqrec 前缀; log_version 从 901 起。生成用 _seqrec(**over) 从
#   _v654_plus 派生 -> 除新开的键外与 v654 逐键一致 (干净对照)。
# 训练侧: dfei_lightning_module._seqrec_loss 用**可微代理损失**训练 lam (ranking hinge)
#   与 cut (双边 hinge); 推理侧按 seqrec_use 在 EventReconstruction 里取代连通分量。
#   缺省全关 -> 与 v654 逐位一致 (单测 + CPU 冒烟已核对)。
# ============================================================================
_SEQREC_BASE = 900


def _seqrec(slug, log_version, **over):
    """以 v654 全部键为基, 追加本方向的新开关 (seqrec_*) 与 tag/log_version。

    除 over 中列出的键外与 v654 逐键一致 -> 干净的单/双变量对照; 生成后逐一 diff 复核。
    """
    d = _v654_plus(**over)
    d["tag"] = f"seqrec_{slug}_{log_version}"
    d["log_version"] = log_version
    return d


# ---- v901: lam 可学习 (度量 mix), cut 固定 0.02 (离线评测的甜点) ----
VARIANTS["seqrec_learned"] = _seqrec("learned", 901,
                                     seqrec_use=True, seqrec_metric="mix",
                                     seqrec_lam=1.0, seqrec_lam_learn=True,
                                     seqrec_cut=0.02, seqrec_cut_learn=False,
                                     seqrec_loss_weight=1.0)
# ---- v902: cut 作为可学习标量 (度量 learned), lam 固定 1.0 ----
VARIANTS["seqrec_cutlearn"] = _seqrec("cutlearn", 902,
                                      seqrec_use=True, seqrec_metric="learned",
                                      seqrec_lam=1.0, seqrec_lam_learn=False,
                                      seqrec_cut=0.02, seqrec_cut_learn=True,
                                      seqrec_loss_weight=1.0)
# ---- v903: lam 与 cut 同时可学习 (度量 mix) ----
VARIANTS["seqrec_both"] = _seqrec("both", 903,
                                  seqrec_use=True, seqrec_metric="mix",
                                  seqrec_lam=1.0, seqrec_lam_learn=True,
                                  seqrec_cut=0.02, seqrec_cut_learn=True,
                                  seqrec_loss_weight=1.0)
# ---- v904: 仅训练时用新度量 (推理仍走连通分量) —— 隔离"度量损失"与"解码器替换" ----
VARIANTS["seqrec_trainonly"] = _seqrec("trainonly", 904,
                                       seqrec_use=False, seqrec_metric="mix",
                                       seqrec_lam=1.0, seqrec_lam_learn=True,
                                       seqrec_cut=0.02, seqrec_cut_learn=True,
                                       seqrec_loss_weight=1.0)

# ============================================================================
# 2026-10-13 方向2: 集合预测头 (set prediction: slot + mask + 匈牙利匹配)
#
# 问题: 现在下游是"逐对四分类 + 取连通分量/组装", **从不显式回答**"这个事件里有几条链、
#   每条链含哪些径迹"。多 B 事件的失效模式正来自这里 —— 边分类只看"这一对像不像同一条链",
#   没有对"到底几条链、哪条链归哪些径迹"做全局、排他的决策。
# 本方向换范式 (DETR / MaskFormer 式): 头部一次输出 K 个"链槽位", 每个槽位给出
#   (i) 存在性 (该槽位是不是一条链) 与 (ii) 对每条径迹的归属 mask; 训练用匈牙利匹配把
#   预测槽位与真值链一一对齐 -> 天然表达"链数"与"每条径迹只属于一条链"。
#
# 实现 (wmpgnn/model/set_pred_head.py + dfei_lightning_module / reconstruction):
#   - pair-wise: 槽位 query 与 tt 边表征 cross-attention, 再把边信息按端点聚合回径迹;
#     mask 头输入显式含两端表征 [h_t, c_k, edgeAgg, h_t*c_k] (不是只对节点表征池化);
#   - 归属归一化 setpred_assign: "softmax_bg"(默认, 逐径迹 softmax 到 K 槽位+背景 -> 天然排他)
#     | "sigmoid"(逐 (径迹,槽位) 独立, 排他靠推理 argmax);
#   - 匹配代价 = -exist_prob + (1 - Dice), scipy linear_sum_assignment (逐事件, CPU);
#   - 开启键: setpred_use(总开关) / setpred_nslots(K) / setpred_loss_w(损失权重) /
#     setpred_dustbin_w(未匹配槽位"无对象"项权重) / setpred_decode(推理用槽位解码取代连通分量);
#     全关时与 v654 逐位一致 (单测 + CPU 冒烟已核对)。
#
# K 的选取: 统计真实训练数据每事件真值链数 -> 0/1/2/3 链 = 43.4/38.6/17.9/0.13%,
#   均值 0.748, P99=2, P100=3 -> K=4 覆盖全部并留头程 (setpred_k4); K=8 作更宽对照。
#
# 命名: 统一 setpred 前缀; log_version 从 801 起; 生成用 _setpred(**over) 从 _v654_plus 派生
#   -> 除新开的键外与 v654 逐键一致 (干净的单/双变量对照)。
# ============================================================================
_SETPRED_BASE = 800


def _setpred(slug, log_version, **over):
    """以 v654 全部键为基, 追加本方向的新开关 (setpred_*) 与 tag/log_version。

    除 over 中列出的键外与 v654 逐键一致 -> 干净对照; 生成后逐一 diff 复核。
    """
    d = _v654_plus(**over)
    d["tag"] = f"setpred_{slug}_{log_version}"
    d["log_version"] = log_version
    return d


_COMMON_SP = dict(setpred_use=True, setpred_loss_w=1.0,
                  setpred_exist_thr=0.5, setpred_assign_thr=0.5)
# ---- v801: K=4 槽位 (覆盖 P99, 实测 P100=3) —— 主力臂 ----
VARIANTS["setpred_k4"] = _setpred("k4", 801, setpred_nslots=4,
                                  setpred_assign="softmax_bg", **_COMMON_SP)
# ---- v802: K=8 槽位 (更宽; 看链数上界是否受限) ----
VARIANTS["setpred_k8"] = _setpred("k8", 802, setpred_nslots=8,
                                  setpred_assign="softmax_bg", **_COMMON_SP)
# ---- v803: K=4 + setpred_decode (推理直接用槽位解码**取代**连通分量) ----
VARIANTS["setpred_decode"] = _setpred("decode", 803, setpred_nslots=4,
                                      setpred_assign="softmax_bg",
                                      setpred_decode=True, **_COMMON_SP)
# ---- v804: K=4 + 损失权重 ×3 (w=1 是否欠训) ----
VARIANTS["setpred_w3"] = _setpred("w3", 804, setpred_nslots=4,
                                  setpred_assign="softmax_bg",
                                  setpred_use=True, setpred_loss_w=3.0,
                                  setpred_exist_thr=0.5, setpred_assign_thr=0.5)
# ---- v805: 对照臂 —— 故意把 mask 建在**节点池化**上 (丢弃成对/边信息), 量化"成对表示"的必要性 ----
VARIANTS["setpred_node_mask"] = _setpred("node_mask", 805, setpred_nslots=4,
                                         setpred_assign="softmax_bg",
                                         setpred_pairwise=False, **_COMMON_SP)


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
        # 版本号: 既有臂由名字 (v654 -> 654) 推出; 方向1 的 part_* 臂名非数字 -> 用显式 log_version
        num = v.get("log_version") or int(ver[1:])
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
                       # [2026-10-09] A2: 最弱环链级存活损失 (softmin over 链内各环)
                       ("chain_weakest_w", "chain_weakest_w"),
                       ("chain_weakest_gamma", "chain_weakest_gamma"),
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
                       ("derived_comp", "derived_comp"),
                       # [2026-10-13] 方向1: 全局划分损失 (传递性/排他性/集合F1)
                       ("part_trans_w", "part_trans_w"),
                       ("part_excl_w", "part_excl_w"),
                       ("part_excl_tau", "part_excl_tau"),
                       ("part_f1_w", "part_f1_w"),
                       # [2026-10-13] 方向4: 几何候选生成 (训练掩码 + 推理门控 + 窗口阈值)
                       ("geomcand_train_mask", "geomcand_train_mask"),
                       ("geomcand_gate", "geomcand_gate"),
                       ("geomcand_max_r", "geomcand_max_r"),
                       ("geomcand_max_doca", "geomcand_max_doca"),
                       ("geomcand_max_dz", "geomcand_max_dz"),
                       # [2026-10-13] 方向3: 可学习顺序重组合解码器 (anti-kT 式层次聚类)
                       ("seqrec_use", "seqrec_use"),
                       ("seqrec_metric", "seqrec_metric"),
                       ("seqrec_lam", "seqrec_lam"),
                       ("seqrec_lam_learn", "seqrec_lam_learn"),
                       ("seqrec_cut", "seqrec_cut"),
                       ("seqrec_cut_learn", "seqrec_cut_learn"),
                       ("seqrec_linkage", "seqrec_linkage"),
                       ("seqrec_loss_weight", "seqrec_loss_weight"),
                       # [2026-10-13] 方向2: 集合预测头 (DETR 式 slot + mask + 匈牙利匹配)
                       ("setpred_use", "setpred_use"),
                       ("setpred_decode", "setpred_decode"),
                       ("setpred_nslots", "setpred_nslots"),
                       ("setpred_loss_w", "setpred_loss_w"),
                       ("setpred_dustbin_w", "setpred_dustbin_w"),
                       ("setpred_hidden", "setpred_hidden"),
                       ("setpred_heads", "setpred_heads"),
                       ("setpred_pairwise", "setpred_pairwise"),
                       ("setpred_assign", "setpred_assign"),
                       ("setpred_exist_thr", "setpred_exist_thr"),
                       ("setpred_assign_thr", "setpred_assign_thr"),
                       ("setpred_exist_weight", "setpred_exist_weight"),
                       ("setpred_assign_weight", "setpred_assign_weight")]:
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
