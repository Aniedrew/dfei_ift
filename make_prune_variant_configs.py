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
                       ("edge_dz_pvz_canon", "edge_dz_pvz_canon")]:
            if k in v:
                c["inference"][key] = v[k]
        if v.get("dz_dict"):
            c["inference"]["dz_norm_dict"] = DZ_DICT_0904
        for k, val in (v.get("gn") or {}).items():      # GN blocks 级开关 (模型结构)
            c["DFEI"]["GNblocks"][k] = val
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
