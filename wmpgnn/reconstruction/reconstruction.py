from tqdm import tqdm

import pandas as pd
import torch

from multiprocessing.pool import ThreadPool

from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.reconstruction.signal_dict import get_ref_signal
from wmpgnn.reconstruction.reco_helper import *
from wmpgnn.reconstruction.quantity_adder import *


class EventReconstruction:
    def __init__(self, configs):
        # boolean whether to use true reconstruction or predicted reconstruction
        self.configs = configs["inference"]
        self.use_lca = True
        if "LCA" in self.configs:
            self.use_lca = self.configs["LCA"]

        self.signal = get_ref_signal(configs["evaluate"]["sample"][0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 候选衰变链选择 MLP (可选): 重建出多条链后, 逐链打分, 过滤低分链
        # 来源二选一:
        #   "builtin"     -> scorer 挂在 model 上 (联合训练), 由 lightning module 注入 self.chain_scorer
        #   "ckpt路径"    -> 独立训练的 scorer checkpoint, 首次使用时按数据维度加载
        self.chain_scorer = None
        self.chain_score_thr = self.configs.get("selection_mlp_margin", None)
        sel_ckpt = self.configs.get("selection_mlp", "")
        self._sel_ckpt = sel_ckpt
        self._scorer_init = sel_ckpt == "builtin" or not sel_ckpt
        if sel_ckpt and sel_ckpt != "builtin":
            from wmpgnn.reconstruction.topk_selection import load_scorer
            self._load_scorer = load_scorer

        self.evt_counter = 0
        self.sig_df, self.evt_df = [], []

        # we look at independent tracks so we can directly sum it which shouldn't be a problem
        self.log = {"pv_corr_ml": {}, "pv_corr_ip": {}, "pv_total": {}, "npvs": {}}

    def collect_results(self):
        sig_dfs = []
        evt_dfs = []

        for event_number, (sig_df, evt_df) in enumerate(zip(self.sig_df, self.evt_df), start=1):
            sig_df = sig_df.copy()
            sig_df.insert(0, 'EventNumber', event_number)
            sig_dfs.append(sig_df)

            evt_df = evt_df.copy()
            evt_df.insert(0, 'EventNumber', event_number)
            evt_dfs.append(evt_df)

        self.sig_df = pd.concat(sig_dfs, ignore_index=True)
        self.evt_df = pd.concat(evt_dfs, ignore_index=True)
        return self.sig_df, self.evt_df

    def reconstruct_heavyhadrons(self, outputs, ft_des=None, pv_des=None):
        # debatching
        graphs = outputs.to_data_list()

        # Output weights
        node_weights = outputs["node_weights"]
        edge_weights = outputs["edge_weights"]
        lca = outputs[("tracks", "tracks")].lca

        track_batch = outputs["tracks"].batch
        pv_batch = outputs['pvs'].batch
        tr_tr_edge_idx = outputs[('tracks', 'tracks')].edge_index
        tr_pv_edge_idx = outputs[('tracks', 'pvs')].edge_index

        n_graphs = track_batch.max().item() + 1
        graph_ids = torch.arange(n_graphs, device=self.device)
        track_masks = track_batch.unsqueeze(1) == graph_ids.unsqueeze(0)  # Shape: [n_tracks, n_graphs]

        precomputed_pv_desc = []
        precomputed_ft_desc = []
        for i in range(n_graphs):
            # Create boolean mask to find per event information
            track_mask = track_masks[:, i]
            tr_tr_mask = track_masks[tr_tr_edge_idx[0], i] & track_masks[tr_tr_edge_idx[1], i]
            pv_mask = pv_batch == i
            tr_pv_mask = track_masks[tr_pv_edge_idx[0], i] & pv_mask[tr_pv_edge_idx[1]]

            graphs[i][("tracks", "tracks")].lca = lca[tr_tr_mask]

            evt_tr_pv_edge_idx = tr_pv_edge_idx[:, tr_pv_mask]
            ntracks = torch.unique(evt_tr_pv_edge_idx[0]).shape[0]
            npvs = torch.unique(evt_tr_pv_edge_idx[1]).shape[0]

            # some temp holding final part ids (公开数据 tracks 无 part_ids, 容错为 None)
            temp = graphs[i]["part_ids"] if "part_ids" in graphs[i] else getattr(graphs[i]["tracks"], "part_ids", None)

            # log pv performance for all tracks
            cluster_pred_pv = None  # 方案7b: pv_cluster_head 稠密得分矩阵 (train/infer 同一分簇器)
            if pv_des is not None:
                # pv_des 中所有量都是 tracks-pvs 边级别的。原实现假设每个事件内 track-pv 图是稠密的
                # (每条 track 连接所有 pv), 用 view(ntracks, npvs) 稠密化; 但公开 LHCb 数据的
                # track-pv 图是稀疏的, 该 view 会因元素数不匹配崩溃。这里改为按 track 分组聚合,
                # 稠密情形是其特例, 两种数据格式都兼容。
                evt_tr_pv = tr_pv_edge_idx[:, tr_pv_mask]       # [2, n_edges] 事件内 track-pv 边
                edge_filter = pv_des["pv_filter"][tr_pv_mask]   # [n_edges] 边是否有效
                edge_true = pv_des["true"][tr_pv_mask]          # [n_edges] 边标签 0/1
                edge_minip = pv_des["minIP"][tr_pv_mask]        # [n_edges]
                edge_pred = pv_des["pred"][tr_pv_mask]          # [n_edges]

                # 边上的 track/pv 全局索引 -> 事件内相对索引 (覆盖没有 tr-pv 边的 track)
                evt_track_idx = track_mask.nonzero()[:, 0]
                evt_pv_idx = pv_mask.nonzero()[:, 0]
                track_local = torch.searchsorted(evt_track_idx, evt_tr_pv[0])
                pv_local = torch.searchsorted(evt_pv_idx, evt_tr_pv[1])
                n_all_tracks = evt_track_idx.shape[0]
                n_all_pvs = evt_pv_idx.shape[0]

                # 每个 track 的 filter: 其所有边都须通过 filter
                pv_filter = torch.full((n_all_tracks,), True, dtype=torch.bool, device=edge_filter.device)
                bad_tracks = torch.unique(track_local[~edge_filter])
                if bad_tracks.numel() > 0:
                    pv_filter[bad_tracks] = False

                # 每 track 一个 pv 决策: true 取 filter 通过边中标签为 1 的第一条;
                # pred 取 filter 通过边中置信度最大者; minIP 取其中 IP 最小者
                y_pv = torch.full((n_all_tracks,), -1, dtype=torch.long, device=edge_true.device)
                pred_pv_track_level = torch.full((n_all_tracks,), -1, dtype=torch.long, device=edge_pred.device)
                min_ip_pv = torch.full((n_all_tracks,), -1, dtype=torch.long, device=edge_minip.device)
                for t in range(n_all_tracks):
                    sel = (track_local == t) & edge_filter
                    if not sel.any():
                        continue
                    true_sel = edge_true[sel] == 1
                    if true_sel.any():
                        y_pv[t] = pv_local[sel][true_sel][0]
                    pred_pv_track_level[t] = pv_local[sel][torch.argmax(edge_pred[sel])]
                    min_ip_pv[t] = pv_local[sel][torch.argmin(edge_minip[sel])]
                y_pv[~pv_filter] = -1
                # 重建 [n_all_tracks, n_all_pvs] 预测得分矩阵 (未连接的 track-pv 对填 0),
                # 供下游 get_pv_asso 按行 argmax / 按列求和使用
                pred_pv = torch.zeros(n_all_tracks, n_all_pvs, device=edge_pred.device)
                pred_pv[track_local, pv_local] = edge_pred
                # ==== 方案7b: pv_cluster_head 得分矩阵 (与 pred_pv 同构, 分簇用同一训练头) ====
                if "cluster_pred" in pv_des:
                    edge_cpred = pv_des["cluster_pred"][tr_pv_mask]       # [n_edges] 事件内
                    cluster_pred_pv = torch.zeros(n_all_tracks, n_all_pvs, device=edge_cpred.device)
                    cluster_pred_pv[track_local, pv_local] = edge_cpred
                # ==== DA 几何基线: raw tr-pv 边特征稠密矩阵 (edge_minip = -log-IP 类, 越小越近) ====
                geom_pv = torch.zeros(n_all_tracks, n_all_pvs, device=edge_minip.device)
                geom_pv[track_local, pv_local] = edge_minip
                # Here we remove ghost/tracks with true pv not being recoed
                if npvs not in self.log["pv_total"].keys():
                    self.log["pv_corr_ml"][npvs], self.log["pv_corr_ip"][npvs], self.log["pv_total"][npvs] = 0, 0, 0
                    self.log["npvs"][npvs] = 0
                self.log["pv_corr_ml"][npvs] += torch.sum(y_pv[pv_filter] == pred_pv_track_level[pv_filter]).item()
                self.log["pv_corr_ip"][npvs] += torch.sum(y_pv[pv_filter] == min_ip_pv[pv_filter]).item()
                self.log["pv_total"][npvs] += int(torch.sum(pv_filter).item())
                self.log["npvs"][npvs] += 1

            # apply the pruning
            # ==== 方案F: seed-expand 连通性保留剪枝 (可选, 优先于 node_prune/edge_topk) ====
            # 在全图(未删节点)上做扩展: 种子节点 → 关联边 top-k → 加回被剪节点 → k 递减迭代。
            # 必须放在 true_node_pruning (原地删节点) 之前; 节点不删, 只生成边集进重建。
            node_expand = self.configs.get("node_expand", False)
            if node_expand:
                method = self.configs.get("node_expand_method", "topk")  # "topk"(方案F) | "ppr"|"heat"(方案F')
                if method == "topk":
                    from wmpgnn.reconstruction.topk_selection import node_expand_selbool
                    node_selbool, edge_selbool = node_expand_selbool(
                        node_weights[track_mask], edge_weights[tr_tr_mask],
                        graphs[i][('tracks', 'to', 'tracks')].edge_index,
                        k0=int(self.configs.get("node_expand_k0", 12)),
                        seed_thr=float(self.configs.get("node_expand_seed_thr", 0.5)),
                        max_hop=int(self.configs.get("node_expand_max_hop", 3)),
                        decay=int(self.configs.get("node_expand_decay", 1)),
                        max_nodes=int(self.configs.get("node_expand_max_nodes", 200)),
                        edge_thr=float(self.configs.get("node_expand_edge_thr", 0.0)),
                    )
                else:
                    # 方案F': 扩散分数 + sweep cut (全局连通强度, 替代贪心 top-k 扩展)
                    from wmpgnn.reconstruction.topk_selection import node_expand_diffuse
                    node_selbool, edge_selbool = node_expand_diffuse(
                        node_weights[track_mask], edge_weights[tr_tr_mask],
                        graphs[i][('tracks', 'to', 'tracks')].edge_index,
                        seed_thr=float(self.configs.get("node_expand_seed_thr", 0.5)),
                        method=method,
                        alpha=float(self.configs.get("node_expand_ppr_alpha", 0.85)),
                        t=float(self.configs.get("node_expand_heat_t", 3.0)),
                        min_score_frac=float(self.configs.get("node_expand_min_score_frac", 0.05)),
                        edge_thr=float(self.configs.get("node_expand_edge_thr", 0.0)),
                        topk=int(self.configs.get("node_expand_k0", 12)) if self.configs.get("node_expand_k0", 0) else None,
                    )
            elif self.configs.get("node_prune", True):
                node_selbool = node_weights[track_mask] > self.configs["node_prune_thr"]
                edge_mask = true_node_pruning(node_selbool, graphs[i], "tracks", [('tracks', 'to', 'tracks')])
                edge_selbool = edge_weights[tr_tr_mask][edge_mask] > self.configs["edge_prune_thr"]
            else:
                edge_selbool = edge_weights[tr_tr_mask] > self.configs["edge_prune_thr"]

            # ==== 方案E: per-track top-k 边选择 (可选, 不污染默认逻辑) ====
            # 生成候选边集(保留更多低置信真边); 链级打分在 reconstruct_single_evt 中做
            # (方案F 已含 top-k 扩展, 与 edge_topk 互斥)
            edge_topk = self.configs.get("edge_topk", 0)
            if edge_topk > 0 and not node_expand:
                from wmpgnn.reconstruction.topk_selection import topk_edge_selbool
                # 节点剪枝后的图: 边索引/权重已与 graphs[i] 对齐
                pruned_graph = graphs[i]
                edge_index_p = pruned_graph[('tracks', 'to', 'tracks')].edge_index
                edge_w_p = edge_weights[tr_tr_mask][edge_mask] if self.configs.get("node_prune", True) \
                    else edge_weights[tr_tr_mask]
                n_nodes_p = pruned_graph['tracks'].x.shape[0]
                all_nodes = torch.ones(n_nodes_p, dtype=torch.bool, device=self.device)
                edge_selbool = topk_edge_selbool(edge_w_p, edge_index_p, all_nodes, edge_topk)

            if self.configs.get("edge_prune", True):
                edge_pruning(edge_selbool, graphs[i], ('tracks', 'to', 'tracks'))

            if node_expand:
                # 方案F: 边剪枝后将节点收缩到扩展后的 S。
                # 扩展保证所有保留边的端点都在 S 内 (keep_e 端点已加回 S), 收缩不丢边;
                # 收缩后 graph 节点数与 node_selbool 对齐, 下游 pv_des/ft_des 过滤一致。
                true_node_pruning(node_selbool, graphs[i], "tracks", [('tracks', 'to', 'tracks')])

            # 挂载剪枝后剩余边的权重到图 (与 graph tt 边一一对齐), 供链中心性过滤等下游使用
            if node_expand or not self.configs.get("node_prune", True):
                reco_edge_w = edge_weights[tr_tr_mask][edge_selbool]
            else:
                reco_edge_w = edge_weights[tr_tr_mask][edge_mask][edge_selbool]
            graphs[i]._edge_w_reco = reco_edge_w.cpu()

            # 挂载剪枝后的权重/LCA到图上, 供链级打分 (选择MLP) 使用
            if self.configs.get("selection_mlp", ""):
                if not self._scorer_init:
                    # 首次初始化选择 MLP (维度由剪枝后图推断)
                    node_dim = 1 + graphs[i]['tracks'].x.shape[1]
                    edge_dim = 1 + 4 + graphs[i][('tracks', 'to', 'tracks')].edges.shape[1]
                    self.chain_scorer = self._load_scorer(self._sel_ckpt, node_dim, edge_dim, device=self.device)
                    self._scorer_init = True
                graphs[i]._chain_node_w = node_weights[track_mask][node_selbool].cpu() if self.configs.get("node_prune", True) \
                    else node_weights[track_mask].cpu()
                graphs[i]._chain_edge_w = edge_weights[tr_tr_mask][edge_mask][edge_selbool].cpu() if self.configs.get("node_prune", True) \
                    else edge_weights[tr_tr_mask][edge_selbool].cpu()
                graphs[i]._chain_lca = graphs[i][('tracks', 'to', 'tracks')].lca.cpu()
                # 图已完成边剪枝, 剩余边即候选边; 链打分时 edge_mask 传 None

            # attach all part ids again
            graphs[i]["pid_holder"] = temp

            # Apply pruning on pv prediction
            if pv_des is not None:
                evt_pv_des = {"true": y_pv[node_selbool].cpu(), "pred": pred_pv[node_selbool].cpu(),
                              "ip": min_ip_pv[node_selbool].cpu(), "npvs": npvs}
                # 方案7b: 训练用的同一 pv_cluster_head 得分 (供分簇, train/infer 对齐)
                if cluster_pred_pv is not None:
                    evt_pv_des["cluster_pred"] = cluster_pred_pv[node_selbool].cpu()
                # DA 几何基线: raw tr-pv 距离矩阵 (供确定性退火软分配)
                evt_pv_des["geom"] = geom_pv[node_selbool].cpu()
                precomputed_pv_desc.append(evt_pv_des)
            else:
                precomputed_pv_desc.append(None)
            # Apply pruning on ft bool
            if ft_des is not None:
                evt_ft_des = ft_des[track_mask][node_selbool]
                precomputed_ft_desc.append(evt_ft_des.cpu())
            else:
                precomputed_ft_desc.append(None)

        # now multiprocess the reco
        args_list = [(graph.cpu(), pv_desc, ft_desc) for graph, pv_desc, ft_desc in
                     zip(graphs, precomputed_pv_desc, precomputed_ft_desc)]
        with ThreadPool(processes=4) as pool:
            res = list(tqdm(pool.imap(self.reconstruct_single_evt, args_list), total=len(args_list),
                            desc="Reconstructing events", leave=False))

        for r in res:
            try:
                self.sig_df.append(pd.DataFrame(r[0]))
                self.evt_df.append(pd.DataFrame([r[1]]))
            except:
                continue
            

    def _reconstruct_pv_clustered(self, graph, pv_des):
        """方案7 (v39): PV 分簇分层重建。

        事件内 tracks 按 PV 分组, 每簇独立做链重建 (reconstruct_decay), 再合并所有簇的链。
        单图有效节点数从 91-139 降到每簇 20-30, 缓解高连通图跨链干扰 (2B/>2B 事件重灾区)。

        pv_des: {"true": y_pv[n], "pred": pred_pv[n,n_pvs], "ip":..., "npvs":...} (与剪枝后节点对齐)

        Returns:
            (rc_dict, num_clusters_per_order): 合并后的重建链 (键=粒子键, 簇间唯一)
        """
        import torch
        from torch_geometric.data import HeteroData
        from torch_geometric.utils import subgraph

        def _fallback():
            reco_LCA = lca_reco_matrix(graph, mode="reco")
            rc, nc, _ = reconstruct_decay(reco_LCA, get_final_keys(graph).tolist())
            return rc, nc

        try:
            n_nodes = graph['tracks'].x.shape[0]
            assign = self.configs.get("pv_cluster_assign", "pred")
            score_mat = None   # 软成员分簇用: [n, n_pvs] 得分矩阵 (高 = 更可能属于)
            if assign == "da":
                # ==== DA 几何分簇 (确定性退火风格软分配 + 置信度拒绝) ====
                # 距离 D = raw tr-pv 边特征 (-log-IP 类, 越小越近);
                # 归属概率 P_ik = softmax(-D_ik / T), T 是"温度"(控制分配锐利度);
                # 固定 PV 中心 (顶点已知, 无需 DA 的顶点分裂), 退火退化为单温度软分配;
                # 默认 argmax (max 概率 < conf_thr 的 track 归无PV组 = DA 的"噪声 track");
                # pv_cluster_overlap=true 时走软成员: track 加入所有 P >= mem_thr 的簇。
                D = pv_des["geom"].cpu()   # [n, n_pvs]
                if D.shape[1] == 0:
                    pv_of_node = torch.full((D.shape[0],), -1, dtype=torch.long)
                else:
                    T = float(self.configs.get("pv_cluster_da_t", 1.0))
                    conf_thr = float(self.configs.get("pv_cluster_da_conf_thr", 0.0))
                    P = torch.softmax(-D / T, dim=1)
                    score_mat = P
                    pmax, arg = P.max(dim=1)
                    pv_of_node = torch.where(pmax >= conf_thr, arg,
                                             torch.tensor(-1, dtype=torch.long))
            elif assign == "true":
                pv_of_node = pv_des["true"].cpu()
            else:
                # 预测得分矩阵 [n, n_pvs], 每 track 取得分最高的 PV。
                # 方案7b: pv_cluster_assign="cluster_head" 时用训练同一 pv_cluster_head 的得分 (对齐),
                # 否则退回 pv_asso head (pred)。
                if assign == "cluster_head" and "cluster_pred" in pv_des:
                    pred = pv_des["cluster_pred"].cpu()
                else:
                    pred = pv_des["pred"].cpu()
                if pred.shape[1] == 0:   # 无 PV 事件: 全部归无PV簇
                    pv_of_node = torch.full((pred.shape[0],), -1, dtype=torch.long)
                else:
                    score_mat = pred
                    pv_of_node = pred.argmax(dim=1)

            # ==== 允许重合的软成员分簇 (pv_cluster_overlap=true) ====
            # 硬划分 (默认): 每条 track 只属于一个簇 (互斥) -> "切簇伤链" 的根源:
            # 一条链里的模糊 track 被错分到别的簇, 链就断了。
            # 软成员: track 加入所有 score >= mem_thr 的簇 (允许重合), 链至少在一个簇里完整;
            # 所有簇都不达标的 track 归无PV组 (-1)。代价: 重复链需去重 (见下)。
            overlap = bool(self.configs.get("pv_cluster_overlap", False))
            if overlap and score_mat is not None:
                mem_thr = float(self.configs.get("pv_cluster_mem_thr", 0.3))
                membership = score_mat >= mem_thr          # [n, n_pvs]
                clusters = {}
                for p in range(score_mat.shape[1]):
                    members = membership[:, p].nonzero(as_tuple=False).flatten().tolist()
                    if members:
                        clusters[p] = members
                none_mem = (~membership.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
                if none_mem:
                    clusters[-1] = none_mem
            else:
                pv_of_node = pv_of_node.numpy()
                # 按 PV 分组 (PV 索引, -1 = 无关联)
                clusters = {}
                for i, p in enumerate(pv_of_node):
                    clusters.setdefault(int(p), []).append(i)

            # 碎片簇 (小于 min_tracks 的 PV 簇) 并入无PV簇, 防碎片链
            min_tracks = int(self.configs.get("pv_cluster_min_tracks", 3))
            small = [p for p, nodes in clusters.items()
                     if p != -1 and len(nodes) < min_tracks]
            for p in small:
                for i in clusters.pop(p):
                    clusters.setdefault(-1, []).append(i)

            edge_index = graph[('tracks', 'to', 'tracks')].edge_index.cpu()
            lca = graph[('tracks', 'to', 'tracks')].lca.cpu()
            part_keys = get_final_keys(graph).cpu()

            rc_dict, nclust = {}, [0, 0, 0, 0]
            for nodes in clusters.values():
                node_idx = torch.tensor(nodes, dtype=torch.long)
                # num_nodes 必须显式传入: 簇内边为空或节点索引超出时, subgraph 默认用
                # edge_index.max()+1 推导 num_nodes, 会导致 index_to_mask 越界崩溃
                # (smoke: "index is out of bounds for dimension with size 0/N")
                sub_ei, sub_lca = subgraph(node_idx, edge_index, edge_attr=lca,
                                           num_nodes=n_nodes, relabel_nodes=True)
                if sub_ei.shape[1] == 0 or sub_lca is None:
                    continue
                # 轻量子图 (reconstruct_decay 只需 tt 边 + lca + part_keys)
                sub = HeteroData()
                sub['tracks'].x = graph['tracks'].x[node_idx].cpu()
                if hasattr(graph['tracks'], 'part_keys'):
                    sub['tracks'].part_keys = part_keys[node_idx]
                sub[('tracks', 'to', 'tracks')].edge_index = sub_ei
                sub[('tracks', 'to', 'tracks')].lca = sub_lca
                sub_lca_mat = lca_reco_matrix(sub, mode="reco")
                if sub_lca_mat.empty:
                    continue
                sub_rc, sub_nclust, _ = reconstruct_decay(sub_lca_mat, get_final_keys(sub).tolist())
                for g in range(4):
                    nclust[g] += sub_nclust[g]
                # 允许重合时同一链会在多个簇出现: 同一 key 只保留最完整 (node_keys 最长) 的副本
                for k, v in sub_rc.items():
                    if k not in rc_dict or len(v["node_keys"]) > len(rc_dict[k]["node_keys"]):
                        rc_dict[k] = v

            # ==== 允许重合: 重复链去重 ====
            # 软成员分簇后, 同一物理链可能在多个簇被重建, 且副本可能因 key 不同 (缺最小粒子)
            # 而逃过上面的按-key 保留。用"包含度"判据: 若两条链中较小的集合被较大的集合
            # 包含 >= dedup_jaccard (inter / min(|A|,|B|)), 判为同一链的副本, 保留更完整的。
            # 硬划分下各簇链互斥, 此步恒为 no-op。
            if len(rc_dict) > 1:
                dedup_jaccard = float(self.configs.get("pv_cluster_dedup_jaccard", 0.8))
                items = sorted(rc_dict.items(), key=lambda kv: -len(kv[1]["node_keys"]))
                kept = {}
                for k, v in items:
                    key_set = set(v["node_keys"])
                    is_dup = False
                    for k2, v2 in kept.items():
                        k2_set = set(v2["node_keys"])
                        inter = len(key_set & k2_set)
                        smaller = min(len(key_set), len(k2_set))
                        if smaller and inter / smaller >= dedup_jaccard:
                            is_dup = True
                            break
                    if not is_dup:
                        kept[k] = v
                rc_dict = kept

            # 极端情况: 分簇后无任何链 -> 回退全图重建
            if not rc_dict:
                return _fallback()
            return rc_dict, nclust
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[pv_cluster] WARN: {type(e).__name__}: {e} -> 回退全图")
            return _fallback()

    def reconstruct_single_evt(self, args):
        graph, pv_des, ft_des = args

        """Obtain the LCA scores of both true and reco graphs"""
        true_LCA = lca_truth_matrix(graph)
        n_part = graph['tracks'].x.shape[0]   # 事件内节点数 (各分支共用)
        if self.use_lca:
            if self.configs.get("pv_cluster", False) and pv_des is not None:
                # ==== 方案7: PV 分簇分层重建 ====
                rc_dict, r_nclust_order = self._reconstruct_pv_clustered(graph, pv_des)
            else:
                reco_LCA = lca_reco_matrix(graph, mode="reco")
                particle_keys = get_final_keys(graph).tolist()
                rc_dict, r_nclust_order, _ = reconstruct_decay(reco_LCA, particle_keys)
        else:
            reco_LCA = lca_reco_matrix(graph, mode="true")
            particle_keys = get_final_keys(graph).tolist()
            rc_dict, r_nclust_order, _ = reconstruct_decay(reco_LCA, particle_keys)

        # ==== 候选衰变链选择 MLP (可选): 逐链打分, 过滤低分链 ====
        if self.chain_scorer is not None and rc_dict != {}:
            from wmpgnn.reconstruction.topk_selection import score_chains
            node_w = graph._chain_node_w
            edge_w = graph._chain_edge_w
            lca = graph._chain_lca
            scores = score_chains(self.chain_scorer, graph, node_w, edge_w, lca,
                                  None, rc_dict, particle_keys)
            # 过滤: 保留 score >= 最高分 - margin (默认保留所有, 由配置决定)
            margin = self.configs.get("selection_mlp_margin", None)
            if margin is not None and scores:
                best = max(scores.values())
                keep = {k for k, s in scores.items() if s >= best - margin}
                rc_dict = {k: v for k, v in rc_dict.items() if k in keep}
            # 记录每链得分, 便于分析
            graph._chain_scores = scores

        # ==== 链中心性过滤器 (可选): 剔除无清晰"根"(B介子)的链, 压低 NoneIso ====
        # 灵感: 传染病溯源的 rumor centrality —— 真实衰变链是"有根的树", 噪声团块无根。
        if self.configs.get("chain_center_filter", False) and rc_dict != {}:
            from wmpgnn.reconstruction.topk_selection import filter_chains_by_center
            clarity_thr = float(self.configs.get("chain_center_clarity", 0.0))
            tree_thr = float(self.configs.get("chain_center_tree_thr", 0.7))
            if clarity_thr > 0 and hasattr(graph, "_edge_w_reco"):
                # 键空间修复 (2026-08-17): 与 chain_lca 同理, rc_dict 的 node_keys 是
                # 粒子键, 需映射为图节点索引后再过滤
                key_to_idx = {int(k): i for i, k in enumerate(particle_keys)}
                rc_dict_idx = {}
                for ck, cluster in rc_dict.items():
                    rc_dict_idx[ck] = {'node_keys': [key_to_idx.get(int(n), -1) for n in cluster['node_keys']]}
                _kept, info = filter_chains_by_center(
                    rc_dict_idx, graph[('tracks', 'to', 'tracks')].edge_index,
                    graph._edge_w_reco, clarity_thr, tree_thr)
                kept_ck = set(_kept.keys())
                rc_dict = {ck: c for ck, c in rc_dict.items() if ck in kept_ck}
                graph._chain_center_info = info

        # ==== 链级 LCA 物理置信度判据 (可选): 剔除由低置信 LCA 边构成的链 ====
        # "最物理"判据: 真链的链内边应是模型高置信的非背景边; 噪声链是勉强/误判成的。
        # 与方案 F/F' (用 edge_weight 选候选) 不重合: 这里判断"已进候选的链是否物理自洽"。
        # chain_lca_filter=true  -> 按 conf_thr 实际过滤 rc_dict
        # chain_lca_record=true  -> 只记录每条链的 conf (供后处理多阈值对比), 不过滤
        #
        # 键空间修复 (2026-08-17): rc_dict 的 node_keys 是粒子键 (particle_keys),
        # 而 filter_chains_by_lca/chain_lca_score 内部用 torch.isin 匹配 edge_index
        # (图节点索引 0..N-1) -> 必须先映射键->索引, 否则所有链 conf=0 被全过滤 (NotFound 100%)。
        self._chain_lca_info = {}
        if (self.configs.get("chain_lca_filter", False) or self.configs.get("chain_lca_record", False)) \
                and rc_dict != {}:
            from wmpgnn.reconstruction.topk_selection import filter_chains_by_lca
            key_to_idx = {int(k): i for i, k in enumerate(particle_keys)}
            rc_dict_idx = {}
            for ck, cluster in rc_dict.items():
                rc_dict_idx[ck] = {'node_keys': [key_to_idx.get(int(n), -1) for n in cluster['node_keys']]}
            _kept, self._chain_lca_info = filter_chains_by_lca(
                rc_dict_idx, graph[('tracks', 'to', 'tracks')].edge_index,
                graph[('tracks', 'to', 'tracks')].lca,
                conf_thr=1e-9)  # 不实际过滤, 只预计算所有链的 conf/class2_frac
            if self.configs.get("chain_lca_filter", False):
                conf_thr = float(self.configs.get("chain_lca_conf_thr", 0.0))
                class2_thr = self.configs.get("chain_lca_class2_thr", None)
                class2_thr = float(class2_thr) if class2_thr not in (None, "", 0) else None
                if conf_thr > 0:
                    kept = {}
                    for ck, cluster in rc_dict.items():
                        conf, cf = self._chain_lca_info[ck]
                        if conf >= conf_thr and (class2_thr is None or cf >= class2_thr):
                            kept[ck] = cluster
                    rc_dict = kept

        particle_keys = get_truth_part_keys(graph).tolist()

        particle_ids = list(map(particle_name, get_truth_part_ids(graph).numpy()))

        tc_dict, t_nclust_order, max_chain_depth = reconstruct_decay(true_LCA, particle_keys,
                                                                     particle_ids=particle_ids,
                                                                     truth_level_simulation=1)
        if tc_dict != {}:
            part_heavy_h = flatten([tc_dict[tc_firstkey]['node_keys'] for tc_firstkey in tc_dict.keys()])
            n_part_heavy_h = len(part_heavy_h)
            n_bkg_part = n_part - n_part_heavy_h

            if rc_dict != {}:
                sel_part = flatten(
                    [rc_dict[tc_firstkey]['node_keys'] for tc_firstkey in rc_dict.keys()])
                n_sel_part = len(sel_part)
                n_sel_heavy_h = len(list(set(sel_part).intersection(part_heavy_h)))
                n_sel_bkg_part = n_sel_part - n_sel_heavy_h
            else:
                n_sel_part, n_sel_heavy_h, n_sel_bkg_part = 0, 0, 0

            perfect_evt_reco = 1  # Flag for perfect event reco
            if n_sel_bkg_part > 0:
                perfect_evt_reco = 0

            # Looping over reco candidates
            sig_dict_holder = []
            for tc_key in tc_dict.keys():
                sig_dict = {'NumParticlesInEvent': n_part,
                            "PerfectReco": 0, "AllParticles": 0, "NoneIso": 0, "PartReco": 0, "NotFound": 0,
                            "NumBkgParticles_noniso": -999,
                            "chain_lca_conf": -1.0, "chain_lca_class2_frac": -1.0}
                tc = tc_dict[tc_key]
                sig_dict["SigMatch"] = 0
                if self.signal:
                    labels = tc['labels']
                    mothers = [label[3:] for label in labels if 'c' == label[0]]
                    node_keys = tc['node_keys']
                    daughters = [label.split(':')[1] for label in labels if
                                 int(float(label.split(':')[0][1:])) in node_keys]
                    if match_decays(daughters, self.signal[0]['daughters']) or match_decays(daughters,
                                                                                            self.signal[1][
                                                                                                'daughters']):
                        check_mothers1 = True
                        check_mothers2 = True
                        for i in range(len(self.signal[0]['mothers'])):
                            if self.signal[0]['mothers'][i] not in mothers:
                                check_mothers1 = False
                            if self.signal[1]['mothers'][i] not in mothers:
                                check_mothers2 = False
                        sig_dict["SigMatch"] = int(check_mothers1 or check_mothers2)

                sig_dict["NumSignalParticles"] = len(tc['node_keys'])

                if tc_key in rc_dict.keys():
                    perfect_sig_reco = int(
                        rc_dict[tc_key]['node_keys'] == tc['node_keys']
                        and rc_dict[tc_key]['LCA_values'] == tc['LCA_values']
                    )
                else:
                    perfect_sig_reco = 0
                perfect_evt_reco *= perfect_sig_reco

                for rc_key, rc in rc_dict.items():
                    true_in_reco = np.sum(np.isin(tc['node_keys'], rc['node_keys'])) / len(tc['node_keys'])
                    if rc['node_keys'] == tc['node_keys']:
                        sig_dict["AllParticles"] = 1
                        if rc['LCA_values'] == tc['LCA_values']:
                            sig_dict["PerfectReco"] = 1
                        sig_dict["NoneIso"] = sig_dict["PartReco"] = 0
                        # 记录该 truth 链匹配到的重建链的 LCA 物理置信度 (供后处理多阈值对比)
                        if rc_key in self._chain_lca_info:
                            sig_dict["chain_lca_conf"], sig_dict["chain_lca_class2_frac"] = self._chain_lca_info[rc_key]
                        if ft_des is not None:
                            sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                            sig_dict = get_asso_frag(sig_dict, graph, rc)
                        if pv_des is not None:
                            get_pv_asso(sig_dict, graph, rc, pv_des)
                        break
                    elif true_in_reco == 1 and len(rc['node_keys']) > len(tc['node_keys']):
                        sig_dict["NoneIso"] = 1  # background tracks in signal
                        sig_dict["PartReco"] = 0
                        if rc_key in self._chain_lca_info:
                            sig_dict["chain_lca_conf"], sig_dict["chain_lca_class2_frac"] = self._chain_lca_info[rc_key]
                        if ft_des is not None:
                            sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                            sig_dict = get_asso_frag(sig_dict, graph, rc)
                        if pv_des is not None:
                            get_pv_asso(sig_dict, graph, rc, pv_des)
                        break
                    elif 0.2 <= true_in_reco < 1:
                        sig_dict["PartReco"] = 1  # FT decision can not be trusted
                        sig_dict["NumBkgParticles_noniso"] = len(rc['node_keys']) - len(tc['node_keys'])
                        if rc_key in self._chain_lca_info:
                            sig_dict["chain_lca_conf"], sig_dict["chain_lca_class2_frac"] = self._chain_lca_info[rc_key]
                        if ft_des is not None:
                            sig_dict = get_pred_ft(sig_dict, graph, rc, ft_des)
                            sig_dict = get_asso_frag(sig_dict, graph, rc)
                        if pv_des is not None:
                            get_pv_asso(sig_dict, graph, rc, pv_des)
                        break
                    """else:
                        sig_dict["final_pid"] = sig_dict["final_b_score"] = sig_dict["final_bbar_score"] = ""
                        sig_dict["ft_b_score"] = sig_dict["ft_bbar_score"] = 0"""

                if sig_dict["AllParticles"] == 0 and sig_dict["NoneIso"] == 0 and sig_dict["PartReco"] == 0:
                    sig_dict["NotFound"] = 1

                # Get origin B id
                indices = [particle_keys.index(x) for x in tc['node_keys']]
                signal_LCA_id = true_LCA[true_LCA['senders'].isin(indices) | true_LCA['receivers'].isin(indices)][
                    "LCA_id"]
                values, counts = np.unique(signal_LCA_id, return_counts=True)
                max_indices = np.where(counts == counts.max())[0]
                if len(max_indices) == 1:
                    sig_dict["B_id"] = values[max_indices[0]]
                else:
                    candidate_lca_ids = values[max_indices]
                    candidates_df = true_LCA[true_LCA['LCA_id'].isin(candidate_lca_ids)]
                    max_chain_per_lca = candidates_df.groupby('LCA_id')['TrueFullChainLCA'].max()
                    sig_dict["B_id"] = max_chain_per_lca.idxmax()
                if "EVENTNUMBER" in graph.keys():
                    sig_dict["EVENTNUMBER"] = graph["EVENTNUMBER"].item()
                    sig_dict["RUNNUMBER"] = graph["RUNNUMBER"].item()
                if "num_pvs" in graph.keys():
                    sig_dict["num_pvs"] = graph["num_pvs"].item()
                else:
                    sig_dict["num_pvs"] = graph["pvs"].x.shape[0]
                if "is_whiten" in graph.keys():
                    sig_dict["is_whiten"] = graph["is_whiten"].item()
                sig_dict_holder.append(sig_dict)

            evt_dict = {'NumParticlesInEvent': n_part,
                        'NumParticlesFromHeavyHadronInEvent': n_part_heavy_h,
                        'NumBackgroundParticlesInEvent': n_bkg_part,
                        'NumSelectedParticlesInEvent': n_sel_part,
                        'NumSelectedParticlesFromHeavyHadronInEvent': n_sel_heavy_h,
                        'NumSelectedBackgroundParticlesInEvent': n_sel_bkg_part,
                        'NumTruthClustersGen1': t_nclust_order[0],
                        'NumTruthClustersGen2': t_nclust_order[1],
                        'NumTruthClustersGen3': t_nclust_order[2],
                        'NumTruthClustersGen4': t_nclust_order[3],
                        'NumRecoClustersGen1': r_nclust_order[0],
                        'NumRecoClustersGen2': r_nclust_order[1],
                        'NumRecoClustersGen3': r_nclust_order[2],
                        'NumRecoClustersGen4': r_nclust_order[3],
                        'MaxTruthFullChainDepthInEvent': max_chain_depth,
                        'PerfectEventReconstruction': perfect_evt_reco,
                        'NumTrueSignalsInEvent': len(tc_dict.keys()),
                        'NumRecoSignalsInEvent': len(rc_dict.keys()),
                        }
            return sig_dict_holder, evt_dict
