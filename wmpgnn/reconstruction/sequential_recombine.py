"""方向3: 可学习顺序重组合解码器 (anti-kT 式层次聚类)。

动机
----
当前重建的最后一步是 "取连通分量" (`reco_helper.reconstruct_decay` 里的
`nx.connected_components`, 见该函数 line 214 / 310)。它**不保证排他性**, 也**不显式决定链数**:
一条径迹可能被多条链同时占用; 最终链数完全由"哪些边被模型判成 LCA>0"落出来。

物理人最熟悉的替代品是 jet clustering 的 sequential recombination (anti-kT 家族):
反复合并"当前最近"的一对, **天然独占** (每个对象只用一次)、**不需要预设簇数**。
本模块把"距离度量"换成**可学习的** (由已有的边分数 p_e + 几何共同决定), 然后跑标准的
层次聚合过程, 用它取代连通分量。

距离度量 (对称, 对端点交换不变)
--------------------------------
  learned : d = -ln(p_e + eps)              (分数越高 -> 越近 -> 越早合并)
  geom    : d = 纯几何距离 (无学习对照)
  mix     : d = -ln(p_e + eps) + lam * d_geom   (lam 可为固定值或可学习标量)

聚合过程 (标准 sequential recombination)
----------------------------------------
每次取"当前最近的一对簇"合并。簇间距离 (linkage) 两种, 可用 `seqrec_linkage` 切换:
  nearest (默认) : 簇间距离 = 成员对间**最小**距离 (single-linkage / 最近邻)。
                   用 Kruskal 式排序边 + 并查集实现 -> 与 "按距离阈值取连通分量" 完全等价,
                   当 cut = +inf 时**逐位还原**原连通分量 (可作回归测试)。
  centroid       : 簇间距离 = 两簇成员对距离的**平均** (average-linkage; 这是在没有
                   度量嵌入的情况下对 "质心距离" 的标准层次聚类对应物)。O(N^3) 朴素实现,
                   仅在小图上用。
合并持续到最小距离 > cut (seqrec_cut), 或缩并到只剩 N 个簇 (seqrec_nclusters)。

排他性: 输出是事件的**划分**——每条径迹只属于一个簇 (这是本方向的核心卖点, 由单链并查集保证)。

确定性: 并列距离按 (distance, i, j) 字典序打破平局 -> 输出可复现。

边界: 空事件 / 单径迹 / 无候选边 / 全孤立 -> 返回全 -1 标签 (无长度>1 的簇), 安全。

耗时估计: single-linkage 是 O(M log M) (M=候选边数, 典型几百~几千/事件)。N~150 时
单事件 < 1 ms; 3000 事件 ~ 数秒 (纯 CPU)。
"""
import numpy as np
import torch

EPS = 1e-6


# ---------------------------------------------------------------------------
# 距离度量
# ---------------------------------------------------------------------------
def metric_distance(p, metric="mix", lam=1.0, d_geom=None, eps=EPS):
    """边级距离 (对称, 不依赖端点顺序)。

    p     : [M] 存活概率 (模型边分数), 越大越"近"
    metric: "learned" | "geom" | "mix"
    lam   : mix 里几何项的权重 (标量, 可学习 -> 传 torch tensor 也支持)
    d_geom: [M] 纯几何距离 (metric="geom"/"mix" 时必需)
    """
    if metric == "geom":
        if d_geom is None:
            raise ValueError("metric='geom' 需要 d_geom")
        return d_geom
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(np.array(p, dtype=np.float64))
    else:
        p = torch.as_tensor(p)
    lterm = -torch.log(p.clamp(min=eps))
    if metric == "learned":
        return lterm
    if metric == "mix":
        if d_geom is None:
            raise ValueError("metric='mix' 需要 d_geom")
        return lterm + lam * torch.as_tensor(d_geom, dtype=lterm.dtype)
    raise ValueError(f"未知 seqrec 度量: {metric}")


# ---------------------------------------------------------------------------
# 纯几何边距离 (绕开上游 log_DOCA_reco 噪声列, 与 derive_pruning_features 同式)
# ---------------------------------------------------------------------------
def geom_edge_features(x, edge_index, nc, ns):
    """单事件图上的几何边特征 (DOCA / |Δ起点| / |Δz| / collinearity)。

    直线模型 P(τ)=A+τu (过生产顶点 A, 方向=单位动量), 与
    dfei_lightning_module.derive_pruning_features 的 derived_vertex_geom **同式**;
    这里为单事件 (非 batched) 版本, 绕开上游 stored 的 log_DOCA_reco (t1 符号 bug)。

    x   : [N, >=6] 归一化节点特征 (前 6 列 = px,py,pz,xProd,yProd,zProd)
    nc/ns: 原特征名 -> 归一化 center/scale (用于反归一化)
    Returns dict: doca / d_start / dz / collin (均 torch [M])
    """
    x = torch.as_tensor(x)
    ei = torch.as_tensor(edge_index)
    if ei.dim() != 2 or ei.shape[1] != 2:
        raise ValueError(f"edge_index 需为 [M,2], 实际 {tuple(ei.shape)}")
    a, b = ei[:, 0].long(), ei[:, 1].long()

    def raw(name, col):
        return x[:, col] * float(ns.get(name, 1.0)) + float(nc.get(name, 0.0))

    px, py, pz = raw("px_reco", 0), raw("py_reco", 1), raw("pz_reco", 2)
    xp, yp, zp = raw("xProd_reco", 3), raw("yProd_reco", 4), raw("zProd_reco", 5)
    pmod = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + 1e-6)
    u = torch.stack([px, py, pz], 1) / pmod.unsqueeze(1)

    A3 = torch.stack([xp[a], yp[a], zp[a]], 1)
    B3 = torch.stack([xp[b], yp[b], zp[b]], 1)
    ua, ub = u[a], u[b]
    w0 = A3 - B3
    cb = (ua * ub).sum(-1)
    dv = (ua * w0).sum(-1)
    ev = (ub * w0).sum(-1)
    dnv = 1.0 - cb * cb
    parv = dnv.abs() < 1e-3                                   # 近平行: 垂足病态
    dnsv = torch.where(parv, torch.ones_like(dnv), dnv)
    tav = ((cb * ev - dv) / dnsv).clamp(-1e4, 1e4)
    tbv = ((ev - cb * dv) / dnsv).clamp(-1e4, 1e4)
    F1 = A3 + tav.unsqueeze(1) * ua
    F2 = B3 + tbv.unsqueeze(1) * ub
    d_par = torch.linalg.norm(w0 - dv.unsqueeze(1) * ua, dim=1)   # 平行: w 的垂距 (良态)
    doca = torch.where(parv, d_par, torch.linalg.norm(F1 - F2, dim=1))
    d_start = torch.linalg.norm(w0, dim=1)
    dz = (zp[a] - zp[b]).abs()
    d_perp = doca
    collin = d_perp / (dz + 1.0)
    nan2num = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return dict(doca=nan2num(doca), d_start=nan2num(d_start),
                dz=nan2num(dz), collin=nan2num(collin))


def combine_geom(gdict, w_doca=1.0, w_start=0.0, w_dz=0.0, w_collin=0.0, scale=100.0):
    """把几何量线性组合成单一几何距离 (默认纯 DOCA/100mm)。"""
    d = (w_doca * gdict["doca"] + w_start * gdict["d_start"]
         + w_dz * gdict["dz"] + w_collin * gdict["collin"])
    return d / scale


# ---------------------------------------------------------------------------
# 核心: sequential recombination (层次聚合)
# ---------------------------------------------------------------------------
def sequential_recombine(n_nodes, edges, distances, cut=None, n_clusters=None,
                         linkage="nearest"):
    """anti-kT 式顺序重组合。

    n_nodes   : 事件内径迹数 N
    edges     : [M,2] (int) 节点对 (无向)
    distances : [M]   对应距离 (越小越"近")
    cut       : 距离阈值; 最小距离 > cut 时停止合并。None -> +inf (合并所有可达对)
    n_clusters: 若给定, 即使超过 cut 也继续合并直到簇数 <= n_clusters (可选)
    linkage   : "nearest" (single-linkage, Kruskal 实现) | "centroid" (average-linkage)

    Returns
    -------
    labels : np.ndarray [N] int; -1 = 未与任何节点合并 (孤立 / 超阈值)
    """
    labels = -np.ones(int(n_nodes), dtype=int)
    edges = np.asarray(edges)
    distances = np.asarray(distances, dtype=float)
    N = int(n_nodes)
    if N == 0 or edges.size == 0:
        return labels

    # 去重 (同一对取最小距离); 剔除自环/越界
    keep = (edges[:, 0] != edges[:, 1]) & (edges[:, 0] >= 0) & (edges[:, 0] < N) \
        & (edges[:, 1] >= 0) & (edges[:, 1] < N)
    edges, distances = edges[keep], distances[keep]
    if edges.shape[0] == 0:
        return labels
    keyed = {}
    for (i, j), d in zip(edges.tolist(), distances.tolist()):
        k = (min(i, j), max(i, j))
        if k not in keyed or d < keyed[k]:
            keyed[k] = d
    ks = np.array(list(keyed.keys()), dtype=int)
    ds = np.array([keyed[(int(i), int(j))] for i, j in ks], dtype=float)

    cut_val = float("inf") if cut is None else float(cut)
    # 确定性排序: (距离, i, j) 字典序
    order = np.lexsort((ks[:, 1], ks[:, 0], ds))
    ks, ds = ks[order], ds[order]

    parent = np.arange(N)

    def find(z):
        while parent[z] != z:
            parent[z] = parent[parent[z]]
            z = parent[z]
        return z

    def n_active():
        return len({find(i) for i in range(N)})

    if linkage == "nearest":
        for (i, j), d in zip(ks.tolist(), ds.tolist()):
            if d > cut_val and (n_clusters is None or n_active() <= n_clusters):
                break
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)     # 确定性: 小根为父
        # 输出标签
        roots = np.array([find(i) for i in range(N)])
        uniq = {r: k for k, r in enumerate(sorted(set(roots.tolist())))}
        labels = np.array([uniq[r] for r in roots.tolist()], dtype=int)
        # 孤立节点 -> -1
        cnt = np.bincount(labels)
        labels = np.where(cnt[labels] > 1, labels, -1)
        return labels

    if linkage == "centroid":
        # average-linkage: 簇间距离 = 跨簇成员对距离均值 (无度量嵌入时的"质心"对应物)
        # 以节点为初始簇, 维护簇间距离矩阵的增量更新。C 个活跃簇时每次合并 O(C^2)。
        # 把 "可达" 用有限距离表示; 无边 -> inf (永不合并)。
        D = np.full((N, N), np.inf)
        for (i, j), d in zip(ks.tolist(), ds.tolist()):
            D[i, j] = D[j, i] = d
        active = list(range(N))
        memb = {i: [i] for i in range(N)}
        while len(active) > 1:
            sub = np.array(active)
            Dsub = D[np.ix_(sub, sub)]
            flat = np.argmin(Dsub)
            a_idx, b_idx = divmod(int(flat), len(sub))
            amin = Dsub[a_idx, b_idx]
            ca, cb = active[a_idx], active[b_idx]
            if not np.isfinite(amin):
                break
            if amin > cut_val and (n_clusters is None or len(active) <= n_clusters):
                break
            na, nb = len(memb[ca]), len(memb[cb])
            for x in active:
                if x == ca or x == cb:
                    continue
                d1, d2 = D[ca, x], D[cb, x]
                if np.isfinite(d1) or np.isfinite(d2):
                    v1 = 0.0 if not np.isfinite(d1) else d1
                    v2 = 0.0 if not np.isfinite(d2) else d2
                    D[ca, x] = D[x, ca] = (na * v1 + nb * v2) / (na + nb)
                else:
                    D[ca, x] = D[x, ca] = np.inf
            memb[ca] = memb[ca] + memb[cb]
            active.remove(cb)
        for c in active:
            if len(memb[c]) > 1:
                for node in memb[c]:
                    labels[node] = c
        # 重映射到 0..K-1
        uniq = sorted(set(labels[labels >= 0].tolist()))
        remap = {r: k for k, r in enumerate(uniq)}
        labels = np.array([remap.get(int(v), -1) for v in labels], dtype=int)
        return labels

    raise ValueError(f"未知 linkage: {linkage}")


def cluster_sizes(labels):
    """{label: size} (忽略 -1)。"""
    labs, cnts = np.unique(labels[labels >= 0], return_counts=True)
    return {int(l): int(c) for l, c in zip(labs, cnts)}


# ---------------------------------------------------------------------------
# 与重建管线对接: 用 seqrec 划分取代连通分量
# ---------------------------------------------------------------------------
def seqrec_chains(lca_matrix, particle_keys, n_nodes, metric="mix", cut=None,
                  linkage="nearest", lam=1.0, n_clusters=None):
    """用 seqrec 在 **LCA 候选边** 上做事件划分, 再逐簇调用原 reconstruct_decay 解析链内结构。

    lca_matrix : pd.DataFrame, 列 = ['senders','receivers','LCA_dec','p','dgeom']。
                 'p' = 该边模型存活概率; 'dgeom' = 几何距离。列 'p'/'dgeom' 可缺省
                 (缺省则退化为纯几何 / 纯拓扑)。
    particle_keys: 事件内逐径迹键 (长度 = n_nodes)
    返回 (rc_dict, num_clusters_per_order) —— 与 reco_helper.reconstruct_decay 同构,
         直接喂给 reconstruct_single_evt 的下游比较 (node_keys / LCA_values)。
    """
    from wmpgnn.reconstruction.reco_helper import reconstruct_decay

    rc_dict = {}
    nclust = [0, 0, 0, 0]
    if lca_matrix is None or len(lca_matrix) == 0:
        return rc_dict, nclust

    # 候选边: 与原 reconstruct_decay 一致 (LCA_dec > 0)
    cand = lca_matrix[lca_matrix["LCA_dec"] > 0]
    if len(cand) == 0:
        return rc_dict, nclust

    e = cand[["senders", "receivers"]].to_numpy()
    if "p" in cand.columns:
        p = cand["p"].to_numpy(dtype=float)
    else:
        p = np.ones(len(cand))
    dgeom = cand["dgeom"].to_numpy(dtype=float) if "dgeom" in cand.columns else None
    dist = metric_distance(p, metric=metric, lam=lam,
                           d_geom=dgeom if metric != "learned" else None).detach().cpu().numpy()

    labels = sequential_recombine(n_nodes, e, dist, cut=cut, n_clusters=n_clusters,
                                 linkage=linkage)

    for lab in np.unique(labels[labels >= 0]):
        nodes = np.nonzero(labels == lab)[0]
        if nodes.size < 2:
            continue
        sub_keys = [particle_keys[i] for i in nodes]
        relabel = {int(g): k for k, g in enumerate(nodes.tolist())}
        sub = cand[cand["senders"].isin(relabel) & cand["receivers"].isin(relabel)].copy()
        sub["senders"] = sub["senders"].map(relabel)
        sub["receivers"] = sub["receivers"].map(relabel)
        sub_rc, sub_nclust, _ = reconstruct_decay(
            sub[["senders", "receivers", "LCA_dec"]], sub_keys)
        for g in range(4):
            nclust[g] += sub_nclust[g]
        # 同一物理链可能因键冲突被覆盖: 保留更完整者
        for k, v in sub_rc.items():
            if k not in rc_dict or len(v["node_keys"]) > len(rc_dict[k]["node_keys"]):
                rc_dict[k] = v
    return rc_dict, nclust


def cluster_chains_from_labels(labels, lca_matrix, particle_keys):
    """[方向2] 用逐节点标签 (>=0; -1=背景) 划分事件, 再逐簇调用 reconstruct_decay 解析链内结构。

    与 seqrec_chains 的"尾段"完全同构, 只是划分来源不同 (集合预测头 slot 归属 vs
    sequential recombination)。lca_matrix: DataFrame(columns 含 senders/receivers/LCA_dec)。
    返回 (rc_dict, num_clusters_per_order) —— 与 reconstruct_decay 同构。
    排他性: labels 是由逐径迹 argmax 得到的**单一标签** -> 每条径迹只属于一个簇。
    """
    import numpy as np
    from wmpgnn.reconstruction.reco_helper import reconstruct_decay

    rc_dict = {}
    nclust = [0, 0, 0, 0]
    if lca_matrix is None or len(lca_matrix) == 0:
        return rc_dict, nclust
    cand = lca_matrix[lca_matrix["LCA_dec"] > 0]
    if len(cand) == 0:
        return rc_dict, nclust
    labels = np.asarray(labels, dtype=int)
    for lab in np.unique(labels[labels >= 0]):
        nodes = np.nonzero(labels == lab)[0]
        if nodes.size < 2:
            continue
        sub_keys = [particle_keys[i] for i in nodes]
        relabel = {int(g): k for k, g in enumerate(nodes.tolist())}
        sub = cand[cand["senders"].isin(relabel) & cand["receivers"].isin(relabel)].copy()
        sub["senders"] = sub["senders"].map(relabel)
        sub["receivers"] = sub["receivers"].map(relabel)
        sub_rc, sub_nclust, _ = reconstruct_decay(
            sub[["senders", "receivers", "LCA_dec"]], sub_keys)
        for g in range(4):
            nclust[g] += sub_nclust[g]
        for k, v in sub_rc.items():
            if k not in rc_dict or len(v["node_keys"]) > len(rc_dict[k]["node_keys"]):
                rc_dict[k] = v
    return rc_dict, nclust


# ---------------------------------------------------------------------------
# 可学习参数 (供模型侧优化器)
# ---------------------------------------------------------------------------
class SeqRecParams(torch.nn.Module):
    """可学习标量: log_lam (mix 度量几何权重) / log_cut (合并阈值)。

    仅当对应开关打开时才注册为 Parameter (否则固定值走 buffer, 不参与优化)。
    cut 的物理标定: d = -ln(p), 故 cut 与概率阈值 p_thr 的关系是 cut = -ln(p_thr)。
    """
    def __init__(self, lam=1.0, cut=None, lam_learn=False, cut_learn=False):
        super().__init__()
        cut_init = float("inf") if cut is None else float(cut)
        self.lam_learn = bool(lam_learn)
        self.cut_learn = bool(cut_learn)
        if lam_learn:
            self.log_lam = torch.nn.Parameter(torch.tensor(float(np.log(max(lam, 1e-6)))))
        else:
            self.register_buffer("log_lam", torch.tensor(float(np.log(max(lam, 1e-6)))))
        if cut_learn:
            # 用一个有限的软阈值参数化; 用 softplus 保证 > 0
            self.cut_raw = torch.nn.Parameter(torch.tensor(_inv_softplus(max(cut_init, 1e-3))))
        else:
            self.register_buffer("cut_raw", torch.tensor(_inv_softplus(1e-3)))

    @property
    def lam(self):
        return torch.exp(self.log_lam)

    @property
    def cut(self):
        if not self.cut_learn:
            return None
        return torch.nn.functional.softplus(self.cut_raw)


def _inv_softplus(y):
    import math
    return math.log(math.expm1(y)) if y > 1e-6 else math.log(y + 1e-6)


# ---------------------------------------------------------------------------
# 可微代理损失: 训练度量 (lam) 与阈值 (cut)
# ---------------------------------------------------------------------------
def seqrec_metric_loss(p, d_geom, y_intra, metric="mix", lam=1.0, margin=0.0, tau=0.5, eps=EPS):
    """链内 (真姐妹) 边距离应 < 链间边距离 (softplus ranking)。训练 lam / 边头。

    y_intra: [M] bool/0-1, 1 = 该边两端在同一真值链 (真姐妹)。
    用 softplus((d_intra - d_inter + margin)/tau): 即使当前已正确分离 (差值为负) 仍留
    非零梯度 (避免 relu 在 0 处完全截断 -> lam 学不动)。返回标量 loss (可微)。
    """
    d = metric_distance(p, metric=metric, lam=lam, d_geom=d_geom, eps=eps)
    y = torch.as_tensor(y_intra).bool()
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    d_intra = d[y].mean()
    d_inter = d[~y].mean()
    return torch.nn.functional.softplus((d_intra - d_inter + margin) / max(tau, 1e-6))


def seqrec_cut_loss(p, d_geom, y_intra, metric="mix", lam=1.0, cut=1.0, eps=EPS):
    """训练阈值 cut: 链内距离应低于 cut, 链间距离应高于 cut (双边 hinge)。

    cut 可微标量 (SeqRecParams.cut)。返回标量 loss。
    """
    d = metric_distance(p, metric=metric, lam=lam, d_geom=d_geom, eps=eps)
    y = torch.as_tensor(y_intra).bool()
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    cut = torch.as_tensor(cut, dtype=d.dtype)
    l_in = torch.nn.functional.relu(d[y] - cut).mean()
    l_out = torch.nn.functional.relu(cut - d[~y]).mean()
    return l_in + l_out
