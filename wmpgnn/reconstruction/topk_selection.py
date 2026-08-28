"""top-k 边选择 + 候选衰变链选择 MLP (方案 E + 选择器)

不修改原重建逻辑; 通过 config inference 段的开关启用:
  edge_topk: 15                 # >0 时启用 per-track top-k 边选择 (替代全局硬阈值)
  selection_mlp: "ckpt/path"    # 非空时启用候选衰变链选择
  edge_topk_k_list: [10,15,20]  # 生成候选边集时扫描的 k 值 (默认 [10,15,20,30])

核心思路:
  1. topk_edge_selbool : 每条径迹保留权重最高的 k 条无向关联边 (双向还原)
  2. 由候选边集重建出多个候选衰变链 (reconstruct_decay 的 cluster_dict)
  3. CandidateScorer : 对单条完整衰变链打分. 链内所有节点特征
     (node_weight + 物理特征) 与所有边特征 (edge_weight + LCA概率 + 边物理特征)
     变长 -> set pooling -> 定长 -> 该链的总 likelihood
  4. score_chains : 对事件内所有候选链打分, 返回排序, 供下游选最高分链
"""
import numpy as np
import torch
import torch.nn as nn

from wmpgnn.reconstruction.reco_helper import flatten


# ============ 1. per-track top-k 边选择 ============

def topk_edge_selbool(edge_weights, edge_index, node_selbool, k):
    """per-track top-k 边选择。

    对每条径迹, 保留与其相连、edge_weight 最高的 k 条无向关联边;
    返回 [E] 布尔掩码 (原始有向边空间, 无向对的两条方向边同时保留)。

    Args:
        edge_weights: [E] 事件内边置信度 (已按节点剪枝前的全事件边对齐)
        edge_index:   [2, E] 事件内边索引 (有向, 含双向)
        node_selbool: [N] 节点级保留掩码 (节点剪枝结果)
        k:            每条径迹最多保留的关联边数 (k<=0 表示不限制)
    """
    device = edge_weights.device
    E = edge_weights.shape[0]
    a, b = edge_index[0], edge_index[1]
    if k is None or k <= 0:
        return torch.ones(E, dtype=torch.bool, device=device) & node_selbool[a] & node_selbool[b]

    # 两端节点都被保留的边才进入候选
    node_keep_edge = node_selbool[a] & node_selbool[b]
    keep = torch.zeros(E, dtype=torch.bool, device=device)

    # 只处理无向对 (a<b) 一次, 避免双向边重复占配额
    undir = (a < b) & node_keep_edge
    idx = torch.nonzero(undir).flatten()
    if idx.numel() == 0:
        return keep

    ew = edge_weights[idx]
    ea, eb = a[idx], b[idx]

    # 按权重降序贪心: 两端配额都未满才选中
    order = torch.argsort(ew, descending=True)
    n_nodes = node_selbool.shape[0]
    count = torch.zeros(n_nodes, dtype=torch.long, device=device)
    sel = torch.zeros(idx.numel(), dtype=torch.bool, device=device)
    for m in order.tolist():
        na, nb = int(ea[m]), int(eb[m])
        if count[na] < k and count[nb] < k:
            sel[m] = True
            count[na] += 1
            count[nb] += 1

    # 无向对掩码 -> 原始有向边空间, 并还原双向
    sel_undir = idx[sel]
    keep[sel_undir] = True
    rev_map = {}
    for i in range(E):
        x, y = int(a[i]), int(b[i])
        rev_map.setdefault((x, y), i)
    for i in sel_undir.tolist():
        j = rev_map.get((int(b[i]), int(a[i])))
        if j is not None:
            keep[j] = True
    return keep


# ============ 1b. 方案F: seed-expand 连通性保留剪枝 ============

def node_expand_selbool(node_weight, edge_weight, edge_index, k0=12, seed_thr=0.5,
                        max_hop=3, decay=1, max_nodes=200, edge_thr=0.0):
    """seed-expand 连通性保留剪枝 (方案 F, 2026-08-14)。

    从硬剪枝幸存节点(种子)出发, 每跳对种子节点的**关联边**做 per-node top-k,
    把 top-k 边连到的被剪节点加回; 迭代且 k 每跳递减, 直到收敛/k=0。
    核心保证: 链上任一节点幸存 -> 整条链通过 top-k 边扩展找回 (连通性恢复),
    消除硬阈值剪枝的 "链级 AND 存活" 级联丢失。

    必须在节点剪枝 (true_node_pruning, 原地删节点) **之前**、在事件全图上调用。

    Args:
        node_weight: [N,1] 节点置信度 (node_weights, sigmoid 后)
        edge_weight: [E,1] 边置信度 (edge_weights, sigmoid 后)
        edge_index:  [2,E] 有向边 (含双向)
        k0:          首跳 per-node top-k (扩展宽度)
        seed_thr:    种子节点阈值 (node_weight > thr 即幸存)
        max_hop:     最大扩展跳数 (第 h 跳宽度 = max(k0 - h*decay, 1))
        decay:       每跳 k 衰减量
        max_nodes:   全局节点上限 budget (防高连通事件雪崩)
        edge_thr:    扩展内边最低权重 (默认 0 保留所有, 即"轻软")

    Returns:
        (node_selbool [N], edge_selbool [E]) 扩展后的节点/边保留掩码
    """
    device = node_weight.device
    N = node_weight.shape[0]
    E = edge_weight.shape[0]
    a, b = edge_index[0], edge_index[1]

    S = (node_weight.squeeze(-1) > seed_thr).clone()
    keep_e = torch.zeros(E, dtype=torch.bool, device=device)
    k = k0

    for hop in range(max_hop):
        # 当前种子节点的关联边 (无向语义: 任一端点在 S 内, 含跨出 S 的边)
        incident = S[a] | S[b]
        if not incident.any():
            break
        sel = _per_node_topk_incident(edge_weight.squeeze(-1), a, b, incident, S, k)
        if edge_thr > 0:
            sel = sel & (edge_weight.squeeze(-1) > edge_thr)
        if not sel.any():
            break
        keep_e = keep_e | sel
        # 加回被 keep 边连到的节点 (整条链找回)
        new_S = S.clone()
        new_S[a[keep_e]] = True
        new_S[b[keep_e]] = True
        if int(new_S.sum()) > max_nodes:  # budget 兜底
            break
        if torch.equal(new_S, S):
            break
        S = new_S
        k = max(k - decay, 1)

    return S, keep_e


def _per_node_topk_incident(ew, a, b, incident, S, k):
    """对 S 内每个节点, 从其关联边中按权重保留 top-k (贪心配额, 双向还原)。

    只处理无向对 (a<b) 一次; 配额仅对 S 内端点生效 (非 S 端点是被拉入者, 不占配额);
    一条边两端都在 S 时, 两端配额都未满才选中 (与 topk_edge_selbool 一致)。
    """
    device = ew.device
    E = ew.shape[0]
    keep = torch.zeros(E, dtype=torch.bool, device=device)

    cand = incident & (a < b)
    cidx = torch.nonzero(cand).flatten()
    if cidx.numel() == 0:
        return keep

    ew_c = ew[cidx]
    ea, eb = a[cidx], b[cidx]
    order = torch.argsort(ew_c, descending=True)

    n_nodes = S.shape[0]
    count = torch.zeros(n_nodes, dtype=torch.long, device=device)
    sel = torch.zeros(cidx.numel(), dtype=torch.bool, device=device)
    for m in order.tolist():
        na, nb = int(ea[m]), int(eb[m])
        if (not S[na] or count[na] < k) and (not S[nb] or count[nb] < k):
            sel[m] = True
            if S[na]:
                count[na] += 1
            if S[nb]:
                count[nb] += 1

    # 无向对掩码 -> 原始有向边空间, 并还原双向
    sel_undir = cidx[sel]
    keep[sel_undir] = True
    rev_map = {}
    for i in range(E):
        x, y = int(a[i]), int(b[i])
        rev_map.setdefault((x, y), i)
    for i in sel_undir.tolist():
        j = rev_map.get((int(b[i]), int(a[i])))
        if j is not None:
            keep[j] = True
    return keep


# ============ 1c. 方案F': 扩散分数 + sweep cut (2026-08-14) ============

def _to_scipy_graph(edge_weight, edge_index, N):
    """把事件图转成 scipy 稀疏矩阵 (加权邻接)。

    tt 边是双向的, 直接用有向边构造即可 (A[i,j] = w(i→j) 与 w(j→i) 同时存在)。
    """
    import numpy as np
    import scipy.sparse as sp
    a = edge_index[0].cpu().numpy()
    b = edge_index[1].cpu().numpy()
    w = edge_weight.squeeze().detach().cpu().numpy()
    A = sp.coo_matrix((w, (a, b)), shape=(N, N)).tocsr()
    return A


def ppr_diffuse(node_mask, edge_weight, edge_index, alpha=0.85, iters=100):
    """个性化 PageRank (PPR) 扩散分数。

    p = (1-α)·s + α·Pᵀp,  P = D⁻¹A (行归一化转移矩阵)。
    直觉: 从种子节点 s 出发, 每次有 α 概率沿边随机游走、1-α 概率跳回种子。
    迭代后 p[i] = "从种子出发的随机游走停在 i 的概率" -> 与种子集连通越强, 分数越高。
    """
    import numpy as np
    N = node_mask.shape[0]
    A = _to_scipy_graph(edge_weight, edge_index, N)
    deg = np.asarray(A.sum(axis=1)).squeeze()
    deg[deg == 0] = 1.0
    P = sp_diags(1.0 / deg) @ A  # P[i,j] = A[i,j]/deg[i]
    s = node_mask.float().cpu().numpy().astype(np.float64)
    total = s.sum()
    if total == 0:
        return torch.zeros(N, device=edge_weight.device)
    s = s / total
    p = s.copy()
    for _ in range(iters):
        p_new = (1.0 - alpha) * s + alpha * (P.T @ p)
        if np.abs(p_new - p).sum() < 1e-10:
            p = p_new
            break
        p = p_new
    return torch.tensor(p, device=edge_weight.device, dtype=torch.float32)


def heat_kernel_diffuse(node_mask, edge_weight, edge_index, t=3.0, iters=30):
    """Heat Kernel 扩散分数。

    p = exp(-t(I-P))·s = e^{-t} Σ_{k=0..∞} (t^k/k!) (Pᵀ)^k s。
    直觉: 热传导——种子节点是"热源", 热量沿边随时间 t 扩散; p[i] 是 i 处温度。
    t 越大扩散越远 (相当于看更远的连通性)。
    """
    import numpy as np
    N = node_mask.shape[0]
    A = _to_scipy_graph(edge_weight, edge_index, N)
    deg = np.asarray(A.sum(axis=1)).squeeze()
    deg[deg == 0] = 1.0
    P = sp_diags(1.0 / deg) @ A
    s = node_mask.float().cpu().numpy().astype(np.float64)
    total = s.sum()
    if total == 0:
        return torch.zeros(N, device=edge_weight.device)
    s = s / total
    # Taylor 截断: e^{-t} Σ (t^k/k!) (Pᵀ)^k s
    p = np.zeros(N)
    term = s.copy()
    fact = 1.0
    for k in range(iters):
        p += term * (t ** k) / fact
        term = P.T @ term
        fact *= (k + 1)
    p *= np.exp(-t)
    return torch.tensor(p, device=edge_weight.device, dtype=torch.float32)


def sp_diags(diag):
    import scipy.sparse as sp
    import numpy as np
    return sp.diags(diag)


def sweep_cut(score, edge_weight, edge_index, min_score_frac=0.05):
    """按扩散分数降序做 conductance sweep, 返回保留节点掩码。

    conductance(S) = cut(S) / min(vol(S), vol(V\\S))
      cut(S) = 跨出 S 的边权和 (S 内外的连接)
      vol(S) = S 内所有节点的加权度之和
    直觉: 一个好的"团块"应内部密集、与外部连接少 -> conductance 小。
    sweep: 按分数从高到低逐个加入节点, 取 conductance 最小的前缀作为保留集。
    分数低于 min_score_frac×max(score) 的节点不允许被保留 (防止扩散尾部噪声)。
    """
    N = score.shape[0]
    a, b = edge_index[0], edge_index[1]
    w = edge_weight.squeeze().detach().float()

    # 加权度 (双向边各计一次, 与无向图一致)
    deg = torch.zeros(N, device=w.device)
    deg.index_add_(0, a, w)
    deg.index_add_(0, b, w)
    vol_total = deg.sum().clamp(min=1e-9)

    order = torch.argsort(score, descending=True)
    min_score = score.max().item() * min_score_frac if score.max() > 0 else 0.0

    # 顺序构造前缀集合的 conductance (N ≤ ~150, 直接 O(N·E) 循环可接受)
    best_cut, best_cond, best_k = 1e30, 1.0, 0
    in_S = torch.zeros(N, dtype=torch.bool, device=w.device)
    volS = 0.0
    for k in range(1, N + 1):
        nid = int(order[k - 1].item())
        if score[nid].item() < min_score:
            break  # 低于最低分数线的节点不再加入
        in_S[nid] = True
        volS += float(deg[nid])
        # cut: 新增节点带来的跨边变化 = 连接到 S 外的权重和 (本节点现在在 S 内)
        # 用 in_S 掩码重算 cut 更稳妥 (N 小)
        cut = float(w[(in_S[a] != in_S[b])].sum())
        if volS <= 0:
            continue
        cond = cut / min(volS, float(vol_total) - volS)
        if cond < best_cond:
            best_cond, best_k = cond, k
    # 保留集 = 前 best_k 个最高分节点 (best_k=0 时退化回纯种子)
    keep = torch.zeros(N, dtype=torch.bool, device=w.device)
    keep[order[:best_k]] = True
    return keep


def node_expand_diffuse(node_weight, edge_weight, edge_index, seed_thr=0.5,
                        method="ppr", alpha=0.85, t=3.0, min_score_frac=0.05,
                        edge_thr=0.0, topk=None):
    """方案 F': 扩散分数 + sweep cut 连通性保留 (替代 F 的贪心 top-k 扩展)。

    1. 种子 S = node_weight > seed_thr        (或 rank-based 前 N%)
    2. 扩散: 从 S 出发算 PPR / heat-kernel 分数 p
    3. sweep cut: 按 p 降序取 conductance 最小的前缀 -> 保留节点集 S'
    4. 边集: S' 内部的边 (可选 per-node top-k 收缩 / edge_thr 过滤)

    Args:
        node_weight: [N,1]
        edge_weight: [E,1]
        edge_index:  [2,E]
        seed_thr:    种子阈值
        method:      "ppr" | "heat"
        alpha:       PPR 跳回概率 (越大扩散越远)
        t:           heat kernel 扩散时间
        min_score_frac: sweep cut 最低分数线 (相对最高分的比例)
        edge_thr:    边最低权重
        topk:        若>0, 保留节点集内再做 per-node top-k 收缩
    Returns:
        (node_selbool [N], edge_selbool [E])
    """
    device = node_weight.device
    a, b = edge_index[0], edge_index[1]
    S = node_weight.squeeze(-1) > seed_thr

    if method == "heat":
        p = heat_kernel_diffuse(S, edge_weight, edge_index, t=t)
    else:
        p = ppr_diffuse(S, edge_weight, edge_index, alpha=alpha)

    keep_n = sweep_cut(p, edge_weight, edge_index, min_score_frac=min_score_frac)
    # 种子节点始终保留 (sweep 可能因 min_score 线截断而漏掉高种子)
    keep_n = keep_n | S

    keep_e = keep_n[a] & keep_n[b]
    if topk and topk > 0:
        keep_e = keep_e & _per_node_topk_incident(edge_weight.squeeze(-1), a, b, keep_e, keep_n, topk)
    if edge_thr > 0:
        keep_e = keep_e & (edge_weight.squeeze(-1) > edge_thr)
    return keep_n, keep_e


# ============ 1d. 链中心性过滤器 (rumor centrality, 2026-08-14) ============
# 灵感: 传染病溯源 (Shah & Zaman 的 rumor centrality) —— 真实衰变链有清晰"根"(B 介子),
# 噪声团块无根。对每条重建链算"中心清晰度", 剔除无清晰中心的链 -> 压低 NoneIso。

def _chain_maxspan_tree(nodes, edge_index, edge_weight):
    """链内最大生成树 (按边权重), 并统计链内边数。

    Args:
        nodes: [k] 链内节点 (图空间索引)
        edge_index: [2,E] 剪枝后图的边
        edge_weight: [E] 剪枝后图的边权重
    Returns:
        (adj, in_nodes, n_in_edges, mst_edges)
        adj: dict {u: [邻居...]} 链内节点的最大生成树
        in_nodes: set 链内节点
        n_in_edges: 链内无向边总数 (判别"树状 vs 乱团"的关键)
        mst_edges: 最大生成树使用的边数 (连通时为 k-1)
    """
    in_nodes = set(int(n) for n in nodes)
    # 链内边 (两端都在链内, 去重无向)
    a, b = edge_index[0], edge_index[1]
    w = edge_weight.squeeze()
    chain_e = []
    for i in range(edge_index.shape[1]):
        ai, bi = int(a[i]), int(b[i])
        if ai in in_nodes and bi in in_nodes:
            if ai < bi:  # 无向去重
                chain_e.append((float(w[i]), ai, bi))
    if not chain_e:
        return {}, in_nodes, 0, 0
    # Kruskal 最大生成树 (链小, 直接排序贪心)
    chain_e.sort(reverse=True)  # 按权重降序
    parent = {n: n for n in in_nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    adj = {n: [] for n in in_nodes}
    mst_edges = 0
    for wt, u, v in chain_e:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            adj[u].append(v)
            adj[v].append(u)
            mst_edges += 1
    return adj, in_nodes, len(chain_e), mst_edges


def _rumor_centrality_log(adj, root):
    """以 root 为根时, rumor centrality 的 log (可比大小, 常数项省略)。

    log R(root) ≈ -Σ_u log(τ_root(u)),  τ_root(u) = 以 root 为根时 u 的子树大小。
    (Shah & Zaman: R(v,T) = N!·Π_u 1/τ_v(u))
    """
    import numpy as np
    n = len(adj)
    if n == 0:
        return 0.0
    # DFS 求子树大小
    parent = {root: -1}
    order = [root]
    stack = [root]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in parent:
                parent[w] = u
                stack.append(w)
                order.append(w)
    size = {u: 1 for u in parent}
    for u in reversed(order):
        for w in adj[u]:
            if parent[w] == u:
                size[u] += size[w]
    return -sum(np.log(size[u]) for u in size)  # -Σ log τ


def chain_center_score(nodes, edge_index, edge_weight):
    """单条链的中心清晰度 + 树状程度 (rumor centrality 视角)。

    Returns:
        (center, clarity, tree_frac):
        center    - 链内 rumor centrality 最大的节点 (可能是"根"/B介子; 链太小时为 None)
        clarity   - max logR - 次大 logR (越大中心越唯一, 链越"单根树")
        tree_frac - MST边数 / 链内无向边数 (1.0=纯树无环, <1=有环/乱团)
    """
    adj, in_nodes, n_in_edges, mst_edges = _chain_maxspan_tree(nodes, edge_index, edge_weight)
    k = len(in_nodes)
    if k < 2 or n_in_edges == 0:
        return None, 0.0, 0.0
    scores = {}
    for root in in_nodes:
        scores[root] = _rumor_centrality_log(adj, root)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_v = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else best_v
    clarity = best_v - second
    tree_frac = mst_edges / n_in_edges
    return best, float(clarity), float(tree_frac)


def filter_chains_by_center(rc_dict, edge_index, edge_weight, clarity_thr, tree_thr=None):
    """按链中心清晰度 + 树状程度过滤 rc_dict (剔除无清晰根 / 有环乱团链)。

    Args:
        rc_dict: reconstruct_decay 输出 {chain_key: {'node_keys': [...]}}
        edge_index / edge_weight: 剪枝后图的边 (与 graph 对齐)
        clarity_thr: 中心清晰度阈值 (max-次大 logR)
        tree_thr: 树状程度阈值 (MST边/链内边, 默认 0.7)
    Returns:
        (filtered_rc_dict, info): 过滤后的链字典 + {chain_key: (center, clarity, tree_frac)}
    """
    if clarity_thr is None or clarity_thr <= 0:
        return rc_dict, {}
    tree_thr = 0.7 if tree_thr is None else tree_thr
    kept, info = {}, {}
    for ck, cluster in rc_dict.items():
        center, clarity, tree_frac = chain_center_score(cluster["node_keys"], edge_index, edge_weight)
        info[ck] = (center, clarity, tree_frac)
        if clarity >= clarity_thr and tree_frac >= tree_thr:
            kept[ck] = cluster
    return kept, info


def truth_chain_roots(y, edge_index, node_batch, device):
    """从 truth 边 (y>0) 构建衰变链, 返回每个节点是否为链根 (rumor centrality argmax)。

    源检测头的标签生成: 每条 truth 链 = 由 truth 非背景边连成的连通分量;
    链的根 = 该链内 rumor centrality 最大的节点 (B 介子候选)。
    在 truth 图上算 (无模型噪声), 供训练时监督 GNN 学习"根-叶结构"。

    Args:
        y: [E] truth 边标签 (0=背景, >0=关联)
        edge_index: [2,E] 有向边 (含双向)
        node_batch: [N] 节点所属事件 (batch 索引)
        device: 计算设备
    Returns:
        [N] float 标签 (1=该节点是某 truth 链的根, 否则 0)
    """
    import numpy as np
    N = node_batch.shape[0]
    labels = torch.zeros(N, dtype=torch.float32, device=device)
    a, b = edge_index[0], edge_index[1]
    y_bin = (y > 0).squeeze(-1) if y.dim() == 2 else (y > 0)
    n_evts = int(node_batch.max().item()) + 1

    for g in range(n_evts):
        tm = node_batch == g                      # 事件 g 的节点
        em = (node_batch[a] == g) & (node_batch[b] == g) & y_bin  # 事件 g 的 truth 边
        if not em.any():
            continue
        ea, eb = a[em], b[em]
        # 事件内局部索引
        gidx = tm.nonzero().flatten()
        local = torch.full((N,), -1, dtype=torch.long, device=device)
        local[gidx] = torch.arange(gidx.numel(), device=device)
        la, lb = local[ea], local[eb]
        n_local = gidx.numel()

        # 连通分量 (并查集)
        parent = list(range(n_local))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(la.numel()):
            ra, rb = find(int(la[i])), find(int(lb[i]))
            if ra != rb:
                parent[ra] = rb
        comp = {}
        for i in range(n_local):
            comp.setdefault(find(i), []).append(i)

        for nodes_local in comp.values():
            if len(nodes_local) < 2:
                continue
            nset = set(nodes_local)
            # 链内无向边 (去重 a<b)
            chain_e = []
            for i in range(la.numel()):
                u, v = int(la[i]), int(lb[i])
                if u in nset and v in nset and u < v:
                    chain_e.append((u, v))
            if not chain_e:
                continue
            # 邻接表 (truth 链近似树, 直接用边)
            adj = {u: [] for u in nset}
            for u, v in chain_e:
                adj[u].append(v)
                adj[v].append(u)
            # rumor centrality: 以每个节点为根的 logR = -Σ log(子树大小)
            best_root, best_logr = None, -1e18
            for root in nset:
                par = {root: -1}
                order = [root]
                stack = [root]
                while stack:
                    u = stack.pop()
                    for w in adj[u]:
                        if w not in par:
                            par[w] = u
                            stack.append(w)
                            order.append(w)
                size = {u: 1 for u in par}
                for u in reversed(order):
                    for w in adj[u]:
                        if par[w] == u:
                            size[u] += size[w]
                logr = -sum(np.log(size[u]) for u in size)
                if logr > best_logr:
                    best_logr, best_root = logr, root
            if best_root is not None:
                labels[gidx[best_root]] = 1.0
    return labels


def truth_chain_structure(y, edge_index, node_batch, device):
    """节点级结构监督标签: 链内掩码 + 到链根的深度 + 归一化 Rumor Centrality。

    用户提议 (2026-08-26): source_head 只预测"根"(1 bit) 信息量低, 升级为连续
    结构监督, 让主干学"节点在衰变树中的位置/层级", 与 LCA 边分类 (母子/同母/祖孙)
    形成全局一致性约束 (class1 边 depth 差 1, class2 差 0, class3 差 >=2)。

    链根 = 链内 rumor centrality 最大节点 (与 truth_chain_roots 一致; 对 truth 链的
    可见径迹集, 质心是"最像树中心"的近似根, 因为 B 介子通常不重建为径迹);
    depth = 到链根的 BFS 拓扑距离 (root=0), 除以链内最大深度归一化到 [0,1];
    rc    = 各节点 logR 在链内 min-max 归一化到 [0,1] (质心=1, 叶子=0)。

    Args:
        y: [E] truth 边标签 (0=背景, >0=关联; 兼容 [E,4] one-hot)
        edge_index: [2,E] 有向边 (含双向)
        node_batch: [N] 节点所属事件 (batch 索引)
        device: 计算设备
    Returns:
        dict: in_chain [N] bool  (链内节点), root [N] 0/1 (链根),
              depth [N] float    (depth/max_depth, 链外 0), rc [N] float (链外 0)
    """
    import numpy as np
    N = node_batch.shape[0]
    in_chain = torch.zeros(N, dtype=torch.bool, device=device)
    root = torch.zeros(N, dtype=torch.float32, device=device)
    depth = torch.zeros(N, dtype=torch.float32, device=device)
    rc = torch.zeros(N, dtype=torch.float32, device=device)
    a, b = edge_index[0], edge_index[1]
    y_bin = (y > 0).squeeze(-1) if y.dim() == 2 else (y > 0)
    n_evts = int(node_batch.max().item()) + 1

    for g in range(n_evts):
        tm = node_batch == g
        em = (node_batch[a] == g) & (node_batch[b] == g) & y_bin
        if not em.any():
            continue
        ea, eb = a[em], b[em]
        gidx = tm.nonzero().flatten()
        local = torch.full((N,), -1, dtype=torch.long, device=device)
        local[gidx] = torch.arange(gidx.numel(), device=device)
        la, lb = local[ea], local[eb]
        n_local = gidx.numel()

        # 连通分量 (并查集)
        parent = list(range(n_local))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(la.numel()):
            ra, rb = find(int(la[i])), find(int(lb[i]))
            if ra != rb:
                parent[ra] = rb
        comp = {}
        for i in range(n_local):
            comp.setdefault(find(i), []).append(i)

        for nodes_local in comp.values():
            if len(nodes_local) < 2:
                continue
            nset = set(nodes_local)
            chain_e = []
            for i in range(la.numel()):
                u, v = int(la[i]), int(lb[i])
                if u in nset and v in nset and u < v:
                    chain_e.append((u, v))
            if not chain_e:
                continue
            adj = {u: [] for u in nset}
            for u, v in chain_e:
                adj[u].append(v)
                adj[v].append(u)

            # rumor centrality: 以每个节点为根的 logR = -Σ log(子树大小)
            logr = {}
            best_root, best_logr = None, -1e18
            for root_cand in nset:
                par = {root_cand: -1}
                order = [root_cand]
                stack = [root_cand]
                while stack:
                    u = stack.pop()
                    for w in adj[u]:
                        if w not in par:
                            par[w] = u
                            stack.append(w)
                            order.append(w)
                size = {u: 1 for u in par}
                for u in reversed(order):
                    for w in adj[u]:
                        if par[w] == u:
                            size[u] += size[w]
                lr = -sum(np.log(size[u]) for u in size)
                logr[root_cand] = lr
                if lr > best_logr:
                    best_logr, best_root = lr, root_cand
            if best_root is None:
                continue

            # 链根 -> BFS 深度 (无向, root=0)
            depth_l = {best_root: 0}
            order_l = [best_root]
            stack = [best_root]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w not in depth_l:
                        depth_l[w] = depth_l[u] + 1
                        stack.append(w)
                        order_l.append(w)
            max_d = max(depth_l.values()) if depth_l else 0

            # rc 归一化 (链内 min-max)
            lrs = [logr[u] for u in nset]
            lmin, lmax = min(lrs), max(lrs)
            rspan = (lmax - lmin) or 1.0

            for u in nset:
                gi = int(gidx[u])
                in_chain[gi] = True
                depth[gi] = (depth_l[u] / max_d) if max_d > 0 else 0.0
                rc[gi] = (logr[u] - lmin) / rspan
            root[int(gidx[best_root])] = 1.0

    return {"in_chain": in_chain, "root": root, "depth": depth, "rc": rc}


# ============ 1e. 链级 LCA 拓扑/置信度判据 (2026-08-14) ============
# "最物理"判据: 重建链由"模型判成 LCA>0"的边构成 (reconstruct_decay 已过滤背景边),
# 真链的链内边应是模型**高置信**的非背景边 (LCA softmax 概率高);
# 噪声链的边是模型勉强/误判成的 (softmax 概率低)。判据 = 链内边平均 LCA 置信度。

def chain_lca_score(nodes, edge_index, lca_probs):
    """单条链的 LCA 物理置信度。

    Args:
        nodes: [k] 链内节点
        edge_index: [2,E] 剪枝后图的边
        lca_probs: [E,4] 链内边所属类别的 softmax 概率 (logits 过 softmax)
    Returns:
        (conf, class2_frac):
        conf       - 链内边"被判类别"的平均 softmax 概率 (高=链由高置信物理关系构成)
        class2_frac- 链内类2/3 边占比 (同B/两B 关系; 完整 B 链更高, 辅助指标)
    """
    if lca_probs is None or lca_probs.shape[0] == 0:
        return 0.0, 0.0
    in_nodes = set(int(n) for n in nodes)
    a, b = edge_index[0], edge_index[1]
    # 链内边 = 两端都在链内 (向量化)
    in_e = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
    if in_nodes:
        nid = torch.tensor(sorted(in_nodes), device=edge_index.device)
        in_e = torch.isin(a, nid) & torch.isin(b, nid)
    if not in_e.any():
        return 0.0, 0.0
    probs = lca_probs[in_e]                       # [n_e, 4]
    conf = probs.max(dim=-1).values.mean().item()  # 被判类别的平均概率
    cls = probs.argmax(dim=-1)
    class2_frac = float(((cls == 2) | (cls == 3)).float().mean().item())
    return conf, class2_frac


def filter_chains_by_lca(rc_dict, edge_index, lca_logits, conf_thr, class2_thr=None):
    """按链级 LCA 物理置信度过滤 rc_dict。

    Args:
        rc_dict: reconstruct_decay 输出
        edge_index: [2,E] 剪枝后图的边
        lca_logits: [E,4] 剪枝后图边的 LCA logits (graph[tt].lca)
        conf_thr:  链内边平均 LCA 置信度阈值 (低于则剔除)
        class2_thr:链内类2/3 边占比下限 (None=不启用; 用于强化"完整B链"判据)
    Returns:
        (filtered_rc_dict, info): 过滤后链 + {chain_key: (conf, class2_frac)}
    """
    if conf_thr is None or conf_thr <= 0:
        return rc_dict, {}
    import torch.nn.functional as F
    lca_probs = F.softmax(lca_logits, dim=-1)
    kept, info = {}, {}
    for ck, cluster in rc_dict.items():
        conf, class2_frac = chain_lca_score(cluster["node_keys"], edge_index, lca_probs)
        info[ck] = (conf, class2_frac)
        if conf >= conf_thr and (class2_thr is None or class2_frac >= class2_thr):
            kept[ck] = cluster
    return kept, info


# ============ 2. 候选衰变链选择 MLP ============

class CandidateScorer(nn.Module):
    """对一条完整重建的衰变链打分, 输出该链的总 likelihood。

    输入为单条链内变长的节点/边特征集合:
      - 节点特征 [N, d_node]: 链内每条径迹 = [node_weight] + 节点物理特征 (tracks.x)
      - 边特征   [M, d_edge]: 链内每条边 = [edge_weight] + [LCA 概率(4)] + 边物理特征 (edges)
    链内节点/边数量 N、M 不定 -> 用 set pooling (mean + max + sum) 聚合成定长向量,
    再拼接数量信息, 经 MLP 输出标量 score (该链越合理分数越高)。
    """
    def __init__(self, node_dim, edge_dim, hidden=64):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # 聚合特征: node(mean,max,sum) + edge(mean,max,sum) + logN + logM
        self.head = nn.Sequential(
            nn.Linear(hidden * 6 + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_feats, edge_feats):
        """
        Args:
            node_feats: [N, d_node] 单条链内所有节点的特征
            edge_feats: [M, d_edge] 单条链内所有边的特征
        Returns:
            score: 标量 tensor
        """
        nh = self.node_enc(node_feats)          # [N, h]
        eh = self.edge_enc(edge_feats)          # [M, h]
        n_mean, n_max = nh.mean(0), nh.max(0).values
        n_sum = nh.sum(0)
        e_mean, e_max = eh.mean(0), eh.max(0).values
        e_sum = eh.sum(0)
        N = nh.shape[0]
        M = eh.shape[0]
        z = torch.cat([n_mean, n_max, n_sum, e_mean, e_max, e_sum,
                       torch.log1p(torch.tensor([float(N)], device=nh.device)),
                       torch.log1p(torch.tensor([float(M)], device=nh.device))])
        return self.head(z).squeeze(-1)


def chain_features(graph, node_weights, edge_weights, lca_probs, edge_mask,
                   chain_node_keys, particle_keys):
    """从单条候选衰变链提取链内节点/边特征 (供 CandidateScorer 打分)。

    一条链由 particle_keys 中若干键组成 (reconstruct_decay 的 cluster node_keys);
    链内边 = 事件边中两端都在链内节点集合里的边。

    注意: 所有输入必须在同一空间对齐 (原始全量 或 节点剪枝后均可, 但需一致):
      graph['tracks'].x      [N, d_phys]   节点物理特征 (与 particle_keys 行对齐)
      node_weights           [N]           节点权重 (与 particle_keys 行对齐)
      graph tt 边            [E]           事件边 (与 edge_weights/lca_probs 对齐)
      edge_weights/lca_probs [E] / [E, 4]  边权重 / LCA 概率
      edge_mask              [E] bool      候选边保留掩码 (top-k 或阈值; None=全保留)
      particle_keys          [N]           节点键 (与 tracks.x 行对齐)

    Returns:
        (node_feats [Nc, d_node], edge_feats [Mc, d_edge]) 或 (None, None) 若无链内边
    """
    dev = graph['tracks'].x.device
    key_pos = {int(k): i for i, k in enumerate(particle_keys)}
    chain_idx = [key_pos[int(k)] for k in chain_node_keys if int(k) in key_pos]
    if not chain_idx:
        return None, None

    ci = torch.tensor(chain_idx, device=dev)
    node_w = node_weights[ci]                       # [Nc]
    node_phys = graph['tracks'].x[ci]               # [Nc, d_phys]
    node_feats = torch.cat([node_w.unsqueeze(-1), node_phys], dim=-1)

    # 链内边: 事件边中两端都在链内, 且被候选掩码保留
    tt_ei = graph[('tracks', 'to', 'tracks')].edge_index   # [2, E]
    in_chain = torch.zeros(len(particle_keys), dtype=torch.bool, device=dev)
    in_chain[ci] = True
    e_in = in_chain[tt_ei[0]] & in_chain[tt_ei[1]]
    if edge_mask is not None:
        e_in = e_in & edge_mask
    if not e_in.any():
        return node_feats, None
    edge_w = edge_weights[e_in]
    lca = lca_probs[e_in]
    # 边物理特征: 优先用挂载的原始物理特征 (model 会覆盖 edges), 否则用 edges
    tt_store = graph[('tracks', 'to', 'tracks')]
    edge_phys = tt_store.phys_edges[e_in] if hasattr(tt_store, "phys_edges") else tt_store.edges[e_in]
    edge_feats = torch.cat([edge_w.unsqueeze(-1), lca, edge_phys], dim=-1)
    return node_feats, edge_feats


def score_chains(scorer, graph, node_weights, edge_weights, lca_probs,
                 edge_mask, cluster_dict, particle_keys):
    """对事件内所有候选衰变链打分, 返回 {chain_key: score}。

    Args:
        cluster_dict: reconstruct_decay 的输出 {chain_key: {'node_keys': [...], ...}}
    Returns:
        {chain_key: float score}
    """
    dev = next(scorer.parameters()).device
    scores = {}
    for ck, cluster in cluster_dict.items():
        nf, ef = chain_features(graph, node_weights, edge_weights, lca_probs,
                                edge_mask, cluster["node_keys"], particle_keys)
        if nf is None or ef is None or ef.shape[0] == 0:
            scores[ck] = -float("inf")
            continue
        with torch.no_grad():
            s = scorer(nf.to(dev), ef.to(dev))
        scores[ck] = float(s.item())
    return scores


def load_scorer(ckpt_path, node_dim, edge_dim, device="cpu"):
    """从 checkpoint 加载 CandidateScorer (兼容旧 ckpt: 无维度元数据时用传入维度)。"""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = state.get("scorer_meta", {})
    nd = meta.get("node_dim", node_dim)
    ed = meta.get("edge_dim", edge_dim)
    scorer = CandidateScorer(nd, ed)
    scorer.load_state_dict(state["scorer_state"])
    scorer.to(device).eval()
    return scorer


# ============ 3. 联合训练辅助: 从 batch 构造链级样本 (正: truth链, 负: 随机链) ============

def build_chain_samples(edge_index, y_bin, node_sig, n_nodes,
                        node_feats_all, edge_feats_all, k_neg=None, max_pos=32):
    """对一个事件, 构造链级训练样本 (正/负), 全部返回定长 batch。

    truth 正边 (y_bin>0 且两端 sig) 的连通分量 = 真链 (正样本);
    随机组合节点 = 假链 (负样本)。

    Args:
        edge_index:     [2, E] 事件内边索引 (含双向)
        y_bin:          [E] 边二元 truth (>0 = 有关联)
        node_sig:       [N] bool, 信号节点掩码
        n_nodes:        int, 节点数
        node_feats_all: [N, d_node] 节点特征 (权重+物理)
        edge_feats_all: [E, d_edge] 边特征 (权重+LCA+物理)
        k_neg:          负样本链节点数 (默认 n_sig//3+2)
        max_pos:        单事件最多取的正样本数 (防止过多链)
    Returns:
        (pos_node, pos_edge, pos_y, neg_node, neg_edge, neg_y) 或 None (无样本)
        pos_node [P, d_node] pos_edge [P, max_edges, d_edge]... 为简化用 list 返回
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    dev = node_feats_all.device
    ei = edge_index.cpu()
    y = y_bin.cpu().bool()
    sig = node_sig.cpu()

    # truth 正边 (去重无向)
    undir = (ei[0] < ei[1]) & y & sig[ei[0]] & sig[ei[1]]
    pos_rows = ei[:, undir]
    if pos_rows.shape[1] == 0:
        return None
    # 连通分量 -> 真链
    adj = sp.coo_matrix((np.ones(pos_rows.shape[1]),
                         (pos_rows[0].numpy(), pos_rows[1].numpy())),
                        shape=(n_nodes, n_nodes)).tocsr()
    adj = (adj + adj.T) > 0
    n_comp, labels = connected_components(adj, directed=False)
    comps = {}
    for nid, c in enumerate(labels):
        if sig[nid] and c not in comps:
            comps[c] = []
        if sig[nid]:
            comps[c].append(nid)
    chains = [v for v in comps.values() if len(v) >= 2][:max_pos]
    if not chains:
        return None

    # 正样本特征: 链内节点 + 链内边
    pos_n, pos_e, pos_y = [], [], []
    sig_idx = torch.nonzero(sig).flatten().cpu().numpy()
    n_sig = len(sig_idx)
    rng = np.random.default_rng(0)
    for c in chains:
        cset = set(c)
        nf = node_feats_all[torch.tensor(c, device=dev)]            # [Nc, d_node]
        e_in = np.isin(ei[0].numpy(), list(cset)) & np.isin(ei[1].numpy(), list(cset))
        if e_in.sum() == 0:
            continue
        ef = edge_feats_all[torch.from_numpy(e_in).to(dev)]         # [Mc, d_edge]
        pos_n.append(nf); pos_e.append(ef); pos_y.append(1.0)
    # 负样本: 随机抽 k 个 sig 节点 (避开与任一真链节点集合一致)
    k = k_neg or max(2, n_sig // 3)
    neg_n, neg_e, neg_y = [], [], []
    n_neg = len(pos_n)
    for _ in range(n_neg):
        picks = rng.choice(sig_idx, size=min(k, n_sig), replace=False)
        if any(set(picks.tolist()) == set(c) for c in chains):
            continue
        nf = node_feats_all[torch.tensor(picks, device=dev)]
        pset = set(picks.tolist())
        e_in = np.isin(ei[0].numpy(), list(pset)) & np.isin(ei[1].numpy(), list(pset))
        ef = edge_feats_all[torch.from_numpy(e_in).to(dev)] if e_in.sum() > 0 else None
        if ef is None:
            continue
        neg_n.append(nf); neg_e.append(ef); neg_y.append(0.0)

    if not pos_n:
        return None
    return (pos_n, pos_e, pos_y, neg_n, neg_e, neg_y)
