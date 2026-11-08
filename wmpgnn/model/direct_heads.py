"""输出侧直接化 (2026-11-08): A 顶点优先 / B 拆分二分 / C 锚点分配。

为什么换输出
------------
现输出是 tt 边的 LCAG 四分类 (0=背景, 1/2/3 = 到最低共同祖先的层距), 解码只取
argmax>0 再做连通分量。与物理目标差两层:
  * 1/2/3 对**连通性**没有区别 (都只表示"同一条链"), 监督却分散到三类, 阈值附近
    校准差; 真正需要的是"同不同链"这一个二分。
  * 连通分量把"远房亲戚"和"直接母子"一视同仁, 容易整片粘连, 丢掉层级/树结构。

三个替代输出 (全部复用现有真值, 不新增标注):
  B  SplitHead  : same-chain 二分 (对称, y>0) + gap 有序三分类 (y=1/2/3)
  C  AnchorHead : 锚点 = 链根 / B 候选 (truth_chain_roots), 每径迹排他分配到锚点
  A  SlotArcHead: 顶点层 (伪顶点 = 生产点聚簇) + 顶点弧 + 森林合并

A 的"顶点"从哪来
---------------
样本 .pt 里**没有逐径迹真值顶点号** (只有 tracks.x/pid/ft、tt 边 y、tp 边 PV 真值)。
因此 A 用物理定义构造伪顶点: 同事件内 (xProd,yProd,zProd) 距离 <= tol_mm 的径迹归为
同一顶点 (单链聚簇); 顶点之间的弧用"两顶点之间是否存在真值同链边"监督 (可导出)。
它不是真值顶点, 但正是解码器物理上需要的对象层。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(int(d_in), int(d_hidden)), nn.GELU(),
                         nn.Linear(int(d_hidden), int(d_out)))


# ------------------------------------------------------------------ 公共工具
def _union_labels(n, snd, rcv, min_deg=1):
    """并查集: (snd, rcv) 连边后的连通分量标签; 度 < min_deg 的节点记 -1。"""
    n = int(n)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = np.zeros(n, dtype=np.int64)
    for a, b in zip(snd, rcv):
        a, b = int(a), int(b)
        deg[a] += 1
        deg[b] += 1
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    remap, lab = {}, np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if deg[i] < min_deg:
            continue
        r = find(i)
        lab[i] = remap.setdefault(r, len(remap))
    return lab


def pseudo_vertex_groups(pos, tol_mm=5.0):
    """单链聚簇: 生产点 (xProd,yProd,zProd, mm) 距离 <= tol 的径迹归为同一伪顶点。

    返回 (groups [G,N] bool, gid [N] long, -1 = 单点/背景未成组)。
    只保留 size>=2 的组 (单点不构成"顶点对象")。
    """
    n = int(pos.shape[0])
    gid = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return torch.zeros((0, n), dtype=torch.bool), gid
    d = torch.cdist(pos.detach().float(), pos.detach().float())
    adj = (d <= float(tol_mm)).cpu().numpy()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    buckets = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    groups, k = [], 0
    for _, idx in buckets.items():
        if len(idx) < 2:
            continue
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.as_tensor(idx, dtype=torch.long)] = True
        groups.append(m)
        gid[idx] = k
        k += 1
    if not groups:
        return torch.zeros((0, n), dtype=torch.bool), gid
    return torch.stack(groups, 0), gid


# ------------------------------------------------------------------ B: 拆分
class SplitHead(nn.Module):
    """把 4 类 LCAG 换成两个更直接的输出。

    same : 是否同一条真值链 (对称) —— 解码只需要这个;
    gap  : 同链时到最低共同祖先的层距 (1/2/3, 有序) —— 给组装器恢复层级用。

    对称性: 用 [h_i+h_j, |h_i-h_j|] 作节点对特征, 交换端点严格不变; 边特征按
    edge_index 顺序拼接 (数据里每对只出现一次, i<j)。
    """

    def __init__(self, d_node, d_edge=16, hidden=64):
        super().__init__()
        self.d_edge = int(d_edge)
        d_in = 2 * int(d_node) + int(d_edge)
        self.same = _mlp(d_in, hidden, 1)
        self.gap = _mlp(d_in, hidden, 3)

    def forward(self, h, edge_index, h_edge):
        i, j = edge_index[0].long(), edge_index[1].long()
        parts = [h[i] + h[j], (h[i] - h[j]).abs()]
        if h_edge is None or int(h_edge.numel()) == 0:
            parts.append(torch.zeros(h[i].shape[0], self.d_edge, device=h.device, dtype=h.dtype))
        else:
            parts.append(h_edge.to(h.dtype))
        z = torch.cat(parts, -1)
        return self.same(z).squeeze(-1), self.gap(z)


def split_loss(same_logit, gap_logit, y, pos_weight=10.0, gap_weight=1.0):
    """(BCE_same, CE_gap). y: [E] LCAG 0..3。pos_weight 抑制 1:700 的极端不平衡。"""
    yb = (y > 0).to(same_logit.dtype)
    pw = torch.as_tensor(float(pos_weight), device=same_logit.device, dtype=same_logit.dtype)
    bce = F.binary_cross_entropy_with_logits(same_logit, yb, pos_weight=pw)
    m = y > 0
    if bool(m.any()):
        ce = F.cross_entropy(gap_logit[m], (y[m].long() - 1).clamp(0, 2))
    else:
        ce = gap_logit.sum() * 0.0
    return bce, float(gap_weight) * ce


def decode_split(same_prob, edge_index, n_nodes, thr=0.5, gap_prob=None,
                 gap_min=None):
    """same>thr 的边做连通分量 -> 逐节点链标签 (-1 = 无链)。

    gap_min: 若给出, 只保留预测层距 >= gap_min 的边 (默认 None = 全保留)。
    """
    p = same_prob.detach().float().cpu().numpy()
    ei = edge_index.detach().cpu().numpy()
    keep = p >= float(thr)
    if gap_prob is not None and gap_min is not None:
        gp = gap_prob.detach().float().cpu().numpy()
        keep = keep & ((gp.argmax(-1) + 1) >= int(gap_min))
    return _union_labels(n_nodes, ei[0][keep], ei[1][keep])


# ------------------------------------------------------------------ C: 锚点
class AnchorHead(nn.Module):
    """锚点 = 链根 (B 候选)。输出 (根 logit, 逐径迹对逐锚点的分配 logit)。

    训练时锚点用**真值根** (teacher forcing), 推理时用预测得分 top-k 的节点。
    分配是排他的: 逐径迹 softmax 到 (k 个锚点 + 背景)。
    """

    def __init__(self, d_node, hidden=64):
        super().__init__()
        d = int(d_node)
        self.root = _mlp(d, hidden, 1)
        self.pair = _mlp(4 * d, hidden, 1)

    def forward(self, h, anchor_idx=None):
        root_logit = self.root(h).squeeze(-1)                       # [N]
        if anchor_idx is None or int(anchor_idx.numel()) == 0:
            return root_logit, None
        A = h[anchor_idx.long()]                                    # [K,d]
        N, K, d = int(h.shape[0]), int(A.shape[0]), int(h.shape[1])
        hi = h.unsqueeze(1).expand(N, K, d)
        hk = A.unsqueeze(0).expand(N, K, d)
        z = torch.cat([hi, hk, hi * hk, (hi - hk).abs()], -1)
        return root_logit, self.pair(z).squeeze(-1)                 # [N,K]


def anchor_loss(root_logit, assign_logit, root_lab, chain_lab, anchor_idx,
                root_weight=1.0, assign_weight=1.0):
    """root BCE + 分配 CE (锚点索引 / 背景)。root_lab: [N] 0/1; chain_lab: [N] -1/0.."""
    l_root = F.binary_cross_entropy_with_logits(root_logit, root_lab.to(root_logit.dtype))
    if assign_logit is None:
        return float(root_weight) * l_root, l_root.detach()
    N, K = int(assign_logit.shape[0]), int(assign_logit.shape[1])
    dev = assign_logit.device
    # 锚点 k 对应的真值链 id -> 该链内径迹的目标槽位
    chain2slot = {}
    for k in range(K):
        c = int(chain_lab[int(anchor_idx[k])])
        if c >= 0 and c not in chain2slot:
            chain2slot[c] = k
    target = torch.full((N,), K, device=dev, dtype=torch.long)      # 默认背景
    for t in range(N):
        c = int(chain_lab[t])
        if c in chain2slot:
            target[t] = chain2slot[c]
    full = torch.cat([assign_logit, torch.zeros(N, 1, device=dev, dtype=assign_logit.dtype)], -1)
    l_asg = F.cross_entropy(full, target)
    return float(root_weight) * l_root + float(assign_weight) * l_asg, l_asg.detach()


def decode_anchor(root_logit, h, head, n_nodes, k=6, root_thr=0.5):
    """推理: top-k 根 -> 分配 argmax -> 逐节点链标签 (-1=背景)。"""
    with torch.no_grad():
        prob = torch.sigmoid(root_logit)
        cand = torch.nonzero(prob >= float(root_thr)).flatten()
        if cand.numel() == 0:
            cand = torch.topk(prob, k=min(int(k), n_nodes)).indices
        if cand.numel() > int(k):
            cand = cand[torch.topk(prob[cand], k=int(k)).indices]
        _, assign_logit = head(h, cand)
        if assign_logit is None:
            return np.full(n_nodes, -1, dtype=np.int64)
        full = torch.cat([assign_logit,
                          torch.zeros(assign_logit.shape[0], 1,
                                      device=assign_logit.device, dtype=assign_logit.dtype)], -1)
        cls = full.argmax(-1).cpu().numpy()
        lab = np.full(n_nodes, -1, dtype=np.int64)
        alive = (prob[cand] >= float(root_thr)).cpu().numpy()
        for t in range(n_nodes):
            c = int(cls[t])
            if c < cand.numel() and bool(alive[c]):
                lab[t] = c
        return lab


# ------------------------------------------------------------------ A: 顶点弧
class SlotArcHead(nn.Module):
    """顶点/槽位之间的弧 logit (对称), 输入是掩码池化后的节点表征。"""

    def __init__(self, d_node, hidden=64):
        super().__init__()
        d = int(d_node)
        self.mlp = _mlp(4 * d, hidden, 1)

    def forward(self, a):
        K, d = int(a.shape[0]), int(a.shape[1])
        gi = a.unsqueeze(1).expand(K, K, d)
        gj = a.unsqueeze(0).expand(K, K, d)
        z = torch.cat([gi, gj, gi * gj, (gi - gj).abs()], -1)
        arc = self.mlp(z).squeeze(-1)
        return 0.5 * (arc + arc.transpose(0, 1))                    # 对称化


def pool_by_mask(h, masks):
    """masks [G,N] (bool/float) -> 池化表征 [G,d]; 空组为 0 向量。"""
    m = masks.to(device=h.device, dtype=h.dtype)
    cnt = m.sum(-1, keepdim=True).clamp_min(1e-6)
    return (m @ h) / cnt


def arc_loss(head, h, masks, chain_lab, min_group=2, weight=1.0):
    """顶点弧 BCE: 两顶点是否属于同一条真值链 (只统计两侧都非空的对)。"""
    if masks is None or int(masks.shape[0]) < 2:
        return None
    masks = masks.to(h.device)
    keep = masks.sum(-1) >= int(min_group)
    if int(keep.sum()) < 2:
        return None
    masks = masks[keep]
    a = pool_by_mask(h, masks)                                      # [G,d]
    arc = head(a)                                                    # [G,G]
    cl = chain_lab.long()
    lab = []
    for i in range(int(masks.shape[0])):
        ci = cl[masks[i]]
        ci = ci[ci >= 0]
        lab.append(int(torch.bincount(ci).argmax()) if ci.numel() > 0 else -1)
    lab = torch.as_tensor(lab, device=h.device)
    same = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
    G = int(masks.shape[0])
    iu = torch.triu_indices(G, G, offset=1, device=h.device)
    if iu.shape[1] == 0:
        return None
    tgt = same[iu[0], iu[1]].to(arc.dtype)
    logit = arc[iu[0], iu[1]]
    return float(weight) * F.binary_cross_entropy_with_logits(logit, tgt)


def decode_vertex(gid, h, head, n_nodes, arc_thr=0.5):
    """顶点组 -> 弧 >= thr 的顶点做并查集合并 -> 逐节点链标签 (-1=背景)。"""
    gid = np.asarray(gid)
    uniq = [g for g in np.unique(gid) if g >= 0]
    if not uniq:
        return np.full(n_nodes, -1, dtype=np.int64)
    masks = torch.zeros((len(uniq), n_nodes), dtype=torch.bool)
    for k, g in enumerate(uniq):
        masks[k, torch.as_tensor(np.nonzero(gid == g)[0], dtype=torch.long)] = True
    with torch.no_grad():
        a = pool_by_mask(h, masks)
        arc = head(a)
    snd, rcv = [], []
    G = len(uniq)
    for i in range(G):
        for j in range(i + 1, G):
            if float(arc[i, j]) >= float(arc_thr):
                snd.append(i)
                rcv.append(j)
    # 顶点级连通分量 -> 节点级链标签
    vlab = _union_labels(G, snd, rcv, min_deg=0)
    lab = np.full(n_nodes, -1, dtype=np.int64)
    for k, g in enumerate(uniq):
        lab[gid == g] = vlab[k]
    return lab
