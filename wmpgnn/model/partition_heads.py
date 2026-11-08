"""输出侧新机制的实现 (M1 Object Condensation / M2 Slot+Mask / M3 分配式解码)。

设计约束
--------
* **只吃节点表示**: 剪枝侧(边打分/节点分数/LCA 头)完全不动, 本模块只在其后接新头。
* 每个机制都配一个**解码器**, 且解码器满足: 一条径迹最多属于一个对象; 可以判为背景(dustbin)。
* 所有损失都有对应的单元测试 (`test_partition_heads.py`), 用手算例子校验数值与梯度。

术语
----
N = 事件内径迹数; K = 槽位数(对象数上界); y_i ∈ {-1, 0..M-1} 为真值对象号(-1=背景)。
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- M1: OC
class OCHead(nn.Module):
    """β(objectness) + 潜坐标。损失: L_beta + L_att + s_rep * L_rep (+ 可选 L_pot)。"""

    def __init__(self, in_dim, emb_dim=8, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.beta = nn.Linear(hidden, 1)
        self.emb = nn.Linear(hidden, emb_dim)

    def forward(self, x):
        z = self.trunk(x)
        return self.beta(z).squeeze(-1), self.emb(z)          # (N,), (N,D)


def oc_loss(beta_logit, emb, y, s_rep=0.06, margin=1.0, bkg_weight=0.05, eps=1e-6):
    """OC 三项损失。

    L_beta : 非对称 BCE (背景权重 bkg_weight, 避免背景淹没信号)
    L_att  : 同一真值对象的节点在潜空间靠近 (L2)
    L_rep  : 不同对象 / 背景的节点对, hinge 推开 (ReLU(margin - d^2))
    返回 (total, dict)。全部为标量, 可反传。
    """
    beta = torch.sigmoid(beta_logit)
    is_sig = (y >= 0).float()
    # --- L_beta
    bce = F.binary_cross_entropy_with_logits(beta_logit, is_sig, reduction="none")
    w = torch.where(is_sig > 0, torch.ones_like(bce), torch.full_like(bce, bkg_weight))
    l_beta = (bce * w).sum() / w.sum().clamp_min(eps)

    # --- 吸引力: 同对象节点对
    d2 = torch.cdist(emb, emb, p=2).pow(2)                    # (N,N) 平方距离
    same = (y[:, None] == y[None, :]) & (y[:, None] >= 0)
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    same = same & ~eye
    diff = (~same) & (~eye)
    if same.any():
        l_att = d2[same].mean()
    else:
        l_att = emb.sum() * 0.0
    # --- 排斥力: 不同对象/背景 的节点对
    # 注意: 归一化用"吸引对数量"而非"排斥对数量"(与 OC 原文一致),
    # 否则大量相距很远的背景对会把信号稀释掉。
    n_att = same.sum().clamp_min(1).float()
    if diff.any():
        l_rep = F.relu(margin - d2[diff]).sum() / n_att
    else:
        l_rep = emb.sum() * 0.0

    total = l_beta + l_att + s_rep * l_rep
    return total, {"L_beta": l_beta.detach(), "L_att": l_att.detach(), "L_rep": l_rep.detach()}


@torch.no_grad()
def decode_oc(beta, emb, beta_thr=0.5, td=1.0, max_rounds=8):
    """OC 后处理: β>thr 且未被更强者"覆盖"的点做凝聚点; 其余分配到最近的凝聚点,
    距离 > td 则进 dustbin(-1)。保证: 每个点最多属于一个对象。"""
    n = len(beta)
    lab = torch.full((n,), -1, dtype=torch.long, device=beta.device)
    order = torch.argsort(-beta)
    cps = []
    for i in order.tolist():
        if beta[i] < beta_thr:
            break
        if cps:
            d = torch.cdist(emb[i:i + 1], emb[cps]).min()
            if d < td:
                continue
        cps.append(i)
    if not cps:
        return lab
    d = torch.cdist(emb, emb[cps])                            # (N, C)
    nearest = d.argmin(dim=1)
    dmin = d.min(dim=1).values
    ok = dmin <= td
    lab[ok] = nearest[ok]
    return lab


# ---------------------------------------------------- M2: Slot + Mask
class SlotMaskHead(nn.Module):
    """K 个槽位, 每槽位对每个节点给 mask logit, 另给槽位 objectness。"""

    def __init__(self, in_dim, n_slots=4, hidden=64):
        super().__init__()
        self.n_slots = n_slots
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.mask = nn.Linear(hidden, n_slots)
        self.obj = nn.Linear(hidden, n_slots)

    def forward(self, x):
        z = self.trunk(x)
        return self.mask(z), self.obj(z)                      # (N,K), (N,K)


class SlotAttentionHead(nn.Module):
    """Slot Attention (Locatello et al. 2020) 版划分头。

    与 SlotMaskHead 同接口: forward(x) -> (mask_logit (N,K), obj_logit (N,K))。
    区别在于槽位不是"每个节点独立出 K 个 logit", 而是 K 个 query 与节点做**竞争式注意力**:
    每轮在槽位维做 softmax (一个节点必须选一个槽), 再按节点归一化后聚合, 用 GRU 更新槽位,
    迭代 iters 轮。最后一轮的注意力权重 A[k,i] 就是归属概率, mask_logit = log A。
    """
    def __init__(self, in_dim, n_slots=8, dim=64, iters=3, hidden=64):
        super().__init__()
        self.n_slots, self.dim, self.iters = n_slots, dim, iters
        self.in_proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.slots = nn.Parameter(torch.randn(n_slots, dim) * 0.5)
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.obj = nn.Linear(hidden, 1)

    def forward(self, x):
        n = x.shape[0]
        v = self.in_proj(x)                                   # (N,D)
        k, val = self.k(v), self.v(v)
        slots = self.slots.unsqueeze(0)                       # (1,K,D)
        scale = self.dim ** -0.5
        for _ in range(self.iters):
            qs = self.q(slots)                                # (1,K,D)
            attn = (qs @ k.T * scale).softmax(dim=1)          # (1,K,N) 槽位维竞争
            attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)
            upd = (attn @ val).reshape(self.n_slots, self.dim)
            slots = self.gru(upd, slots.reshape(self.n_slots, self.dim)).unsqueeze(0)
        qs = self.q(slots)
        A = (qs @ k.T * scale).squeeze(0).softmax(dim=0)      # (K,N) 每节点对槽位归一化
        mask_logit = torch.log(A.clamp_min(1e-6)).T.contiguous()   # (N,K)
        obj = self.obj(self.mlp(slots.squeeze(0)))            # (K,1)
        return mask_logit, obj.T.expand(n, self.n_slots)


def _bruteforce_assign(cost):
    """K 槽位 × M 真值对象的最优匹配: 返回 slot->truth 的排列 (-1 = 该槽位未匹配)。

    [2026-10-26 FIX] 支持 M > K: 原来枚举 permutations(range(K), M) 在 M>K 时为空,
    best_perm=None -> TypeError, 整批 loss 被吞掉, 头等于没训。现在 M>K 走
    scipy 线性分配 (K×M 矩形, 只匹配 min(K,M) 对), 多出来的真值对象不匹配。
    """
    K, M = cost.shape
    c = cost.detach().cpu()
    if M <= K:
        best, best_perm = float("inf"), None
        for perm in itertools.permutations(range(K), M):
            v = sum(float(c[perm[m], m]) for m in range(M))
            if v < best:
                best, best_perm = v, perm
        out = [-1] * K
        for m, k in enumerate(best_perm):
            out[k] = m
        return torch.tensor(out), best
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(c.numpy())
        out = [-1] * K
        for k, m in zip(rows.tolist(), cols.tolist()):
            out[int(k)] = int(m)
        return torch.tensor(out), float(c.numpy()[rows, cols].sum())
    except Exception:
        # 无 scipy 时的确定性贪心: 每次取全局最小代价, 行/列各用一次
        out, used_k, used_m, tot = [-1] * K, set(), set(), 0.0
        while len(used_k) < K and len(used_m) < M:
            bk, bm, bv = -1, -1, float("inf")
            for k in range(K):
                if k in used_k:
                    continue
                for m in range(M):
                    if m in used_m:
                        continue
                    v = float(c[k, m])
                    if v < bv:
                        bk, bm, bv = k, m, v
            if bk < 0:
                break
            out[bk] = bm; used_k.add(bk); used_m.add(bm); tot += bv
        return torch.tensor(out), tot


def slot_mask_loss(mask_logit, obj_logit, y, bkg_weight=0.05, w_obj=0.5):
    """匈牙利匹配后计算 mask BCE + 槽位 objectness BCE。返回 (total, dict, assign)。"""
    n, K = mask_logit.shape
    objs = [int(v) for v in torch.unique(y) if int(v) >= 0]
    M = len(objs)
    if M == 0:
        z = mask_logit.sum() * 0.0
        return z, {"mask": z.detach(), "obj": z.detach()}, None
    # 代价矩阵: 槽位 k 与真值对象 m 的 mask BCE
    cost = torch.zeros(K, M, device=mask_logit.device)
    for m, o in enumerate(objs):
        tgt = (y == o).float()
        w = torch.where(tgt > 0, torch.ones_like(tgt), torch.full_like(tgt, bkg_weight))
        for k in range(K):
            bce = F.binary_cross_entropy_with_logits(mask_logit[:, k], tgt, reduction="none")
            cost[k, m] = (bce * w).sum() / w.sum().clamp_min(1e-6)
    assign, _ = _bruteforce_assign(cost)
    assign = assign.to(mask_logit.device)

    # mask loss: 匹配上的槽位学真值; 未匹配槽位学习"全背景"
    l_mask = mask_logit.sum() * 0.0
    for k in range(K):
        m = int(assign[k])
        tgt = (y == objs[m]).float() if m >= 0 else torch.zeros(n, device=mask_logit.device)
        w = torch.where(tgt > 0, torch.ones_like(tgt), torch.full_like(tgt, bkg_weight))
        bce = F.binary_cross_entropy_with_logits(mask_logit[:, k], tgt, reduction="none")
        l_mask = l_mask + (bce * w).sum() / w.sum().clamp_min(1e-6)
    l_mask = l_mask / K

    tgt_obj = (assign >= 0).float()
    l_obj = F.binary_cross_entropy_with_logits(obj_logit.mean(dim=0), tgt_obj)
    total = l_mask + w_obj * l_obj
    return total, {"mask": l_mask.detach(), "obj": l_obj.detach()}, assign


# --------------------------------------- 链级 set prediction / 可微组装
def chain_complete_loss(mask_logit, y, assign, eps=1e-6):
    """[2026-11-03] 链级完整性损失 —— 直接监督"链", 而不是逐个节点。

    逐节点 BCE 只罚"这个节点进错槽", 一条真值链被拆成两半时, 每一半的单点损失
    都不大, 所以模型没有动机把它拼回去。这里对每个**匹配上的槽位**直接算该真值链的
    soft recall: 链内节点落到这个槽位的概率均值, 损失 = 1 - recall。
    同时算 soft precision 的镜像项, 罚"槽位里混进别的链的节点"。

    输入:
      mask_logit (N,K)  logits
      y          (N,)   真值链 id, -1 = 背景
      assign     (K,)   slot -> 真值对象下标 (-1 = 未匹配), 来自 slot_mask_loss
    """
    n, K = mask_logit.shape
    p = torch.sigmoid(mask_logit)                              # (N,K)
    objs = [int(v) for v in torch.unique(y) if int(v) >= 0]
    if not objs or assign is None:
        return mask_logit.sum() * 0.0
    tot, cnt = mask_logit.sum() * 0.0, 0
    for k in range(K):
        m = int(assign[k])
        if m < 0 or m >= len(objs):
            continue
        tgt = (y == objs[m]).float()
        s_in = (p[:, k] * tgt).sum()
        rec = s_in / tgt.sum().clamp_min(eps)
        prec = s_in / p[:, k].sum().clamp_min(eps)
        tot = tot + (1.0 - rec) + (1.0 - prec)
        cnt += 1
    return tot / max(cnt, 1)


def sinkhorn_assign(mask_logit, tau=1.0, iters=20, eps=1e-6):
    """[2026-11-03] Sinkhorn 软分配: 把 (N,K) logits 变成近双随机的 (N,K) 归属矩阵。

    让"一条径迹只能属于一条链"这个约束在训练时就以可微的方式存在, 而不是等到
    推理时的 argmax。行和 = 1 (每节点归一), 列和 = N/K (每个槽位容量均等)。
    """
    a = (mask_logit / max(tau, eps)).exp()                     # (N,K)
    n, K = a.shape
    col_tot = float(n) / float(max(K, 1))
    for _ in range(int(iters)):
        a = a / a.sum(dim=1, keepdim=True).clamp_min(eps)      # 行归一
        a = a / a.sum(dim=0, keepdim=True).clamp_min(eps) * col_tot   # 列归一
    return a


def sinkhorn_loss(mask_logit, y, assign, tau=1.0, iters=20):
    """用 Sinkhorn 软分配对匈牙利匹配结果做交叉熵 (可微组装)。"""
    n, K = mask_logit.shape
    objs = [int(v) for v in torch.unique(y) if int(v) >= 0]
    if not objs or assign is None:
        return mask_logit.sum() * 0.0
    A = sinkhorn_assign(mask_logit, tau=tau, iters=iters)
    T = torch.zeros(n, K, device=mask_logit.device)
    for k in range(K):
        m = int(assign[k])
        if 0 <= m < len(objs):
            T[:, k] = (y == objs[m]).float()
    return -(T * torch.log(A.clamp_min(1e-6))).sum(dim=1).mean()


@torch.no_grad()
def decode_slots(mask_logit, obj_logit, mask_thr=0.5):
    """逐节点取 argmax 槽位(天然排他); 概率低于阈值 -> dustbin(-1)。
    另外: objectness 低的槽位整体丢弃。"""
    p = torch.sigmoid(mask_logit)                             # (N,K)
    o = torch.sigmoid(obj_logit.mean(dim=0))                  # (K,)
    p = p * (o >= 0.5).float().unsqueeze(0)
    pmax, arg = p.max(dim=1)
    lab = arg.clone()
    lab[pmax < mask_thr] = -1
    return lab


# ------------------------------------------- M3: 分配式(替代"删边")
@torch.no_grad()
def decode_assign(score, edges, n, thr=0.5, dustbin_thr=0.5):
    """分配式解码: 按边分数从高到低, 把节点"分配"进已有簇(而非删边)。
    一个节点一旦入簇就不再改; 与任何簇都不相连的节点 -> dustbin。
    score: (E,) 同对象概率; edges: (2,E)。"""
    lab = torch.full((n,), -1, dtype=torch.long, device=score.device)
    order = torch.argsort(-score)
    clusters = []                                             # list[set]
    for idx in order.tolist():
        if score[idx] < thr:
            break
        a, b = int(edges[0, idx]), int(edges[1, idx])
        ca = next((c for c, s in enumerate(clusters) if a in s), -1)
        cb = next((c for c, s in enumerate(clusters) if b in s), -1)
        if ca == -1 and cb == -1:
            clusters.append({a, b})
        elif ca == -1:
            clusters[ca if False else cb].add(a)
        elif cb == -1:
            clusters[ca].add(b)
        elif ca != cb:
            clusters[ca] |= clusters[cb]
            clusters.pop(cb)
    for c, s in enumerate(clusters):
        for i in s:
            lab[i] = c
    return lab


def partition_relax_loss(mask_logit, y, eps=1e-6):
    """M4 严格版: 由槽位 mask 诱导成对"同对象"概率 P_ij = sum_k p_ik p_jk,
    直接惩罚"两个真值对象被合并"(P 高但不同) 与"一个对象被切开"(P 低但相同)。
    与逐边 BCE 的区别: 这是**划分级**目标, 经 mask 诱导, 梯度穿过整个划分。"""
    p = torch.sigmoid(mask_logit)                      # (N,K)
    P = p @ p.t()                                      # (N,N)
    same = (y[:, None] == y[None, :]) & (y[:, None] >= 0)
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    same = same & ~eye
    diff = (~same) & (~eye)
    total, n = 0.0, 0
    if same.any():
        total = total - torch.log(P[same].clamp_min(eps)).mean(); n += 1
    if diff.any():
        total = total - torch.log((1.0 - P[diff]).clamp_min(eps)).mean(); n += 1
    if n == 0:
        return mask_logit.sum() * 0.0
    return total / n
