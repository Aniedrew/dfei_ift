"""方向2: 集合预测头 (set prediction: slot + mask + 匈牙利匹配)  [2026-10-13]

动机
----
当前下游是"逐对四分类 + 取连通分量/组装", **从不显式回答**"这个事件里有几条链、
每条链含哪些径迹"。集合预测 (DETR / MaskFormer 式) 一次性输出 K 个"链槽位":
  (i)  该槽位是否存在 (objectness);
  (ii) 它对**每条径迹**的归属 mask。
训练用匈牙利匹配把预测槽位与真值链一一对齐; 它**天然表达"链的数量"**与
"每条径迹只属于一条链", 并直接对着唯一验收判据 (整链完整恢复) 建模。

pair-wise 设计 (关键; 上一轮原型败在"只对节点表征池化"丢了成对信息)
------------------------------------------------------------------
槽位 query 与 **tt 边表征** 做 cross-attention, 再把边级信息按端点聚合回径迹;
mask 头的输入显式含两端表征与边聚合:
    feat[t, k] = [ h_t, c_k, edgeAgg_{t,k}, h_t * c_k ]
-> 直接用了成对 (边) 信息, 不是只对节点表征池化。

归属归一化 (两种, 开关 assign_mode)
----------------------------------
  - "softmax_bg" (默认): 逐径迹 softmax 到 K 个槽位 + 1 个"背景"类 -> **天然排他**;
  - "sigmoid"          : 逐 (径迹, 槽位) 独立 sigmoid -> 排他性靠推理 argmax。
选 softmax_bg 作默认的理由: 要保证"每条径迹只属于一个槽位", 逐径迹 softmax 到
"不含径迹的背景类"最稳 (背景类吸收"未匹配/无链"的径迹; 硬性排他由 argmax 保证)。

排他性 / 事件隔离 / 边界
------------------------
  - 排他性: softmax_bg 下每个径迹的类别分布只有 1 个 argmax -> 只属于一个槽位 (或背景)。
  - 事件隔离: 前向 (forward_event) 与匹配 (setpred_event_loss) 都**逐事件**进行;
    不同事件的径迹绝不进同一槽位匹配 (对抗用例见 test_setpred.py)。
  - 空事件 / 无链 / 链数 > K 均安全 (见 setpred_event_loss 的 G==0 分支与
    Hungarian 的矩形处理)。

缺省关闭 (setpred_use=False, setpred_loss_w=0) -> 与 v654 逐位一致。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:                      # pragma: no cover - scipy 缺失时用贪心回退
    HAS_SCIPY = False


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class SetPredHead(nn.Module):
    """K 个可学习链槽位: 与 tt 边表征 cross-attention -> 逐径迹归属 + 槽位存在性。

    输入 (逐事件, 由调用方按事件切片): 节点表征 h_node [N,d_n] / tt 边表征 h_edge [M,d_e]
    / edge_index [2,M]。输出: exist_logit [K], assign_logit [N,K] (+ softmax_bg 的 bg_logit)。
    """

    def __init__(self, d_node, d_edge, n_slots=8, hidden=64, heads=4,
                 assign_mode="softmax_bg", dropout=0.0, edge_bias=True, pairwise=True):
        super().__init__()
        self.n_slots = int(n_slots)
        self.hidden = int(hidden)
        self.assign_mode = assign_mode
        self.edge_bias = bool(edge_bias)
        # pairwise=True (默认): mask 建在"节点表征 + 槽位 query 对 tt 边表征 cross-attention
        #   再按端点聚合回径迹"上 (显式用到成对/边信息);
        # pairwise=False: **对照臂** —— 只用节点表征 [h_t, c_k] 生成 mask (丢掉成对信息),
        #   用来复现/量化"上一轮原型败在只对节点池化"的结论 (setpred_node_mask 臂)。
        self.pairwise = bool(pairwise)
        self.node_proj = nn.Linear(int(d_node), self.hidden)
        self.edge_proj = nn.Linear(int(d_edge), self.hidden)
        self.slot = nn.Parameter(torch.randn(self.n_slots, self.hidden) * 0.02)
        self.slot_norm = nn.LayerNorm(self.hidden)
        self.edge_attn = nn.MultiheadAttention(self.hidden, int(heads),
                                               dropout=float(dropout), batch_first=True)
        # 4*hidden: [h_t, c_k, edgeAgg, h_t*c_k] -> 逐 (径迹, 槽位) 的归属 logit
        self.mask_mlp = _mlp(4 * self.hidden, self.hidden, 1)
        self.obj_mlp = _mlp(self.hidden, self.hidden, 1)
        self.bg_head = nn.Linear(self.hidden, 1) if assign_mode == "softmax_bg" else None

    def forward_event(self, h_node, edge_index, h_edge):
        """单事件前向。返回 dict:
            exist_logit [K], assign_logit [N,K], assign_prob [N,K]
            (+ bg_logit [N] 仅 softmax_bg)。
        """
        N = int(h_node.shape[0])
        K = self.n_slots
        dev = h_node.device
        H = self.node_proj(h_node)                              # [N, h]
        M = int(h_edge.shape[0]) if h_edge is not None else 0
        if self.pairwise and M > 0:
            E = self.edge_proj(h_edge)                          # [M, h]
            Q = self.slot.unsqueeze(0)                          # [1, K, h]
            ctx, attn_w = self.edge_attn(Q, E.unsqueeze(0), E.unsqueeze(0),
                                         need_weights=True)     # [1,K,h], [1,K,M]
            C = self.slot_norm(ctx.squeeze(0))                  # [K, h]
            attn_w = attn_w.squeeze(0)                          # [K, M]
            # 边 -> 节点 聚合 (逐槽位): agg[k, t] = Σ_{m∋t} attn_w[k,m] * E[m] / deg(t)
            a = edge_index[0].long()
            b = edge_index[1].long()
            w_mk = attn_w.transpose(0, 1)                       # [M, K]
            val = w_mk.unsqueeze(-1) * E.unsqueeze(1)           # [M, K, h]
            inc_val = torch.cat([val, val], 0).to(H.dtype)      # [2M, K, h]
            inc_node = torch.cat([a, b], 0)                     # [2M]
            agg = torch.zeros(N, K, self.hidden, device=dev, dtype=H.dtype).index_add(
                0, inc_node, inc_val)
            cnt = torch.zeros(N, device=dev, dtype=H.dtype).index_add(
                0, inc_node, torch.ones(2 * M, device=dev, dtype=H.dtype))
            agg = agg / cnt.clamp_min(1.0)[:, None, None]
        else:
            C = self.slot_norm(self.slot)                       # 无候选边: 退化纯槽位
            agg = torch.zeros(N, K, self.hidden, device=dev, dtype=H.dtype)
        if not self.edge_bias:
            agg = torch.zeros_like(agg)
        Ht = H.unsqueeze(1).expand(N, K, self.hidden)
        Ck = C.unsqueeze(0).expand(N, K, self.hidden)
        # pairwise=False (对照臂): 丢掉边聚合与节点×槽位交互项 -> feat=[h_t, c_k, 0, 0], 纯节点池化。
        inter = (Ht * Ck) if self.pairwise else torch.zeros_like(Ht)
        feat = torch.cat([Ht, Ck, agg, inter], -1)              # [N, K, 4h]
        assign_logit = self.mask_mlp(feat).squeeze(-1)          # [N, K]
        exist_logit = self.obj_mlp(C).squeeze(-1)               # [K]
        out = dict(exist_logit=exist_logit, assign_logit=assign_logit)
        if self.assign_mode == "softmax_bg":
            bg = self.bg_head(H).squeeze(-1)                    # [N]
            out["bg_logit"] = bg
            full = torch.cat([assign_logit, bg.unsqueeze(-1)], -1)     # [N, K+1]
            out["assign_prob"] = torch.softmax(full, dim=-1)[:, :K]
        else:
            out["assign_prob"] = torch.sigmoid(assign_logit)
        return out


# ===========================================================================
# 匹配 / 损失 / 解码
# ===========================================================================
def dice_cost(exist_prob, assign_prob, gt_masks):
    """DETR 匹配代价 (越小越好): -exist_prob + (1 - Dice)。

    exist_prob [K] / assign_prob [N,K] / gt_masks [G,N] (bool 或 0-1)。
    返回 [K, G]。全张量, 期望在 no_grad 下调用 (匹配是"目标分配", 不可导)。
    """
    K = int(exist_prob.shape[0])
    G = int(gt_masks.shape[0])
    if G == 0:
        return torch.zeros((K, 0), device=assign_prob.device, dtype=assign_prob.dtype)
    gt = gt_masks.to(assign_prob.dtype)                         # [G, N]
    inter = assign_prob.transpose(0, 1) @ gt.transpose(0, 1)    # [K,N]@[N,G] -> [K,G]
    pu = assign_prob.sum(0).unsqueeze(1)                        # [K,1]
    gu = gt.sum(1).unsqueeze(0)                                 # [1,G]
    dice = 2.0 * inter / (pu + gu).clamp_min(1e-6)
    return -exist_prob.unsqueeze(1) + (1.0 - dice)


def hungarian_match(cost):
    """[K,G] 代价 -> 一对一匹配对 [(k,g)] (每槽位/每条真值链至多用一次)。

    scipy 不可用时退化为"按代价从小到大贪心"(同样保证一对一)。
    """
    K, G = int(cost.shape[0]), int(cost.shape[1])
    if K == 0 or G == 0:
        return []
    c = cost.detach().cpu().numpy()
    if HAS_SCIPY:
        r, col = linear_sum_assignment(c)
        return list(zip(r.tolist(), col.tolist()))
    pairs, used_k, used_g = [], set(), set()
    for idx in np.argsort(c, axis=None):
        k, g = divmod(int(idx), G)
        if k in used_k or g in used_g:
            continue
        used_k.add(k); used_g.add(g); pairs.append((k, g))
    return pairs


def _objectness_loss(exist_logit, exist_t, dustbin_weight=1.0):
    """逐槽位 objectness BCE; **未匹配槽位** (target=0, dustbin 语义) 的权重乘 dustbin_weight。

    dustbin_weight=1.0 (默认) 时与普通 mean BCE 逐位一致 -> 缺省不改变行为。
    """
    bce = F.binary_cross_entropy_with_logits(exist_logit, exist_t, reduction="none")
    if float(dustbin_weight) == 1.0:
        return bce.mean()
    w = torch.where(exist_t > 0.5, torch.ones_like(exist_t),
                    torch.full_like(exist_t, float(dustbin_weight)))
    return (bce * w).mean()


def setpred_event_loss(out, gt_masks, exist_weight=1.0, assign_weight=1.0,
                       assign_mode="softmax_bg", dustbin_weight=1.0):
    """单事件的集合预测损失 (匈牙利匹配后)。gt_masks: [G,N] (bool / 0-1)。

    返回 (loss, diag)。空事件 / 无链 (G==0) -> 只监督 objectness=0。
    排他: softmax_bg 下 loss = 逐径迹 CE 到 (匹配槽位 | 背景) 的**单一**类别 ->
          每个径迹的软目标天然只落在一个类别上。
    """
    exist_logit = out["exist_logit"]
    assign_logit = out["assign_logit"]
    N, K = int(assign_logit.shape[0]), int(assign_logit.shape[1])
    dev = assign_logit.device
    exist_t = torch.zeros(K, device=dev, dtype=assign_logit.dtype)
    G = int(gt_masks.shape[0])
    if G == 0:
        l_exist = _objectness_loss(exist_logit, exist_t, dustbin_weight)
        return exist_weight * l_exist, dict(n_gt=0, n_matched=0,
                                            exist=float(l_exist.detach()), mask=0.0)
    with torch.no_grad():
        cost = dice_cost(torch.sigmoid(exist_logit), out["assign_prob"], gt_masks)
        pairs = hungarian_match(cost)
    if assign_mode == "softmax_bg":
        full = torch.cat([assign_logit, out["bg_logit"].unsqueeze(-1)], -1)   # [N, K+1]
        target = torch.full((N,), K, device=dev, dtype=torch.long)            # 背景类
        for k, g in pairs:
            target = torch.where(gt_masks[g].to(dev), torch.full_like(target, k), target)
        l_mask = F.cross_entropy(full, target)
    else:
        l_mask = torch.zeros((), device=dev, dtype=assign_logit.dtype)
        for k, g in pairs:
            l_mask = l_mask + F.binary_cross_entropy_with_logits(
                assign_logit[:, k], gt_masks[g].to(assign_logit.dtype))
        if pairs:
            l_mask = l_mask / len(pairs)
    ks = [k for k, _ in pairs]
    if ks:
        exist_t[torch.as_tensor(ks, device=dev)] = 1.0
    l_exist = _objectness_loss(exist_logit, exist_t, dustbin_weight)
    loss = exist_weight * l_exist + assign_weight * l_mask
    return loss, dict(n_gt=G, n_matched=len(pairs),
                      exist=float(l_exist.detach()), mask=float(l_mask.detach()))


def decode_slots(out, exist_thr=0.5, assign_mode="softmax_bg", assign_thr=0.5):
    """槽位输出 -> 逐径迹标签 [N] (-1 = 背景 / 未分配)。

    - softmax_bg: argmax over (K 槽位 + 背景); 落到背景 -> -1; 落到未存活槽位 -> -1。
      (不用概率阈值: 背景类已表达"不属于任何存活槽位")
    - sigmoid   : 逐径迹取最大槽位概率, 且需 >= assign_thr 且该槽位存在。
    返回 (labels [N] long, alive [K] bool)。
    """
    dev = out["assign_logit"].device
    N, K = int(out["assign_logit"].shape[0]), int(out["assign_logit"].shape[1])
    exist_prob = torch.sigmoid(out["exist_logit"])             # [K]
    alive = exist_prob > exist_thr
    labels = torch.full((N,), -1, dtype=torch.long, device=dev)
    if N == 0 or int(alive.sum()) == 0:
        return labels, alive
    if assign_mode == "softmax_bg":
        full = torch.cat([out["assign_logit"], out["bg_logit"].unsqueeze(-1)], -1)  # [N,K+1]
        cls = full.argmax(-1)                                  # [N]
        for t in range(N):
            k = int(cls[t])
            if k < K and bool(alive[k]):
                labels[t] = k
    else:
        m = out["assign_prob"] * alive.to(out["assign_prob"].dtype).unsqueeze(0)
        best, k = m.max(-1)
        labels = torch.where(best >= assign_thr, k, torch.full_like(k, -1))
    return labels, alive
