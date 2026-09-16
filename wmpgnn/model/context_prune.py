#!/usr/bin/env python3
"""方案 A: 上下文感知剪枝头 (context-aware pruning head)。

动机
----
推理剪枝的裁决只依赖**该点/该边自己的**对数几率 (最后 GN block 的 MLP_infer)。
诊断 (2026-09-12) 显示: thr0.9 下 12436/17561 = 70.8% 真值链可用, **26% 的链被节点剪枝整条抹掉**;
关掉点剪枝能救回链 (N→16743) 但背景涌入 (NoneIso 86.7%)。即:
"整条链的点置信度都 ≤ thr" 是模型逐点判断无法自纠的系统性错误。

做法 (SAGPool / CRF 式)
----------------------
在剪枝裁决前, 让每个点的分数显式看到**邻域的点/边分数与表征**:
    s'_v = s_v + γ · MLP([ h_v, s_v, AttnAgg_{u∈N(v)}( h_u, s_u, e_vu, s_e ), mean_edge_score_v ])
残差分支末层置零 (γ=1) → 训练起始严格恒等 (对已有权重零扰动), 之后由监督 BCE 学出需要的邻域修正。
同链邻居都高置信时, 本点分数被"拉起来", 从而减少整链误删。

参考: SAGPool (Lee et al., ICML 2019, 自注意力打分) / Graph U-Net gPool (Gao & Ji, ICML 2019, top-k)
      / PSGNN (SDM 2024, 局部视角不足需全局信息) / GLANCE (2025, 注意力剪边)。

开关 (GNblocks.context_prune)
----------------------------
    mode:        off | mean | attn | attn_topk      (off=恒等, 零影响)
    heads, hidden, dropout, topk, refine_edge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9  # 掩码用有限大负数, 避免 -inf 参与 max 后出现 nan


def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """按 index 分组做 softmax。logits: [E, H], index: [E] -> [E, H]。"""
    dev, dt = logits.device, logits.dtype
    mx = torch.full((n, logits.shape[1]), NEG, device=dev, dtype=dt)
    mx = mx.index_reduce_(0, index, logits, "amax", include_self=True)
    z = torch.exp(logits - mx[index])
    den = torch.zeros(n, logits.shape[1], device=dev, dtype=dt).index_add_(0, index, z)
    return z / den[index].clamp_min(1e-12)


def _per_node_topk_mask(scores: torch.Tensor, index: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """每个 (index) 组内按 scores 降序保留前 k 个, 返回 [E] bool。向量化实现。"""
    e = scores.shape[0]
    if e <= k:
        return torch.ones(e, dtype=torch.bool, device=scores.device)
    o1 = torch.argsort(scores, descending=True, stable=True)     # 全局分数降序
    o2 = torch.argsort(index[o1], stable=True)                   # 组内仍保持分数降序
    counts = torch.bincount(index, minlength=n)
    offsets = torch.zeros(n, dtype=torch.long, device=scores.device)
    offsets[1:] = torch.cumsum(counts, 0)[:-1]
    pos = torch.arange(e, device=scores.device) - offsets[index[o1][o2]]
    keep_in_o2 = pos < k
    keep = torch.ones(e, dtype=torch.bool, device=scores.device)
    keep[o2] = keep_in_o2
    return keep


class ContextPruneHead(nn.Module):
    """把邻域点/边分数喂进剪枝裁决 (γ 门控残差, 初始恒等)。"""

    def __init__(self, node_dim: int, edge_dim: int, cfg: dict):
        super().__init__()
        self.mode = str(cfg.get("mode", "attn"))
        self.n_heads = int(cfg.get("heads", 4))
        self.hidden = int(cfg.get("hidden", 32))
        self.topk = int(cfg.get("topk", 0))
        self.refine_edge = bool(cfg.get("refine_edge", False))
        self.dropout = float(cfg.get("dropout", 0.0))
        assert self.hidden % self.n_heads == 0, "hidden 必须能被 heads 整除"
        self.h = self.hidden // self.n_heads

        msg_dim = node_dim + 1 + edge_dim + 1          # [h_u, s_u, e_vu, s_e]
        self.q = nn.Linear(node_dim + 1, self.hidden)
        self.k = nn.Linear(msg_dim, self.hidden)
        self.v = nn.Linear(msg_dim, self.hidden)
        self.out = nn.Linear(self.hidden, self.hidden)
        # 节点精修: [h_v, s_v, ctx, mean_incident_edge_score]
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + 1 + self.hidden + 1, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        self.gamma = nn.Parameter(torch.ones(1))       # γ=1, 末层置零 -> 起始恒等且梯度可通
        if self.refine_edge:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim + 3, self.hidden), nn.ReLU(),
                nn.Linear(self.hidden, 1),
            )
            self.gamma_e = nn.Parameter(torch.ones(1))
        # 只把**末层**置零 (残差分支起始为 0 -> 恒等)。注意: 不可同时把 γ 也置 0,
        # 否则 ∂L/∂γ = <∂L/∂s, f> 与 ∂L/∂W2 = γ·(...) 同时为 0, 头将永久锁死在恒等 (实测教训)。
        nn.init.zeros_(self.node_mlp[-1].weight)
        nn.init.zeros_(self.node_mlp[-1].bias)
        if self.refine_edge:
            nn.init.zeros_(self.edge_mlp[-1].weight)
            nn.init.zeros_(self.edge_mlp[-1].bias)
        print(f"[context_prune] 启用: mode={self.mode} heads={self.n_heads} hidden={self.hidden} "
              f"topk={self.topk} refine_edge={self.refine_edge}")

    def forward(self, x, s_node, edge_index, e_lat, s_edge):
        """x:[N,Dn] 节点表征; s_node:[N,1] 节点对数几率;
        edge_index:[2,E]; e_lat:[E,De] 边表征; s_edge:[E,1] 边对数几率。
        返回精修后的 (s_node', s_edge')。"""
        n = x.shape[0]
        dev = x.device

        if edge_index.shape[1] == 0:
            return s_node, s_edge

        tgt, src = edge_index[1].long(), edge_index[0].long()

        # ---- 边精修 (用两端点分数): 让边裁决也看到端点信息 ----
        if self.refine_edge:
            e_in = torch.cat([e_lat, s_edge, s_node[src], s_node[tgt]], dim=1)
            s_edge_new = s_edge + self.gamma_e * self.edge_mlp(e_in)
        else:
            s_edge_new = s_edge

        # ---- 节点精修 (邻域注意力) ----
        msg = torch.cat([x[src], s_node[src], e_lat, s_edge_new], dim=1)
        if self.mode == "mean":
            # 对照档: 等权均值 (无注意力) —— 先投影到 hidden 再平均
            pv = self.v(msg)
            ctx = torch.zeros(n, self.hidden, device=dev, dtype=pv.dtype).index_add_(0, tgt, pv)
            cnt = torch.zeros(n, 1, device=dev, dtype=pv.dtype).index_add_(
                0, tgt, torch.ones(msg.shape[0], 1, device=dev, dtype=pv.dtype))
            ctx = F.relu(self.out(ctx / cnt.clamp_min(1)))
        else:
            q = self.q(torch.cat([x[tgt], s_node[tgt]], dim=1)).view(-1, self.n_heads, self.h)
            k = self.k(msg).view(-1, self.n_heads, self.h)
            v = self.v(msg).view(-1, self.n_heads, self.h)
            logits = (q * k).sum(-1) / (self.h ** 0.5)          # [E, H]
            if self.mode == "attn_topk" and self.topk > 0:
                keep = _per_node_topk_mask(logits.mean(-1).detach(), tgt, n, self.topk)
                logits = logits.masked_fill(~keep.unsqueeze(-1), NEG)
            attn = _segment_softmax(logits, tgt, n)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            ctx = torch.zeros(n, self.n_heads, self.h, device=dev, dtype=attn.dtype).index_add_(
                0, tgt, attn.unsqueeze(-1) * v)
            ctx = F.relu(self.out(ctx.reshape(n, self.hidden)))

        # 自身入射边的平均分数 (局部证据标量)
        e_agg = torch.zeros(n, 1, device=dev, dtype=s_edge_new.dtype).index_add_(0, tgt, s_edge_new)
        e_cnt = torch.zeros(n, 1, device=dev, dtype=s_edge_new.dtype).index_add_(
            0, tgt, torch.ones_like(s_edge_new))
        e_agg = e_agg / e_cnt.clamp_min(1)

        feat = torch.cat([x, s_node, ctx, e_agg], dim=1)
        s_node_new = s_node + self.gamma * self.node_mlp(feat)
        return s_node_new, s_edge_new
