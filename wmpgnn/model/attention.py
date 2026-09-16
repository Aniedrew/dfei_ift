#!/usr/bin/env python3
"""Track 级自注意力 (试验开关: DFEI.node_attention = true)。

方案 2 "纯内容版": 在 GN blocks 之后、decoder 之前, 让同一事件内的所有 track
互相做多头自注意力 (跨事件用 batch 索引 mask 掉), 给每条 track 一个事件级上下文。

参考: Particle Transformer (ParT) 的自注意力 / CMS MLPF (transformer 全事件重建)。
第一版故意不加 pair 特征 bias (JetFormer 经验: 纯内容注意力足够测机制),
若有效再升级为 edge-feature 作 attention bias 的版本。
"""
import torch
import torch.nn as nn


class TrackSelfAttention(nn.Module):
    """同事件 track 多头自注意力 (残差 + LayerNorm)。输入 x: [N, D], batch: [N]。

    edge_dim 非 None 时启用 ParT 式"边特征作注意力 bias" (P-MHA 思路):
    对每一条 tt 候选边 (rows, cols), 用 MLP 把边特征映射成每头一个 logit bias,
    scatter 进注意力分数 (重复边累加), 让"有结构关联的 track 对"在注意力里获得
    成对信息的先验。无边的 pair 仍靠纯内容注意力。
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0,
                 edge_dim: int | None = None):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} 必须能被 n_heads {n_heads} 整除"
        self.dim, self.n_heads = dim, n_heads
        self.h = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self._edge_bias = None
        if edge_dim is not None:
            self._edge_bias = nn.Linear(edge_dim, n_heads)   # 边特征 -> 每头一个 bias logit
            print(f"[attention] 边特征 bias 启用: edge_dim={edge_dim}")

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                edge_index: torch.Tensor | None = None,
                edge_feat: torch.Tensor | None = None) -> torch.Tensor:
        N = x.shape[0]
        h = self.h
        # qkv: [N, 3*dim] -> [N, 3, n_heads, h]
        qkv = self.qkv(x).reshape(N, 3, self.n_heads, h).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]          # each [N, n_heads, h]
        # 转成 [n_heads, N, h]: heads 作 matmul batch 维, 让所有 track 两两 attend
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        attn = q @ k.transpose(-2, -1) / (h ** 0.5)     # [n_heads, N, N]
        # ==== ParT 式: tt 边特征作注意力 bias ====
        if self._edge_bias is not None:
            rows, cols = edge_index[0].long(), edge_index[1].long()
            b = self._edge_bias(edge_feat)              # [E, n_heads]
            for h in range(self.n_heads):
                # attn[h]: [N,N]; (row,col) 处累加该头 bias, 重复边叠加
                attn[h].index_put_((rows, cols), b[:, h], accumulate=True)
        # 只允许同一事件内的 track 互相 attend
        mask = (batch.unsqueeze(1) != batch.unsqueeze(0)).unsqueeze(0)  # [1,N,N]
        attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v)                             # [n_heads, N, h]
        out = out.transpose(0, 1).reshape(N, self.dim)  # [N, D]
        return self.norm(x + self.out(out))
