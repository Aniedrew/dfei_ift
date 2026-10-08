#!/usr/bin/env python3
"""tt 边图 (line-graph) 注意力消息传递 —— 让 tt 边之间互相传消息, 学"三角传递性"。

动机
----
tt 边剪枝头当前只看**单条边自己的**表征/派生量, 看不到"相邻边"的强弱。
而真值链结构具有强的传递性: 若边 (i,j) 与边 (j,k) 都很强, 则 (i,k) 通常也应为强边
(三条径迹同属一条链)。单一 MLP 无归纳偏置表达这种二阶关系, 只能靠逐边特征近似。

做法 (line graph / 边图 GNN)
---------------------------
把每条 tt 边当作**边图的节点**: 两条 tt 边若共享一个 track 端点, 则在边图里相邻。
在边图上做 `n_rounds` 轮多头注意力, 每条边聚合其"邻居边"的表征:
    h_e' = h_e + Out( AttnAgg_{f∈N(e)} ( h_f ) )
其中 Out 的权重/bias **零初始化** -> 未训练时严格恒等 (与关闭开关逐位一致),
模型只在"邻居边确实有增益"时才学出非零映射。

实现要点
--------
* 邻接由 tt `edge_index` 的共享端点在**事件内**构造: 用 incidence 展开 (节点 -> 入射边)
  + 组内笛卡尔积, 全程向量化 (无 Python 双重循环), 未裁剪时复杂度 ~ Σ_v deg(v)^2。
* **严格按事件隔离**: 用每条边的 `edge_batch` (事件 id) 对邻接对做掩码, 绝不跨事件传消息
  (不同事件的节点全局索引本就互斥, 掩码是额外防线)。
* 平行边 (两条边共享**两个**端点) 会产生重复邻接对, 用 packed key 去重。

邻居数上限 (top-k 采样, 显存控制)
--------------------------------
每个共享端点 v 的入射边集合 (deg = d_v) 内部, 只保留"分数最高的 top-min(d_v, k) 条"作为
provider: 把 incidence 数组按 (节点升序, 边分数降序) 双次 argsort 分组排序, 每个接收槽位
只对组内**前 m = min(d_v, k) 个** position 展开 -> 代价从 Σ_v d_v^2 降到 Σ_v d_v·min(d_v, k),
且从不物化完整的 Σ d^2 张量。k=0/负数 -> 完全不排序不裁剪 (与旧行为逐位一致)。
分数用**邻居边自身的表征范数** (即时可算的代理; 若要精确 q·k 相似度需物化全部 pair, 违背
本机目的)。共享端点的所有边必属同一事件 (全局 track 索引事件互斥), 故排序天然事件隔离。

开关 (GNblocks.line_graph_attn 等)
---------------------------------
    line_graph_attn:          true/false   (默认 false, 关闭时零影响)
    line_graph_rounds:        注意力轮数    (默认 1)
    line_graph_heads:         注意力头数    (默认 4)
    line_graph_hidden:        注意力隐层宽度 (默认 32, 需被 heads 整除)
    line_graph_max_neighbors: 每共享端点的邻居数上限 k (默认 32; <=0 表示不限制, 保持旧行为)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmpgnn.model.context_prune import _segment_softmax


def build_line_graph(edge_index, edge_batch=None, max_neighbors=0, edge_score=None):
    """由 tt edge_index 构造边图有向邻接 (共享端点的边对)。

    edge_index:    (2,E) long, 第 0 行 = sender 全局 track 索引, 第 1 行 = receiver 全局 track 索引
    edge_batch:    (E,) long 可选, 每条边所属事件 id (用于事件隔离掩码; 可为 None)
    max_neighbors: int, 每个共享端点的邻居数上限 k。>0 时每组只保留分数最高的 min(deg,k) 条
                   provider (向量化双次 argsort); <=0 时不裁剪 (保持旧行为)。
    edge_score:    (E,) float 可选, 每条边的分数 (越大越优先保留), 用于 top-k 排序。
                   max_neighbors>0 且为 None 时退化为按入射槽位原顺序取前 k 条。

    返回 (line_src, line_dst): (P,) long, 表示 "边 line_src 收到来自边 line_dst 的消息"。
    两条边共享任一端点即相连 (双向各出现一次), 已去重且排除自环。
    """
    dev = edge_index.device
    e = int(edge_index.shape[1])
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if e == 0:
        return empty, empty

    k = int(max_neighbors)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    # incidence 展开: 每个 (节点, 入射边) 槽位
    nodes = torch.cat([src, dst])                                   # (2E,)
    eids = torch.cat([torch.arange(e, device=dev), torch.arange(e, device=dev)])

    if k > 0 and edge_score is not None:
        # 组内按分数降序: 先全局分数降序, 再对节点做稳定排序 -> 每组内仍保持分数降序
        sc = edge_score.reshape(-1).to(device=dev, dtype=torch.float32)[eids]
        _o = torch.argsort(sc, descending=True, stable=True)
        order = _o[torch.argsort(nodes[_o], stable=True)]
    else:
        order = torch.argsort(nodes, stable=True)                   # 按节点聚合
    ns, es = nodes[order], eids[order]
    uniq, counts = torch.unique_consecutive(ns, return_counts=True)
    gid = torch.repeat_interleave(torch.arange(uniq.numel(), device=dev), counts)  # 每个槽位的组号
    # 每个槽位展开的对数: 不裁剪 = deg; 裁剪 = min(deg, k) (只对组内前 k 个 position 展开)
    m = counts.clamp_max(k) if k > 0 else counts
    rep = m[gid]
    total = int(rep.sum().item())
    if total == 0:
        return empty, empty

    starts = torch.zeros(uniq.numel(), dtype=torch.long, device=dev)
    starts[1:] = torch.cumsum(counts, 0)[:-1]                       # 各组在排序数组中的起点
    # src 位置: 每个槽位 p 重复 rep[p] 次
    src_pos = torch.repeat_interleave(torch.arange(2 * e, device=dev), rep)
    # dst 位置: starts[gid[p]] + 0..rep[p]-1
    base = torch.repeat_interleave(starts[gid], rep)
    block_start = torch.cumsum(rep, 0) - rep
    within = torch.arange(total, device=dev) - torch.repeat_interleave(block_start, rep)
    dst_pos = base + within

    line_src = es[src_pos]
    line_dst = es[dst_pos]
    keep = line_src != line_dst                                     # 去自环
    # 事件隔离: 只保留同一事件的边对 (额外防线, 绝不跨事件传消息)
    if edge_batch is not None:
        eb = edge_batch.long().to(dev)
        keep = keep & (eb[line_src] == eb[line_dst])
    line_src, line_dst = line_src[keep], line_dst[keep]
    if line_src.numel() == 0:
        return empty, empty

    # 去重 (平行边共享两端点时会产生重复对): packed key -> 唯一有序对
    key = torch.unique(line_src * e + line_dst)
    return key // e, key % e


class LineGraphAttention(nn.Module):
    """tt 边图上的多头注意力 (零初始化输出投影 -> 起始恒等)。"""

    def __init__(self, edge_dim: int, n_rounds: int = 1, n_heads: int = 4,
                 hidden: int = 32, dropout: float = 0.0, max_neighbors: int = 32,
                 bias_cols=None, bias_hidden: int = 32):
        super().__init__()
        self.n_rounds = max(int(n_rounds), 1)
        self.n_heads = int(n_heads)
        self.hidden = int(hidden)
        self.max_neighbors = int(max_neighbors)     # 每共享端点的邻居上限; <=0 = 不限制
        assert self.hidden % self.n_heads == 0, "line_graph_hidden 必须能被 line_graph_heads 整除"
        self.h = self.hidden // self.n_heads
        self.dropout = float(dropout)

        self.q = nn.Linear(edge_dim, self.hidden)
        self.k = nn.Linear(edge_dim, self.hidden)
        self.v = nn.Linear(edge_dim, self.hidden)
        self.out = nn.Linear(self.hidden, edge_dim)
        # 末层置零 -> 残差分支起始为 0 -> 打开开关但未训练时与关闭严格等价
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # ==== [2026-10-05] 几何 pair-bias (Graphormer / AlphaFold 思路) ====
        # 动机: 小模型探针里"几何量与注意力直接耦合"是唯一与大模型 v637 不同、值得单变量检验的差异;
        #   主模型此前把几何量当普通输入列喂给 MLP, 从未让它直接调制 attention logit。
        # 形式: 对相邻边对 (e,f) 取 [b_e, b_f, |b_e-b_f|, b_e*b_f] -> 小 MLP -> 每头一个标量偏置,
        #   加到 logits 上。**对 (e,f) 交换对称**(用和/差/积而非顺序拼接), 与 0702 delta_z0 的教训一致。
        # bias_cols = 用 der_edges 的哪几列当 b (如顶点几何 doca/|Δ起点|); None/空 = 不加偏置(旧行为)。
        self.bias_cols = [int(c) for c in bias_cols] if bias_cols else None
        self.bias_dim = len(self.bias_cols) if self.bias_cols else 0
        self.bias_mlp = None
        if self.bias_dim > 0:
            self.bias_mlp = nn.Sequential(nn.Linear(4 * self.bias_dim, int(bias_hidden)),
                                          nn.ReLU(),
                                          nn.Linear(int(bias_hidden), self.n_heads))
            # 末层置零 -> 起始偏置恒为 0 -> 与"无 bias"逐位一致 (真正的单变量对照)
            nn.init.zeros_(self.bias_mlp[-1].weight)
            nn.init.zeros_(self.bias_mlp[-1].bias)

        print(f"[line_graph] tt 边图注意力启用: dim={edge_dim} rounds={self.n_rounds} "
              f"heads={self.n_heads} hidden={self.hidden} "
              f"max_neighbors={self.max_neighbors if self.max_neighbors > 0 else 'off'} "
              f"bias_cols={self.bias_cols if self.bias_cols else 'off'} "
              f"(输出投影零初始化, 起始恒等)")

    def forward(self, h, edge_index, edge_batch=None, bias_x=None):
        """h: (E,D) tt 边表征; edge_index: (2,E); edge_batch: (E,) 事件 id。
        bias_x: (E,K) 可选, 每条边的几何量 (只取 self.bias_cols 那几列做 pair-bias)。
        返回 (E,D): 每条边经边图注意力精修后的表征 (零初始化时恒等于输入)。"""
        e = int(h.shape[0])
        if e == 0 or self.hidden == 0:
            return h
        # top-k 排序代理分数 = 邻居边自身表征的 L2 范数 (即时可算, 无需物化 pair)
        score = h.detach().norm(dim=-1) if self.max_neighbors > 0 else None
        ls, ld = build_line_graph(edge_index, edge_batch,
                                  max_neighbors=self.max_neighbors, edge_score=score)
        if ls.numel() == 0:                     # 边图无任何相邻对 (每条边都孤立)
            return h

        # 几何 pair-bias: 只依赖静态几何量, 与轮数无关 -> 循环外算一次
        bias_term = None
        if self.bias_mlp is not None:
            if bias_x is None or bias_x.shape[0] != e or bias_x.shape[1] <= max(self.bias_cols):
                # [2026-10-08 FIX S2] 原来这里**静默**退化成"无 bias" -> 配置写了 bias_cols 但
                #   实际没生效却不报错 (本项目反复出现的"开关没生效"模式)。首次遇到时告警一次。
                if not getattr(self, "_warned_no_bias", False):
                    self._warned_no_bias = True
                    _why = ("bias_x=None (der_edges 未挂上?)" if bias_x is None
                            else f"bias_x.shape={tuple(bias_x.shape)} 与需要的 (E={e}, <= {max(self.bias_cols)+1} 列) 不匹配")
                    print(f"[line_graph] WARN: 配了 line_graph_bias_cols={self.bias_cols} 但本步用不上 "
                          f"({_why}) -> 该步退化为无 bias")
            else:
                b = bias_x[:, self.bias_cols].to(dtype=h.dtype, device=h.device)
                bi, bj = b[ls], b[ld]
                # [b_e, b_f, |b_e-b_f|, b_e*b_f]; b 为逐边量, 故偏置是"有向消息"的合法函数
                feat = torch.cat([bi, bj, (bi - bj).abs(), bi * bj], dim=-1)
                bias_term = self.bias_mlp(feat)                          # (P, heads)

        for _ in range(self.n_rounds):
            q = self.q(h).view(e, self.n_heads, self.h)
            k = self.k(h).view(e, self.n_heads, self.h)
            v = self.v(h).view(e, self.n_heads, self.h)
            logits = (q[ls] * k[ld]).sum(-1) / (self.h ** 0.5)      # (P, heads)
            if bias_term is not None:
                logits = logits + bias_term.to(logits.dtype)
            attn = _segment_softmax(logits, ls, e)                  # 按接收边分段 softmax
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            ctx = torch.zeros(e, self.n_heads, self.h, device=h.device, dtype=attn.dtype).index_add_(
                0, ls, attn.unsqueeze(-1) * v[ld])
            delta = self.out(ctx.reshape(e, self.hidden))
            h = h + delta.to(h.dtype)
        return h


if __name__ == "__main__":
    torch.manual_seed(0)

    # 事件 A: 3 条径迹 (0,1,2) 的边; 事件 B: 3 条径迹 (3,4,5) 的边
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0),
             (3, 4), (4, 3), (4, 5), (5, 4), (3, 5), (5, 3)]
    ei = torch.tensor([[a for a, _ in pairs], [b for _, b in pairs]])
    eb = torch.tensor([0] * 6 + [1] * 6)
    e = len(pairs)

    def edge_of(s, d):
        m = (ei[0] == s) & (ei[1] == d)
        return int(m.nonzero().reshape(-1)[0])

    # (1) 邻接正确性: 边 (0,1) 的邻居 = 与 0 或 1 共享端点的其它边 (事件 A 内)
    ls, ld = build_line_graph(ei, eb)
    nb = sorted(ld[ls == edge_of(0, 1)].tolist())
    # 共享端点 0: (1,0),(0,2),(2,0); 共享端点 1: (1,0),(1,2),(2,1) (反向边 (1,0) 也是合法邻居)
    want = sorted([edge_of(1, 0), edge_of(0, 2), edge_of(2, 0), edge_of(1, 2), edge_of(2, 1)])
    print("[1] 边(0,1) 邻居:", nb, " 期望:", want)
    assert nb == want, (nb, want)
    # 事件隔离: 边(0,1) 的邻居全部在事件 A
    assert all(eb[idx].item() == 0 for idx in nb), "邻居跨事件!"
    # 事件 B 的边不出现
    assert all(idx < 6 for idx in ld[ls < 6]), "事件 A 收到了 B 的消息"

    # (2) 零初始化 -> 恒等
    lg = LineGraphAttention(edge_dim=4, n_rounds=1, n_heads=2, hidden=8)
    h = torch.randn(e, 4)
    h0 = lg(h.clone(), ei, eb)
    assert torch.equal(h, h0), "零初始化下非恒等!"
    print("[2] 零初始化: 输出与输入逐位相等 ✓")

    # (3) 非零输出投影后: 与共享端点的另一条边相关, 与另一事件无关
    with torch.no_grad():
        lg.out.weight.normal_(0, 0.5)
        lg.out.bias.normal_(0, 0.1)
    h = torch.randn(e, 4)
    base = lg(h.clone(), ei, eb)
    k2 = h.clone(); k2[edge_of(1, 2)] += 5.0          # 改事件 A 的邻居边 (1,2)
    tgt = edge_of(0, 1)
    outA = lg(k2, ei, eb)
    print("[3] 改邻居边(1,2) -> 边(0,1) 输出变化 |Δ|=%.6f" % (outA[tgt] - base[tgt]).abs().sum().item())
    assert not torch.allclose(outA[tgt], base[tgt]), "信息未传递!"
    # 改事件 B 的边 -> 事件 A 的边(0,1) 输出不变
    k3 = h.clone(); k3[edge_of(4, 5)] += 5.0
    outB = lg(k3, ei, eb)
    assert torch.equal(outB[:6], base[:6]), "事件隔离失败: A 被 B 影响!"
    print("[3] 改事件B边(4,5) -> 事件A全部边输出不变 ✓")

    # (4) 可微性
    hg = torch.randn(e, 4, requires_grad=True)
    lg(hg, ei, eb).sum().backward()
    assert hg.grad is not None and torch.isfinite(hg.grad).all(), "梯度非法"
    print("[4] 梯度有限 ✓ (|grad|max=%.6g)" % hg.grad.abs().max().item())

    # (5) 多轮 + 空输入稳健
    lg2 = LineGraphAttention(edge_dim=4, n_rounds=3, n_heads=2, hidden=8)
    _ = lg2(h.clone(), ei, eb)
    assert lg(h[:0], ei[:, :0], eb[:0]).shape == (0, 4)
    print("[5] 多轮 / 空输入 稳健 ✓")

    # (6) [2026-10-05] 几何 pair-bias 路径
    lgb = LineGraphAttention(edge_dim=4, n_rounds=2, n_heads=2, hidden=8, bias_cols=[0, 1])
    assert lgb.bias_dim == 2 and lgb.bias_mlp is not None
    bx = torch.randn(e, 3)
    hb = torch.randn(e, 4)
    # 零初始化 (含 bias 末层) -> 仍与输入逐位相等
    assert torch.equal(hb, lgb(hb.clone(), ei, eb, bx)), "带 bias 的零初始化下非恒等!"
    # 打散权重 -> bias 真的进 logits: 只改 bias_x 的一列, 邻居边输出必须变
    with torch.no_grad():
        lgb.out.weight.normal_(0, 0.5); lgb.out.bias.normal_(0, 0.1)
        lgb.bias_mlp[-1].weight.normal_(0, 0.5); lgb.bias_mlp[-1].bias.normal_(0, 0.1)
    o1 = lgb(hb.clone(), ei, eb, bx)
    bx2 = bx.clone(); bx2[edge_of(0, 1), 0] += 3.0
    o2 = lgb(hb.clone(), ei, eb, bx2)
    assert not torch.allclose(o2[edge_of(1, 0)], o1[edge_of(1, 0)]), "bias 未进入 logits!"
    assert torch.isfinite(o1).all(), "输出含 nan/inf"
    # bias_x 列数不够 / 为 None -> 静默退化为无 bias (不报错)
    _ = lgb(hb.clone(), ei, eb, torch.randn(e, 1))
    _ = lgb(hb.clone(), ei, eb, None)
    # **端点交换不变性** (0702 delta_z0 泄漏的同类防线): 交换某条边两端点, 结果必须不变
    ei_sw = ei.clone()
    kk = edge_of(0, 1)
    ei_sw[0, kk], ei_sw[1, kk] = ei[1, kk].item(), ei[0, kk].item()
    o3 = lgb(hb.clone(), ei_sw, eb, bx)
    d_sw = (o3 - o1).abs().max().item()
    print("[6] 几何 pair-bias: 端点交换 max|Δ|=%.2e" % d_sw)
    assert d_sw < 1e-6, "端点交换改变了输出 (对称性泄漏)!"
    # 梯度有限
    hg2 = torch.randn(e, 4, requires_grad=True)
    lgb(hg2, ei, eb, bx).sum().backward()
    assert hg2.grad is not None and torch.isfinite(hg2.grad).all(), "bias 路径梯度非法"
    print("[6] 几何 pair-bias 路径 ✓ (含端点交换不变性与梯度)")

    print("全部自测通过 (断言全过)")
