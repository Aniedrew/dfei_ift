#!/usr/bin/env python3
"""按事件分段计算的径迹对 "是否来自同一 PV" 软重叠特征 (纯张量函数, 端到端可微)。

动机
----
边剪枝头目前完全看不到 PV 关联头的信息。本模块把每条 (track,pv) 关联边的 logit
转成概率 p_i(pv) —— 在**该径迹所属事件内**对所有 pv 做 softmax(温度=1) —— 再为
每条 tt 边 (i,j) 算三个对称标量, 供剪枝头使用:

    [0] soft_overlap = Σ_pv p_i(pv)·p_j(pv)          两条径迹关联分布的重叠
    [1] same_argmax  = 1[ argmax_pv(i) == argmax_pv(j) ]
    [2] min_overlap  = max_pv min( p_i(pv), p_j(pv) )

实现要点
--------
* softmax **必须按事件分段**: 不同事件的 pv 若混在一起归一化, 概率无物理意义。
* 无 (track,pv) 边的径迹 p_i ≡ 0 (不产生 NaN), 与之相连的 tt 边三列均输出 0。
* 全程逐事件处理 (事件数一般 8~64, 循环开销可忽略), 事件内全部向量化。
* 结果恒在 [0,1] (soft_overlap / min_overlap 都是概率的乘积/求和)。
"""
import torch

NEG = -1e9  # 掩码用有限大负数, 避免 -inf 参与 exp/max 后出现 nan (与 context_prune.py 一致)


def pv_same_soft(pv_logits, pv_edge_index, tt_edge_index, n_tracks, batch, eps=1e-12):
    """径迹对 "同一 PV" 软重叠特征 (按事件分段 softmax)。

    pv_logits:      (E_pv,) 或 (E_pv,1) float tensor, ('tracks','to','pvs') 边的关联 logit
    pv_edge_index:  (2,E_pv) long, 第 0 行 = track 全局索引, 第 1 行 = pv 全局索引
    tt_edge_index:  (2,E_tt) long, 径迹-径迹边 (全局索引, 可能双向重复)
    n_tracks:       int, batch 内径迹总数
    batch:          (n_tracks,) long, 每条径迹的事件 id (0..n_ev-1)
    返回: (E_tt, 3) float tensor, 列依次为
      [0] soft_overlap = sum_pv p_i(pv) * p_j(pv)
      [1] same_argmax  = 1.0 若 argmax_pv(i) == argmax_pv(j) 否则 0.0
      [2] min_overlap  = max_pv min(p_i(pv), p_j(pv))

    其中 p_i 是 track i 在**其所属事件内**对所有 pv 的 softmax (温度=1)。
    设计取舍:
      * 重复的 (track,pv) 对: 其 logit **求和**后再做 softmax (等价于把重复边当同一
        证据叠加), 不会崩。
      * 跨事件的 tt 边 (两端点不同事件, 理论上不该出现): 三列均输出 0。
    """
    dev = pv_logits.device
    dt = pv_logits.dtype

    logits = pv_logits.reshape(-1)                        # (E_pv,1) -> (E_pv,)
    pv_ei = pv_edge_index.long().to(dev)
    tt_ei = tt_edge_index.long().to(dev)
    batch = batch.long().to(dev)

    n_tt = int(tt_ei.shape[1])
    out = torch.zeros(n_tt, 3, device=dev, dtype=dt)
    if n_tt == 0 or int(n_tracks) == 0 or logits.numel() == 0:
        return out

    n_ev = int(batch.max().item()) + 1

    tt_src, tt_dst = tt_ei[0], tt_ei[1]
    tt_same_evt = batch[tt_src] == batch[tt_dst]          # 排除理论上的跨事件边
    pv_evt = batch[pv_ei[0]]                              # 每条 pv 边所属事件

    idx_all, val_all = [], []
    for e in range(n_ev):
        # 该事件的径迹 (全局索引, 升序) 及其事件内行号映射
        tr_global = (batch == e).nonzero(as_tuple=False).reshape(-1)
        n_tr = int(tr_global.numel())
        if n_tr == 0:
            continue

        emask = pv_evt == e
        if not bool(emask.any()):
            continue                                      # 该事件无 pv 边 -> 相关 tt 边保持 0
        tr_g = pv_ei[0][emask]
        pv_g = pv_ei[1][emask]
        lg = logits[emask]

        row_of_global = torch.full((int(n_tracks),), -1, dtype=torch.long, device=dev)
        row_of_global[tr_global] = torch.arange(n_tr, device=dev)
        r = row_of_global[tr_g]                           # 事件内行号

        _, inv = torch.unique(pv_g, return_inverse=True)  # 全局 pv -> 事件内紧凑列号
        n_pv = int(inv.max().item()) + 1
        c = inv

        # 重复 (track,pv) 对: logit 求和; has 标记哪些格有边
        ones = torch.ones_like(lg)
        seen = torch.zeros(n_tr, n_pv, dtype=dt, device=dev).index_put(
            (r, c), ones, accumulate=True)
        acc = torch.zeros(n_tr, n_pv, dtype=dt, device=dev).index_put(
            (r, c), lg, accumulate=True)
        has = seen > 0                                    # (n_tr, n_pv) bool
        dense = torch.where(has, acc, torch.full_like(acc, NEG))  # 无边 = NEG

        # ---- 按事件分段 softmax (逐行掩码; 无 pv 边的行 -> 全 0, 无 NaN) ----
        has_any = has.any(dim=1, keepdim=True)            # (n_tr, 1)
        row_max = dense.max(dim=1, keepdim=True).values
        row_max = torch.where(has_any, row_max, torch.zeros_like(row_max))
        w = torch.exp(dense - row_max)                    # 背景项 exp(NEG - m) ~ 0
        w = torch.where(has, w, torch.zeros_like(w))
        p = w / w.sum(dim=1, keepdim=True).clamp_min(eps)  # 空行: 0 / eps = 0

        # ---- 该事件的 tt 边 (向量化) ----
        tmask = (batch[tt_src] == e) & tt_same_evt
        idx = tmask.nonzero(as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue

        pi = p[row_of_global[tt_src[idx]]]               # (E_e, n_pv)
        pj = p[row_of_global[tt_dst[idx]]]

        soft = (pi * pj).sum(dim=1)
        minov = torch.minimum(pi, pj).max(dim=1).values
        same = (pi.argmax(dim=1) == pj.argmax(dim=1)).to(dt)

        valid = has.any(dim=1)[row_of_global[tt_src[idx]]] & \
            has.any(dim=1)[row_of_global[tt_dst[idx]]]
        res = torch.stack([soft, same, minov], dim=1)
        res = torch.where(valid.unsqueeze(1), res, torch.zeros_like(res))

        idx_all.append(idx)
        val_all.append(res)

    if val_all:
        out = out.index_copy(0, torch.cat(idx_all), torch.cat(val_all))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)

    # 事件 A: 3 条径迹 (全局 0,1,2), 2 个 pv (全局 0,1)
    #   track0/track1 强烈关联 pv0, track2 关联 pv1
    # 事件 B: 3 条径迹 (全局 3,4,5), 1 个 pv (全局 2); track5 无任何 pv 边
    n_tracks = 6
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    pv_edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 4],      # track 全局索引
        [0, 1, 0, 1, 0, 1, 2, 2],      # pv 全局索引
    ])
    pv_logits = torch.tensor([10.0, -10.0, 8.0, -8.0, -10.0, 10.0, 0.0, 0.3])

    tt_pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
                (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4)]
    tt_edge_index = torch.tensor([[a for a, _ in tt_pairs],
                                  [b for _, b in tt_pairs]])

    def edge_of(s, d):
        m = (tt_edge_index[0] == s) & (tt_edge_index[1] == d)
        return int(m.nonzero().reshape(-1)[0])

    out = pv_same_soft(pv_logits, pv_edge_index, tt_edge_index, n_tracks, batch)
    e01, e02, e34 = edge_of(0, 1), edge_of(0, 2), edge_of(3, 4)

    print("[值] (0,1)=%s  (0,2)=%s  (3,4)=%s  (3,5)=%s" % (
        out[e01].tolist(), out[e02].tolist(), out[e34].tolist(), out[edge_of(3, 5)].tolist()))

    # 1) 事件 A: 同一 pv 的两条径迹重叠 ~1, 不同 pv ~0
    assert abs(out[e01, 0].item() - 1.0) < 1e-4, out[e01]
    assert out[e02, 0].item() < 1e-4, out[e02]
    assert out[e01, 1].item() == 1.0
    assert out[e02, 1].item() == 0.0
    # 2) 事件 B: 只有 1 个 pv -> softmax 恒为 1
    assert abs(out[e34, 0].item() - 1.0) < 1e-6, out[e34]
    assert out[e34, 1].item() == 1.0
    assert abs(out[e34, 2].item() - 1.0) < 1e-6, out[e34]
    # 3) track5 无 pv 边 -> 相关 tt 边三列全 0
    for (s, d) in [(3, 5), (5, 3), (4, 5), (5, 4)]:
        row = out[edge_of(s, d)]
        assert torch.count_nonzero(row).item() == 0, (s, d, row)
    # (d) (E_pv,1) 输入自动 squeeze
    out2 = pv_same_soft(pv_logits.view(-1, 1), pv_edge_index, tt_edge_index, n_tracks, batch)
    assert torch.allclose(out, out2), "unsqueeze 形状结果不一致"
    # (c) 重复 (track,pv) 对: 不崩 (logit 求和)
    dup_ei = torch.cat([pv_edge_index, pv_edge_index[:, :1]], dim=1)
    dup_lg = torch.cat([pv_logits, torch.tensor([0.0])])
    out_dup = pv_same_soft(dup_lg, dup_ei, tt_edge_index, n_tracks, batch)
    assert out_dup.shape == (len(tt_pairs), 3)
    # (b) 单径迹事件 + 单 pv: softmax=1, 无 tt 边 -> 形状 (0,3)
    o_single = pv_same_soft(torch.tensor([3.0]), torch.tensor([[0], [0]]),
                            torch.zeros(2, 0, dtype=torch.long), 1, torch.tensor([0]))
    assert o_single.shape == (0, 3)
    # 空输入稳健
    assert pv_same_soft(torch.zeros(0), torch.zeros(2, 0, dtype=torch.long),
                        tt_edge_index, n_tracks, batch).numel() == len(tt_pairs) * 3

    # 4) 可微性: 对 pv_logits 有梯度且有限
    lg = pv_logits.clone().requires_grad_(True)
    o = pv_same_soft(lg, pv_edge_index, tt_edge_index, n_tracks, batch)
    o.sum().backward()
    assert lg.grad is not None, "pv_logits.grad 为 None"
    assert torch.isfinite(lg.grad).all(), "梯度含 NaN/Inf"
    print("[梯度] 非零元素 %d, |grad| max=%.6g" % (
        int(torch.count_nonzero(lg.grad).item()), float(lg.grad.abs().max())))

    # 5) 值域 [0,1]
    assert (out >= -1e-6).all() and (out <= 1.0 + 1e-6).all(), "返回值越界"
    lo, hi = out.min(dim=0).values, out.max(dim=0).values
    for i, name in enumerate(["soft_overlap", "same_argmax", "min_overlap"]):
        print("  %-13s min=%.6f max=%.6f" % (name, lo[i].item(), hi[i].item()))

    print("全部自测通过 (断言全过)")
