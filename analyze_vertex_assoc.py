"""次级顶点归属 (standalone 小题): 上下文上界探针 + 独立小模型

背景 (2026-09-30)
    跨链假边里真正难的是"同一 PV、不同次级顶点"的那些 (fake_inter same_pv AUC 0.736;
    而 diff_pv 有 0.897)。所以"区分链内/链外"本质上是 **track -> 次级顶点归属** 问题。
    此前所有模型侧手段都在 DFEI 那个 6 任务 GNN 里做, 结论还被"评测事件抽样不可复现"
    污染过。这里把它**单独拎出来**: 真值干净、指标阈值无关(难池 AP)、模型小、迭代快,
    与主模型完全独立 -> 可以并行训练。

两个子命令
    probe : GBDT 逐组累加特征, 回答"上下文(第三方信息)能不能突破局部特征的上界"。
            这是"要不要上注意力/匹配"的闸门 —— 若推不动, 上限就是信息限制, 加结构无益。
    train : 独立小模型 (MLP / 事件内注意力), 直接对比"只看局部" vs "看上下文"。

数据来源: analyze_feature_ceiling.py 产出的 npz (X/y/grp/names)。
用法:
    python3 analyze_vertex_assoc.py probe --npz report_figs/feat_ceiling_ctx.npz
    python3 analyze_vertex_assoc.py train --npz report_figs/feat_ceiling_ctx.npz --model attn
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch          # 顶层导入: _sinkhorn_logprob 在模块级使用

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

RAW_NAMES = ["fspv", "theta", "trdist", "dz0", "logDOCA"]
DER_NAMES = ["dR", "m", "absSumQ", "dpT", "dIP", "rank", "dz", "sup", "rk_aff"]
VRT_NAMES = ["zcpa", "flight", "collin"]
# 注意: analyze_feature_ceiling.py 里生成的是 a_n0..a_n7 / b_n0..b_n7 (n 前缀不能少)
NODE_NAMES = [f"{p}_{n}" for p in ("a", "b") for n in [f"n{i}" for i in range(8)]]
GEO_NAMES = ["doca", "logdoca", "dstart"]
CTX_NAMES = ["rk_s", "gap_s", "ratio_s", "hi_s", "deg_s",
             "rk_t", "gap_t", "ratio_t", "hi_t", "deg_t", "mutual"]

# 特征组 -> 列名 (按 analyze_feature_ceiling.py 的拼装顺序)
GROUPS = {"raw": RAW_NAMES, "der": DER_NAMES, "vrt": VRT_NAMES,
          "node": NODE_NAMES, "geo": GEO_NAMES, "ctx": CTX_NAMES}

# 逐步累加的顺序 (只比较"信息量层次": 局部 -> 局部+几何 -> 再加上下文)
CUMULATIVE = [
    ("node",                                 ["node"]),
    ("node+geo",                             ["node", "geo"]),
    ("node+geo+raw+der",                     ["node", "geo", "raw", "der"]),
    ("node+geo+raw+der+vrt",                 ["node", "geo", "raw", "der", "vrt"]),
    ("node+geo+raw+der+vrt+ctx",             ["node", "geo", "raw", "der", "vrt", "ctx"]),
    ("ctx_only",                             ["ctx"]),
    ("geo_only",                             ["geo"]),
    ("ctx+geo",                              ["ctx", "geo"]),
]


def _multib_mask(npz, grp, min_nb, max_nb=10 ** 9):
    """[2026-10-08] 多 B 专用子集: 返回"真值链数在 [min_nb, max_nb]"的**边掩码**。

    动机: DFEI 的目标场景是"一个事件里有多条 B", 但这类事件在总表里只占少数 (~30%),
    容易被大量单 B 事件淹没 -> 在同一多 B 子集上比较不同优化, 才能看出**哪个对多 B 最有效**。
    多 B = min_nb>=2; 单 B 对照 = min_nb=1 且 max_nb=1。min_nb<=1 且 max_nb 无限时不做限制。
    """
    if min_nb <= 1 and max_nb >= 10 ** 8:
        return np.ones(len(grp), dtype=bool), None
    d = np.load(npz, allow_pickle=True)
    if "nb" not in d:
        raise SystemExit("npz 缺 nb (每事件真值链数): 请用更新后的 analyze_feature_ceiling.py 重新导出")
    nbe = {int(e): int(v) for e, v in zip(d["grp"], d["nb"])}
    ke = {e for e, v in nbe.items() if min_nb <= v <= max_nb}
    return np.isin(grp, list(ke)), (len(ke), len(nbe))


def _thr_at_precision(s, y, target=0.90):
    """[2026-10-09 A2] 使**边精度 >= target** 且召回最大的分数阈值 (分数越大越像真边)。"""
    s = np.asarray(s, float)
    order = np.argsort(-s)
    ys = np.asarray(y, int)[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.nonzero(prec >= target)[0]
    if len(ok) == 0:
        return float("inf")
    return float(s[order][ok[-1]])


def _chain_ids_batch(yb, ib0, ib1, ntb, mb):
    """[2026-10-09 A2] 从 batch 张量算**真值链**分组。

    真值链定义: 该事件内 y==1 的边构成子图的连通分量 (与 bench / setpred 原型一致)。
    ib0/ib1 已逐事件重映射到 0..ntb-1, 所以**每个事件必须独立做并查集**。
    返回 (cid, nchain): cid 形状 (B,T), -1 表示该边不属于任何真链 (或 padding)。
    """
    B, T = mb.shape
    cid = np.full((B, T), -1, dtype=np.int64)
    nch = 0
    for bi in range(B):
        n = max(int(ntb[bi]), 1)
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        idx = np.nonzero(mb[bi])[0]
        for k in idx:
            if yb[bi, k] > 0.5:
                ra, rb = find(int(ib0[bi, k])), find(int(ib1[bi, k]))
                if ra != rb:
                    par[ra] = rb
        seen = {}
        for k in idx:
            if yb[bi, k] > 0.5:
                r = find(int(ib0[bi, k]))
                if r not in seen:
                    seen[r] = nch
                    nch += 1
                cid[bi, k] = seen[r]
    return cid, nch


def _chain_loss(p, cid, nch, pool="softmin", gamma=10.0):
    """[2026-10-09 A2] **链级存活**损失 (AND 语义的可微代理)。

    动机: 物理判据是"整条链必须每个环节都对"(一失毁全链), 而训练一直是逐边 BCE。
    - pool="prod"    : 存活 = ∏ p_e 的几何平均。**注意它≈逐边 BCE**(对数下就是平均),
                       留作对照, 说明"只换成乘积形式"没有新信息。
    - pool="softmin" : 存活 = softmin_γ(p) = -log(mean e^{-γ p})/γ, γ 越大越接近 min(p)。
                       它把梯度集中到链里**最弱的那一环** -> 对"指数级链存活衰减"才是
                       正确的代理。这才是本项要测的东西。
    """
    if nch <= 0:
        return None
    p = p.reshape(-1).clamp(1e-6, 1 - 1e-6)
    flat = torch.as_tensor(cid.reshape(-1), device=p.device)
    sel = flat >= 0
    if not bool(sel.any()):
        return None
    c, pv = flat[sel], p[sel]
    one = torch.ones_like(pv)
    cnt = torch.zeros(nch, device=p.device).scatter_add_(0, c, one).clamp_min(1.0)
    if pool == "prod":
        s = torch.zeros(nch, device=p.device).scatter_add_(0, c, torch.log(pv))
        surv = torch.exp(s / cnt)
    else:
        e = torch.exp(-gamma * pv)
        m = torch.zeros(nch, device=p.device).scatter_add_(0, c, e)
        surv = -torch.log((m / cnt).clamp_min(1e-9)) / gamma
    return -torch.log(surv.clamp_min(1e-6)).mean()


def load_npz(path, want_ids=False):
    d = np.load(path, allow_pickle=True)
    X, y, grp = d["X"], d["y"], d["grp"]
    names = [str(x) for x in d["names"]]
    if want_ids:
        for k in ("t0", "t1", "ntr"):
            if k not in d:
                raise KeyError(f"npz 缺 {k}: 全局匹配/Sinkhorn 需要每条边的两端径迹 id。"
                               f"请用更新后的 analyze_feature_ceiling.py 重新导出 (它会写 t0/t1/ntr)。")
        return X, y, grp, names, d["t0"], d["t1"], d["ntr"]
    return X, y, grp, names


def _sinkhorn_logprob(s, i, j, n, tau, iters, dust):
    """全局匹配 (Sinkhorn): 事件内 track x track 的**软分配** log 概率。

    动机: 逐对分类只看"这一对像不像真边", 而物理约束是**排他** —— 一条径迹只属于一条链。
    把边分数放进 (n+1)x(n+1) 矩阵 (末行/末列是 dustbin, 允许未匹配), 做 log-domain Sinkhorn
    归一, 得到"在全局可分配方案下这条边被选中的 log 概率" -> 让不同分配之间**竞争**,
    这是逐对 softmax 做不到的。整个操作可微, 可端到端训练。

    s: [E] 该事件的边 logit;  i,j: [E] 两端径迹的事件内局部 id (0..n-1);  n: 径迹数
    """
    dev, dt = s.device, s.dtype
    K = torch.full((n + 1, n + 1), -1e4, device=dev, dtype=dt)
    K[i, j] = s / tau
    K[j, i] = s / tau
    ld = math.log(dust)
    K[n, :] = ld
    K[:, n] = ld
    logr = torch.full((n + 1,), -math.log(n + 1), device=dev, dtype=dt)
    u = torch.zeros(n + 1, device=dev, dtype=dt)
    v = torch.zeros(n + 1, device=dev, dtype=dt)
    for _ in range(iters):
        u = logr - torch.logsumexp(K + v.unsqueeze(0), dim=1)
        v = logr - torch.logsumexp(K + u.unsqueeze(1), dim=0)
    return (K + u.unsqueeze(1) + v.unsqueeze(0))[i, j]


def ap_score(score, label):
    """Average Precision (与 analyze_prune_loss.py 同一实现)"""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    if label.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    lab = label[order]
    csum = np.cumsum(lab)
    prec = csum / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / lab.sum())


def prec_at_recall(score, label, r):
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    order = np.argsort(-score, kind="mergesort")
    lab = label[order]
    csum = np.cumsum(lab)
    need = int(np.ceil(r * lab.sum()))
    k = int(np.searchsorted(csum, need)) + 1
    return float(csum[k - 1] / k) if k <= len(lab) else float("nan")


def auc(score, label):
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    pos = label == 1
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(score))
    return float((r[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum()))


def split_events(grp, seed=0, frac=(0.6, 0.2, 0.2)):
    ev = np.unique(grp)
    rng = np.random.default_rng(seed)
    ev = ev[rng.permutation(len(ev))]
    n1 = int(len(ev) * frac[0]); n2 = int(len(ev) * (frac[0] + frac[1]))
    return ev[:n1], ev[n1:n2], ev[n2:]


def cols_for(names, groups):
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for g in groups:
        for n in GROUPS[g]:
            if n not in idx:
                raise KeyError(f"npz 里没有特征 {n} (可用: {names[:8]} ...)")
            out.append(idx[n])
    return out


# ---------------------------------------------------------------- probe
def cmd_probe(a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    X, y, grp, names = load_npz(a.npz)
    _m, _info = _multib_mask(a.npz, grp, a.min_nb, a.max_nb)
    if _info is not None:
        print(f"[probe] 子集 nb in [{a.min_nb},{a.max_nb}]: {int(_m.sum())}/{len(_m)} 条边, "
              f"{_info[0]}/{_info[1]} 个事件 ({100*_info[0]/max(_info[1],1):.1f}%)")
        X, y, grp = X[_m], y[_m], grp[_m]
    tr, va, te = split_events(grp, seed=a.seed)
    m_tr = np.isin(grp, tr); m_te = np.isin(grp, te)
    print(f"[probe] 样本 {X.shape} | 事件 train/val/test = {len(tr)}/{len(va)}/{len(te)} "
          f"| 真边率 {100*y.mean():.1f}% | 测试集边数 {int(m_te.sum())} (真边 {int(y[m_te].sum())})")
    print(f"\n{'特征组':<34}{'维度':>5}{'AUC':>9}{'AP':>9}{'p@r90':>9}")
    res = {}
    for tag, gs in CUMULATIVE:
        c = cols_for(names, gs)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                             max_leaf_nodes=31, random_state=0)
        clf.fit(X[m_tr][:, c], y[m_tr])
        s = clf.predict_proba(X[m_te][:, c])[:, 1]
        r = dict(dim=len(c), auc=auc(s, y[m_te]), ap=ap_score(s, y[m_te]),
                 p90=prec_at_recall(s, y[m_te], 0.90))
        res[tag] = r
        print(f"{tag:<34}{r['dim']:>5}{r['auc']:>9.4f}{r['ap']:>9.4f}{r['p90']:>9.4f}")

    # 上下文的**边际**贡献 = 关键闸门
    base = res.get("node+geo+raw+der+vrt", {})
    full = res.get("node+geo+raw+der+vrt+ctx", {})
    print(f"\n[闸门] 加上 ctx 后:  AP {base.get('ap', float('nan')):.4f} -> {full.get('ap', float('nan')):.4f}"
          f"  (Δ{full.get('ap', float('nan')) - base.get('ap', float('nan')):+.4f})"
          f"   p@r90 {base.get('p90', float('nan')):.4f} -> {full.get('p90', float('nan')):.4f}"
          f"  (Δ{full.get('p90', float('nan')) - base.get('p90', float('nan')):+.4f})")
    print("  判读: ΔAP 明显 >0 -> 上下文里有可用信息, 注意力/匹配值得投入;"
          " ≈0 -> 上限是信息限制, 加结构无益")

    c_all = cols_for(names, ["node", "geo", "raw", "der", "vrt", "ctx"])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         max_leaf_nodes=31, random_state=0)
    clf.fit(X[m_tr][:, c_all], y[m_tr])
    pi = permutation_importance(clf, X[m_te][:, c_all], y[m_te],
                                n_repeats=5, random_state=0, scoring="average_precision")
    order = np.argsort(-pi.importances_mean)[:15]
    print("\n--- permutation importance (AP, 前 15) ---")
    for i in order:
        print(f"  {names[c_all[i]]:<12} {pi.importances_mean[i]:+.4f} ± {pi.importances_std[i]:.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\n[probe] 已写 {a.out}")


# ---------------------------------------------------------------- train
def _attn_class():
    """事件内 token = 候选边, 自注意力让每条边看到同事件其它候选。三个新机制 (本轮主线):
      (1) 物理偏置 bias: 在 attention logit 上加上由**几何列**(doca / |Δ起点|)导出的成对偏置,
          即 Graphormer / AlphaFold 的 pair-bias 思路 —— 让最强的物理量与注意力**直接耦合**,
          而不是让它当第 47 个输入特征。
      (2) 稀疏化 attn_fn: softmax / sparsemax(欧氏投影到单纯形) / topk(硬保留每个 query 的 top-k).
          注意: 这是**模型内可学习的稀疏注意力**, 与旧做法(推理时 edge_topk 截断)完全不同 ——
          旧 topk 已被证明只是"工作点旋钮"。
      (3) 任意层数/头数。
    (必须定义在工厂里返回, 否则 _build_model 拿不到这个名字。)
    """
    import math
    import torch
    import torch.nn as nn

    def _sparsemax(z):
        """Martins & Astudillo 2016: sparsemax(z) = argmin_p ||p - z||^2 s.t. p 在单纯形上。

        [2026-10-08 FIX D2] 原实现用 `(cumsum(z) - 1) > 0` 定支撑集, 判据错误 ->
        输出**根本不是概率分布** (实测: z=[0.5,0.4,0.1] 本身就在单纯形上, 却输出和=2.5;
        z=[0.6,0.6,0.6] 输出和=3.0)。正确支撑大小 = 满足
        `cumsum_k(z) - 1 < k·z_(k)` 的最大 k (等价于 `1 + k·z_(k) > cumsum_k`),
        再取 tau = (cumsum_k - 1)/k, p = relu(z - tau)。
        """
        z = z - z.amax(-1, keepdim=True)
        zs, _ = torch.sort(z, dim=-1, descending=True)
        kk = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
        cum = zs.cumsum(-1)
        # 逐 k 判据; 用累计与保证取到"最大 k"(判据在 k 上单调下降, 但保险起见不直接 sum)
        ok = (cum - 1.0 < kk * zs).to(z.dtype)
        k = ok.cumprod(-1).sum(-1, keepdim=True).clamp(min=1).long()   # gather 需要 int64
        tau = (cum.gather(-1, k - 1) - 1.0) / k.to(cum.dtype)
        return (z - tau).clamp(min=0.0)

    def _topk_norm(sc, k):
        if k <= 0 or k >= sc.size(-1):
            return torch.softmax(sc, -1)
        v, _ = torch.topk(sc, k, dim=-1)
        m = sc >= v[..., -1:]
        return torch.softmax(sc.masked_fill(~m, float("-inf")), -1)

    class AttnBlock(nn.Module):
        def __init__(self, d, heads, dropout, attn_fn, topk, n_bias):
            super().__init__()
            self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.heads, self.attn_fn, self.topk, self.n_bias = heads, attn_fn, topk, n_bias
            self.bias_mlp = (nn.Sequential(nn.Linear(4 * n_bias, 32), nn.ReLU(), nn.Linear(32, heads))
                             if n_bias > 0 else None)
            self.out = nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
            self.drop = nn.Dropout(dropout)

        def forward(self, h, mask, bcols):
            B, T, D = h.shape
            H, dh = self.heads, D // self.heads
            x = self.norm1(h)
            q, k, v = self.qkv(x).view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
            sc = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            if self.bias_mlp is not None and bcols is not None:
                bi = bcols.unsqueeze(2).expand(B, T, T, -1)
                bj = bcols.unsqueeze(1).expand(B, T, T, -1)
                feat = torch.cat([bi, bj, (bi - bj).abs(), bi * bj], -1)
                sc = sc + self.bias_mlp(feat).permute(0, 3, 1, 2)
            kpm = (~mask).view(B, 1, 1, T).expand_as(sc)
            if self.attn_fn == "sparsemax":
                a = _sparsemax(sc.masked_fill(kpm, -1e9))
                a = a * mask.view(B, 1, T, 1).to(a.dtype)
            elif self.attn_fn == "topk":
                a = _topk_norm(sc.masked_fill(kpm, float("-inf")), self.topk)
            else:
                a = torch.softmax(sc.masked_fill(kpm, float("-inf")), -1)
            o = (self.drop(a) @ v).transpose(1, 2).reshape(B, T, D)
            h = h + self.out(o)
            return h + self.ff(self.norm2(h))

    class EvtAttn(nn.Module):
        def __init__(self, n_feat, d, heads, layers, dropout, attn_fn="softmax", topk=0, bias_idx=None):
            super().__init__()
            self.inp = nn.Linear(n_feat, d)
            self.bias_idx = list(bias_idx or [])
            self.blocks = nn.ModuleList([AttnBlock(d, heads, dropout, attn_fn, topk, len(self.bias_idx))
                                         for _ in range(layers)])
            self.out = nn.Linear(d, 1)

        def forward(self, x, mask):
            bcols = x[:, :, self.bias_idx] if self.bias_idx else None
            h = self.inp(x)
            for blk in self.blocks:
                h = blk(h, mask, bcols)
            return self.out(h).squeeze(-1)

    return EvtAttn


def _build_model(kind, n_feat, hidden, heads, layers, dropout,
                 attn_fn="softmax", attn_topk=0, bias_idx=None):
    import torch.nn as nn
    if kind == "mlp":
        return nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden // 2, 1))
    if kind == "attn":
        return _attn_class()(n_feat, hidden, heads, layers, dropout, attn_fn, attn_topk, bias_idx)
    raise ValueError(kind)


def cmd_train(a):
    import torch
    import torch.nn as nn

    if a.match != "none" and a.model != "attn":
        raise SystemExit("--match 需要 --model attn (Sinkhorn 按事件做, 只有 attn 路径有事件分批)")
    need_ids = (a.match == "sinkhorn") or (a.chain_loss_w > 0)

    # [2026-10-03] 之前 --seed 只控制"事件切分"与 batch 打乱, **没控制模型初始化/dropout** ->
    # 同一 seed 重跑结果也不同。这里统一固定 numpy/torch 种子, 保证可复现 (多种子对照的前提)。
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    if need_ids:
        X, y, grp, names, T0, T1, NTR = load_npz(a.npz, want_ids=True)
    else:
        X, y, grp, names = load_npz(a.npz)
    # [2026-10-08] 多 B 专用子集 (见 _multib_mask 说明)
    _m, _info = _multib_mask(a.npz, grp, a.min_nb, a.max_nb)
    if _info is not None:
        print(f"[train] 子集 nb in [{a.min_nb},{a.max_nb}]: {int(_m.sum())}/{len(_m)} 条边, "
              f"{_info[0]}/{_info[1]} 个事件 ({100*_info[0]/max(_info[1],1):.1f}%)")
        X, y, grp = X[_m], y[_m], grp[_m]
        if need_ids:
            T0, T1, NTR = T0[_m], T1[_m], NTR[_m]
    y = y.astype(np.float32)          # npz 里 y 是 int8; BCE 要求与 logits 同 dtype

    gs = a.feats.split("+")
    c = cols_for(names, gs)
    Xf = X[:, c].astype(np.float32)
    tr, va, te = split_events(grp, seed=a.seed)
    mu = Xf[np.isin(grp, tr)].mean(0, keepdims=True)
    sd = Xf[np.isin(grp, tr)].std(0, keepdims=True) + 1e-6
    Xf = (Xf - mu) / sd
    print(f"[train] 模型={a.model} 特征={'+'.join(gs)} ({len(c)} 维) attn_fn={a.attn_fn} "
          f"topk={a.attn_topk} bias={a.attn_bias} match={a.match} seed={a.seed}")

    ev_of = {e: i for i, e in enumerate(np.unique(grp))}
    gidx = np.array([ev_of[e] for e in grp])
    parts = {"tr": np.isin(grp, tr), "va": np.isin(grp, va), "te": np.isin(grp, te)}

    def batches(mask, bs, shuffle, rng=None):
        """统一产出 (xb, yb, mb, ib0, ib1, ntb); MLP 路径下 id 字段为 None。"""
        if a.model == "mlp":
            idx = np.nonzero(mask)[0]
            if shuffle:
                rng.shuffle(idx)
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]
                yield Xf[j], y[j], None, None, None, None
            return
        evs = np.unique(gidx[mask])
        if shuffle:
            rng.shuffle(evs)
        for i in range(0, len(evs), bs):
            sel = evs[i:i + bs]
            idx = np.nonzero(np.isin(gidx, sel))[0]
            sizes = [int((gidx[idx] == e).sum()) for e in sel]
            Tb = max(sizes)
            xb = np.zeros((len(sel), Tb, Xf.shape[1]), dtype=np.float32)
            yb = np.zeros((len(sel), Tb), dtype=np.float32)
            mb = np.zeros((len(sel), Tb), dtype=bool)
            ib0 = np.zeros((len(sel), Tb), dtype=np.int64)
            ib1 = np.zeros((len(sel), Tb), dtype=np.int64)
            ntb = np.zeros(len(sel), dtype=np.int64)
            p = 0
            for bi, sz in enumerate(sizes):
                sl = idx[p:p + sz]
                xb[bi, :sz] = Xf[sl]
                yb[bi, :sz] = y[sl]
                mb[bi, :sz] = True
                if need_ids:
                    # 把事件内局部 track id 重映射到 0..n-1 (不同事件可能重号, 必须逐事件重映射)
                    ids = np.unique(np.concatenate([T0[sl], T1[sl]]))
                    rmap = {int(v): k for k, v in enumerate(ids)}
                    ib0[bi, :sz] = [rmap[int(v)] for v in T0[sl]]
                    ib1[bi, :sz] = [rmap[int(v)] for v in T1[sl]]
                    ntb[bi] = len(ids)
                p += sz
            yield xb, yb, mb, ib0, ib1, ntb

    bias_idx = []
    _bn = {"geom": ("doca", "dstart"),
           "geom4": ("doca", "dstart", "dz0", "logDOCA")}.get(a.attn_bias, ())
    if a.model == "attn" and _bn:
        for nm in _bn:
            if nm in names and names.index(nm) in c:
                bias_idx.append(c.index(names.index(nm)))
        if not bias_idx:
            print("[train] WARN: 特征里没有几何列, 几何 bias 退化为 none")
        elif len(bias_idx) < len(_bn):
            # [2026-10-08 FIX D3] 原来只在"一个都没匹配"时才告警 -> 部分匹配 (如 geom4 的
            #   dz0/logDOCA 不在 node+geo+ctx 里) 会**静默退化**成列更少的那一档 (历史踩过:
            #   --attn_bias geom4 静默等价于 geom, 单变量对照因此失效)。
            _miss = [n for n in _bn if not (n in names and names.index(n) in c)]
            print(f"[train] WARN: --attn_bias {a.attn_bias} 只匹配到 {len(bias_idx)}/{len(_bn)} 列 "
                  f"(缺 {_miss}) -> 实际等价于列更少的那一档, 不是你以为的配置!")
    net = _build_model(a.model, Xf.shape[1], a.hidden, a.heads, a.layers, a.dropout,
                       a.attn_fn, a.attn_topk, bias_idx)

    # [2026-10-02 FIX] 原来先建 BCEWithLogitsLoss(pos_weight=pw) 再 `pw = pw.to(dev)` ——
    #   .to() 返回新张量, 模块仍持有 CPU 上那个 -> GPU 报 "two devices" (CPU 跑不暴露)。
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pw = torch.tensor([(y[parts["tr"]] == 0).sum() / max(1, (y[parts["tr"]] == 1).sum())],
                      dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    net.to(dev)

    def match_scores(logits, mb, ib0, ib1, ntb):
        """给定 per-token logits, 返回 Sinkhorn 匹配分数 [B,T] (只在该事件有效位置上填值)"""
        B = logits.shape[0]
        out = torch.zeros_like(logits)
        for bi in range(B):
            m_ = mb[bi]
            if int(m_.sum()) == 0:
                continue
            ii = ib0[bi][m_]
            jj = ib1[bi][m_]
            lp = _sinkhorn_logprob(logits[bi][m_], ii, jj, int(ntb[bi]),
                                   a.match_tau, a.match_iters, a.match_dust)
            out[bi][m_] = (lp - lp.mean()) / (lp.std() + 1e-6)      # 归一化, 便于与 logits 同量级
        return out

    def forward_scores(xt, mb):
        if a.model == "attn":
            lg = net(xt, mb)
            return lg
        return net(xt).squeeze(-1)

    def collect(mask, bs=0):
        """返回 dict: pairwise / match / comb 三种分数 + 标签"""
        bs = bs or a.bs
        S, SM, Y = [], [], []
        CH = []      # [2026-10-09 A2] 每条真链的"最弱环"分数
        for xb, yb, mb, ib0, ib1, ntb in batches(mask, bs, False):
            xt = torch.as_tensor(xb, device=dev)
            lg = forward_scores(xt, None if mb is None else torch.as_tensor(mb, device=dev))
            if a.model == "attn":
                m = torch.as_tensor(mb, device=dev)
                S.append(lg[m].detach().cpu().numpy())
                Y.append(torch.as_tensor(yb, device=dev)[m].cpu().numpy())
                if need_ids:
                    sm = match_scores(lg, mb, ib0, ib1, ntb)
                    SM.append(sm[m].detach().cpu().numpy())
                    _cb, _nch = _chain_ids_batch(yb, ib0, ib1, ntb, mb)
                    if _nch:
                        _lgn = lg.detach().cpu().numpy().reshape(-1)
                        _cf = _cb.reshape(-1)
                        _sel = _cf >= 0
                        if _sel.any():
                            _cm = np.full(_nch, np.inf)
                            np.minimum.at(_cm, _cf[_sel], _lgn[_sel])
                            CH.append(_cm)
            else:
                S.append(lg.reshape(-1).detach().cpu().numpy())
                Y.append(yb.reshape(-1))
        out = {"pairwise": np.concatenate(S), "y": np.concatenate(Y).astype(int)}
        if CH:
            out["chain_min"] = np.concatenate(CH)
        if SM and a.match != "none":      # 只在真开了 match 时才输出, 否则会误标成 "+Sinkhorn"
            out["match"] = np.concatenate(SM)
            out["comb"] = out["pairwise"] + out["match"]
        return out

    best, best_ep, bad, best_state = -1.0, -1, 0, None
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        net.train()
        for xb, yb, mb, ib0, ib1, ntb in batches(parts["tr"], a.bs, True, rng):
            xt = torch.as_tensor(xb, device=dev)
            yt = torch.as_tensor(yb, device=dev)
            lg = forward_scores(xt, None if mb is None else torch.as_tensor(mb, device=dev))
            if a.model == "attn":
                m = torch.as_tensor(mb, device=dev)
                loss = lossf(lg[m], yt[m])
                if need_ids:
                    sm = match_scores(lg, mb, ib0, ib1, ntb)
                    loss = loss + a.match_w * lossf(sm[m], yt[m])
            else:
                loss = lossf(lg, yt)
            # [2026-10-09 A2] 链级存活损失 (对真链的"最弱环"施加额外惩罚)
            if a.chain_loss_w > 0:
                _cb, _nch = _chain_ids_batch(yb, ib0, ib1, ntb, mb)
                _cl = _chain_loss(torch.sigmoid(lg), _cb, _nch, a.chain_pool, a.chain_gamma)
                if _cl is not None:
                    loss = loss + a.chain_loss_w * _cl
            opt.zero_grad()
            loss.backward()
            opt.step()
        r = collect(parts["va"])
        ap = ap_score(r["pairwise"], r["y"])
        if ap > best + 1e-4:
            best, best_ep, bad = ap, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f"  ep{ep:>3} val_AP={ap:.4f} (best {best:.4f}@{best_ep})", flush=True)
        if bad >= a.patience:
            print(f"  early stop @ep{ep} (best {best:.4f}@{best_ep})")
            break
    net.load_state_dict(best_state)
    net.eval()
    r = collect(parts["te"])
    # [2026-10-09 A2] 链级指标: 用"边精度 90%"的工作点看**整链存活**(AND 语义)
    #   为什么必须补这个: 只报边级 AP 会漏掉"整链一失毁全链"的物理判据, 而 A2 的损失
    #   正是针对它设计的; 没有链级指标就无法判断 A2 到底有没有用。
    _cmsg = ""
    if "chain_min" in r:
        _thr = _thr_at_precision(r["pairwise"], r["y"], 0.90)
        _cs = 100.0 * float(np.mean(r["chain_min"] > _thr))
        _cmsg = f" | 链存活@边精度90%={_cs:.1f}% (阈={_thr:+.3f}, 真链数={len(r['chain_min'])})"
    msg = (f"[train] 测试集: pairwise AUC={auc(r['pairwise'], r['y']):.4f} AP={ap_score(r['pairwise'], r['y']):.4f} "
           f"p@r90={prec_at_recall(r['pairwise'], r['y'], 0.90):.4f}")
    if "match" in r:
        msg += (f" || +Sinkhorn: AP={ap_score(r['match'], r['y']):.4f} "                f"p@r90={prec_at_recall(r['match'], r['y'], 0.90):.4f}"
                f" || 相加: AP={ap_score(r['comb'], r['y']):.4f} "
                f"p@r90={prec_at_recall(r['comb'], r['y'], 0.90):.4f}")
    print(msg + _cmsg + f"   (真边率 {100*r['y'].mean():.1f}%, n={len(r['y'])})")
    if a.out:
        np.savez_compressed(a.out, pairwise=r["pairwise"], y=r["y"],
                            **({"match": r["match"], "comb": r["comb"]} if "match" in r else {}))
        print(f"[train] 已写 {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("probe")
    p2 = sub.add_parser("train")
    for p in (p1, p2):
        p.add_argument("--npz", default="report_figs/feat_ceiling_ctx.npz")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default="")
        # [2026-10-08] 多 B 专用: 只保留每事件真值链数 >= N 的事件 (DFEI 目标场景)
        p.add_argument("--min_nb", type=int, default=0,
                       help="只保留每事件真值链数 >= N 的事件 (多 B 用 2; 0/1 = 不设下界)")
        p.add_argument("--max_nb", type=int, default=10 ** 9,
                       help="每事件真值链数上界 (单 B 对照用 --min_nb 1 --max_nb 1)")
    p2.add_argument("--model", default="mlp", choices=["mlp", "attn"])
    p2.add_argument("--attn_fn", default="softmax", choices=["softmax", "sparsemax", "topk"])
    p2.add_argument("--attn_topk", type=int, default=0)
    p2.add_argument("--attn_bias", default="none", choices=["none", "geom", "geom4"])
    # [2026-10-03] 全局匹配 (排他约束): Sinkhorn 软分配
    p2.add_argument("--match", default="none", choices=["none", "sinkhorn"])
    p2.add_argument("--match_tau", type=float, default=0.5)
    p2.add_argument("--match_iters", type=int, default=10)
    p2.add_argument("--match_dust", type=float, default=0.1)
    p2.add_argument("--match_w", type=float, default=0.5)
    # [2026-10-09 A2] 链级存活损失: 物理判据是"整链每个环节都要对"(AND), 训练却是逐边 BCE
    p2.add_argument("--chain_loss_w", type=float, default=0.0,
                    help="链级存活损失权重 (0=关; 需要事件内 t0/t1, 会自动打开 ids 加载)")
    p2.add_argument("--chain_pool", default="softmin", choices=["softmin", "prod"],
                    help="softmin=集中罚链内最弱环(推荐); prod=几何平均(≈逐边 BCE 的对照)")
    p2.add_argument("--chain_gamma", type=float, default=10.0,
                    help="softmin 的温度: 越大越接近 min(p)")
    p2.add_argument("--feats", default="node+geo+ctx")
    p2.add_argument("--hidden", type=int, default=256)
    p2.add_argument("--heads", type=int, default=4)
    p2.add_argument("--layers", type=int, default=2)
    p2.add_argument("--dropout", type=float, default=0.1)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--bs", type=int, default=16)
    p2.add_argument("--epochs", type=int, default=60)
    p2.add_argument("--patience", type=int, default=12)
    a = ap.parse_args()
    {"probe": cmd_probe, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
