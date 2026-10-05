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
        """Martins & Astudillo 2016: sparsemax(z) = argmin_p ||p - z||^2 s.t. p 在单纯形上"""
        z = z - z.amax(-1, keepdim=True)
        zs, _ = torch.sort(z, dim=-1, descending=True)
        kk = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
        cum = zs.cumsum(-1) - 1.0
        k = (cum > 0).sum(-1, keepdim=True).clamp(min=1)
        tau = cum.gather(-1, (k - 1)).div(k)
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
    need_ids = (a.match == "sinkhorn")

    # [2026-10-03] 之前 --seed 只控制"事件切分"与 batch 打乱, **没控制模型初始化/dropout** ->
    # 同一 seed 重跑结果也不同。这里统一固定 numpy/torch 种子, 保证可复现 (多种子对照的前提)。
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    if need_ids:
        X, y, grp, names, T0, T1, NTR = load_npz(a.npz, want_ids=True)
    else:
        X, y, grp, names = load_npz(a.npz)
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
            else:
                S.append(lg.reshape(-1).detach().cpu().numpy())
                Y.append(yb.reshape(-1))
        out = {"pairwise": np.concatenate(S), "y": np.concatenate(Y).astype(int)}
        if SM:
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
    msg = (f"[train] 测试集: pairwise AUC={auc(r['pairwise'], r['y']):.4f} AP={ap_score(r['pairwise'], r['y']):.4f} "
           f"p@r90={prec_at_recall(r['pairwise'], r['y'], 0.90):.4f}")
    if "match" in r:
        msg += (f" || +Sinkhorn: AP={ap_score(r['match'], r['y']):.4f} "
                f"p@r90={prec_at_recall(r['match'], r['y'], 0.90):.4f}"
                f" || 相加: AP={ap_score(r['comb'], r['y']):.4f} "
                f"p@r90={prec_at_recall(r['comb'], r['y'], 0.90):.4f}")
    print(msg + f"   (真边率 {100*r['y'].mean():.1f}%, n={len(r['y'])})")
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
