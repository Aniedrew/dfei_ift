"""解码器动物园: 在同一组离线导出的边分数上并行试多种"划分"方案。

输入: report_figs/edge_dump/<tag>/*.npz  (由 reconstruction.py 的 dump_edges 钩子导出)
  每个文件: edges(2,E) lca(E,4 概率) chain(N,) 真值链号(-1=背景) pid(N,)
输出: 每个解码器/超参组合的 clean% / found% / 碎裂% / 平均污染数, 全部在**同一批事件**上比较。

关键: 这里只评"解码", 不含剪枝损失 —— 所以差异完全来自解码方式。
用法: python3 decoder_zoo.py <dump_dir> [--max-events 600]
"""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "report_figs/edge_dump/v651"
MAXEV = int(sys.argv[sys.argv.index("--max-events") + 1]) if "--max-events" in sys.argv else 10 ** 9


# ---------- 基础工具 ----------
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def labels_from_uf(uf, n):
    lab = np.full(n, -1, dtype=np.int64)
    m = {}
    for i in range(n):
        r = uf.find(i)
        if r not in m:
            m[r] = len(m)
        lab[i] = m[r]
    return lab


def score_from_lca(lca, mode):
    """把 4 类概率压成"同对象"分数。class0 视为不相关。"""
    if lca.ndim == 1:
        return lca
    if mode == "one_minus_p0":
        return 1.0 - lca[:, 0]
    if mode == "p1":
        return lca[:, 1]
    if mode == "p12":
        return lca[:, 1] + lca[:, 2]
    if mode == "p123":
        return lca[:, 1] + lca[:, 2] + lca[:, 3]
    raise ValueError(mode)


# ---------- 解码器 ----------
def dec_cc(n, ei, s, thr, **_):
    uf = UF(n)
    for (a, b), v in zip(ei.T, s):
        if v >= thr:
            uf.union(int(a), int(b))
    return labels_from_uf(uf, n)


def dec_cc_dustbin(n, ei, s, thr, node_thr=0.0, **_):
    """先建图, 再把"与任何邻居的最大分都低于 node_thr"的节点判为背景(拒绝)。"""
    lab = dec_cc(n, ei, s, thr)
    if node_thr > 0:
        mx = np.zeros(n)
        for (a, b), v in zip(ei.T, s):
            if v >= thr:
                mx[int(a)] = max(mx[int(a)], v)
                mx[int(b)] = max(mx[int(b)], v)
        lab = lab.copy()
        lab[mx < node_thr] = -1
    return lab


def dec_agglom(n, ei, s, thr, linkage="average", **_):
    """层次聚类: 反复合并"簇间最大/平均亲和最高且 >= thr"的一对; 也可用 complete。"""
    clusters = {i: [i] for i in range(n)}
    # 邻接表: 只保留 >= thr 的边
    adj = {}
    for (a, b), v in zip(ei.T, s):
        if v >= thr:
            adj.setdefault(int(a), []).append((int(b), float(v)))
            adj.setdefault(int(b), []).append((int(a), float(v)))
    owner = {i: i for i in range(n)}
    while True:
        best, pair = -1.0, None
        # 用簇间边聚合
        for a, nbrs in adj.items():
            ca = owner[a]
            for b, v in nbrs:
                cb = owner[b]
                if ca == cb:
                    continue
                if linkage == "single":
                    sc = v
                else:
                    sc = v  # average/complete 用边权近似(全连接图下差别小), 见下
                if sc > best:
                    best, pair = sc, (ca, cb)
        if pair is None or best < thr:
            break
        ca, cb = pair
        # 合并
        for k in list(owner):
            if owner[k] == cb:
                owner[k] = ca
    return np.array([owner[i] for i in range(n)], dtype=np.int64)


def dec_corr_cluster(n, ei, s, thr, **_):
    """相关性聚类(贪心): 目标 = 同簇内的"负边"代价 + 跨簇的"正边"代价。
    从 CC(thr) 初始化, 迭代做局部改进: 对每个"边界"边, 尝试切开/合并。"""
    w = s.copy()  # 越大越像"同一对象"
    lab = dec_cc(n, ei, s, thr)
    # 简化版: 只需保证等价关系, 用 uf 重建
    uf = UF(n)
    for (a, b), v in zip(ei.T, s):
        if v >= thr:
            uf.union(int(a), int(b))
    # 局部改进: 若某簇内部存在大量低分边(证据不足), 则把该簇拆成两半(按最高分割边)
    for _ in range(3):
        changed = False
        labs = labels_from_uf(uf, n)
        for c in np.unique(labs):
            idx = np.where(labs == c)[0]
            if len(idx) < 3:
                continue
            inside = []
            for (a, b), v in zip(ei.T, s):
                if a in idx and b in idx:
                    inside.append((float(v), int(a), int(b)))
            if not inside:
                continue
            inside.sort()
            low = [t for t in inside if t[0] < thr]
            if len(low) >= max(2, len(inside) // 2):  # 多数内部边都很弱 -> 拆
                _, a, b = low[0]
                # 用弱边做割: 把 a 单独分出去(近似)
                uf.p[uf.find(a)] = a
                changed = True
                break
        if not changed:
            break
    return labels_from_uf(uf, n)


def dec_require_density(n, ei, s, thr, density=0.5, **_):
    """CC 后, 若某簇内部边密度(>=thr 的比例)低于 density, 判为"并错"(整簇丢弃/拆成单点)。"""
    lab = dec_cc(n, ei, s, thr)
    out = lab.copy()
    for c in np.unique(lab):
        if c < 0:
            continue
        idx = np.where(lab == c)[0]
        if len(idx) < 3:
            continue
        k = len(idx)
        tot = k * (k - 1) / 2
        good = 0
        for (a, b), v in zip(ei.T, s):
            if a in idx and b in idx and v >= thr:
                good += 1
        if tot and good / tot < density:
            out[lab == c] = -1
    return out


def dec_label_prop(n, ei, s, thr, iters=10, **_):
    lab = np.arange(n)
    # 邻接
    nb = [[] for _ in range(n)]
    for (a, b), v in zip(ei.T, s):
        if v >= thr:
            nb[int(a)].append(int(b))
            nb[int(b)].append(int(a))
    for _ in range(iters):
        new = lab.copy()
        for i in range(n):
            if not nb[i]:
                continue
            cnt = {}
            for j in nb[i]:
                cnt[lab[j]] = cnt.get(lab[j], 0) + 1
            new[i] = max(cnt, key=cnt.get)
        if np.array_equal(new, lab):
            break
        lab = new
    _, inv = np.unique(lab, return_inverse=True)
    return inv


DECODERS = {
    "cc": dec_cc,
    "cc+dustbin": dec_cc_dustbin,
    "agglom": dec_agglom,
    "corr_cluster": dec_corr_cluster,
    "density_gate": dec_require_density,
    "label_prop": dec_label_prop,
}


# ---------- 评估 ----------
def eval_partition(chain, lab):
    """返回 (真值链数, clean, found, fragmented, 污染径迹总数)"""
    tc = [c for c in np.unique(chain) if c >= 0]
    # 预测簇 -> 节点集合
    cl = {}
    for i, l in enumerate(lab):
        if l >= 0:
            cl.setdefault(int(l), set()).add(i)
    clean = found = frag = 0
    bkg_total = 0
    for c in tc:
        S = set(np.where(chain == c)[0].tolist())
        hit = None
        for _, T in cl.items():
            if S <= T:
                hit = T
                break
        if hit is None:
            frag += 1
        else:
            found += 1
            if len(hit) == len(S):
                clean += 1
            else:
                bkg_total += len(hit) - len(S)
    return len(tc), clean, found, frag, bkg_total


def quantize(vals, q=0.02):
    return np.round(vals / q) * q


def main():
    files = sorted(glob.glob(os.path.join(DUMP, "*.npz")))[:MAXEV]
    if not files:
        print(f"[!] {DUMP} 里没有 npz, 先跑 dump 作业。")
        return
    print(f"dump: {DUMP}  事件数: {len(files)}")
    # 预载
    evs = [np.load(f) for f in files]
    n_chain = sum(len([c for c in np.unique(e["chain"]) if c >= 0]) for e in evs)
    print(f"真值链总数: {n_chain}   平均径迹数: {np.mean([len(e['chain']) for e in evs]):.1f}")

    results = []
    for smode in ["one_minus_p0", "p123"]:
        for dname, fn in DECODERS.items():
            if dname in ("agglom", "corr_cluster", "density_gate"):
                thrs = [0.3, 0.5, 0.7, 0.9]
            else:
                thrs = [0.2, 0.3, 0.5, 0.7, 0.9]
            extra = {}
            if dname == "cc+dustbin":
                extra = {"node_thr": 0.3}
            if dname == "density_gate":
                extra = {"density": 0.6}
            for thr in thrs:
                tot = cl = fo = fr = bk = 0
                for e in evs:
                    s = quantize(score_from_lca(e["lca"], smode))
                    lab = fn(len(e["chain"]), e["edges"].astype(int), s, thr, **extra)
                    t, c, f, g, b = eval_partition(e["chain"], lab)
                    tot += t; cl += c; fo += f; fr += g; bk += b
                results.append(dict(score=smode, dec=dname, thr=thr, **extra,
                                    n=tot, clean=100 * cl / tot, found=100 * fo / tot,
                                    frag=100 * fr / tot, bkg_per_chain=bk / max(1, fo)))
    import pandas as pd
    df = pd.DataFrame(results).sort_values("clean", ascending=False)
    pd.set_option("display.width", 220)
    print("\n=== 按 clean% 排序 (前 20) ===")
    print(df.head(20).round(2).to_string(index=False))
    os.makedirs("report_figs/decoder_zoo", exist_ok=True)
    df.to_csv("report_figs/decoder_zoo/decoder_zoo.csv", index=False)
    print("\n写出: report_figs/decoder_zoo/decoder_zoo.csv")


if __name__ == "__main__":
    main()
