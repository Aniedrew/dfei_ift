"""离线解码器对比 (CPU): 在同一批导出的边分数上比较
    cc      : 连通分量 (当前基线)
    forest  : 最大生成森林 (Kruskal, 无环)
    grow    : 从种子生长的树 (每条链从最强边开始, 逐步吸收最优邻居)
    oracle@k: 把预测分量按链内平均分数排序, 看真值链是否落在前 k 名

输入: report_figs/edge_dump/<tag>/*.npz, 每个文件含
      edges(2,E) int32 / lca(E,4) float32 概率 / chain(N,) int32 真值链号(-1=背景) / pid(N,)
用法: python3 analyze_odo_decoders.py report_figs/edge_dump/v651 [--thr 0.95] [--max-events 800]

指标口径与主流程一致: 只有"节点集合与某条真值链完全相同"才算 exact;
另给出 found (真值链被某个分量**完整覆盖**但带杂质) 与 frag (真值链被拆成多个分量)。
"""
import glob
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "report_figs/edge_dump/v651"
THR = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.95
MAXEV = int(sys.argv[sys.argv.index("--max-events") + 1]) if "--max-events" in sys.argv else 10 ** 9


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
            return True
        return False


def comps(lab, n):
    out = {}
    for i in range(n):
        if lab[i] < 0:
            continue
        out.setdefault(int(lab[i]), []).append(i)
    return out


def dec_cc(n, ei, s, thr):
    uf = UF(n)
    for (a, b), v in zip(ei.T, s):
        if v >= thr:
            uf.union(int(a), int(b))
    return np.array([uf.find(i) for i in range(n)], dtype=np.int64)


def dec_forest(n, ei, s, thr):
    """Kruskal 最大生成森林: 按分数降序加边, 成环即丢 -> 严格无环。"""
    uf = UF(n)
    lab = np.full(n, -1, dtype=np.int64)
    order = np.argsort(-s)
    for k in order:
        if s[k] < thr:
            break
        uf.union(int(ei[0, k]), int(ei[1, k]))
    return np.array([uf.find(i) for i in range(n)], dtype=np.int64)


def dec_grow(n, ei, s, thr):
    """从最强边开始生长: 每条链维护一个节点集合, 反复吸收"与该集合相邻且分数最高"的节点。

    与 forest 的区别: forest 只保证无环, 不限制"一条链能否吸收一个已经属于别的链的节点";
    grow 显式检查目标节点是否还没被占用 —— 这是一条**排他**约束, 更接近"一条径迹只属于一条链"。
    """
    order = np.argsort(-s)
    adj = [[] for _ in range(n)]
    for k in order:
        if s[k] < thr:
            break
        a, b = int(ei[0, k]), int(ei[1, k])
        adj[a].append((float(s[k]), b))
        adj[b].append((float(s[k]), a))
    owner = np.full(n, -1, dtype=np.int64)
    cid = 0
    for k in order:
        if s[k] < thr:
            break
        a, b = int(ei[0, k]), int(ei[1, k])
        if owner[a] < 0 and owner[b] < 0:
            owner[a] = owner[b] = cid
            seed = cid
            cid += 1
        elif owner[a] < 0 or owner[b] < 0:
            seed = owner[a] if owner[a] >= 0 else owner[b]
            owner[a] = owner[b] = seed
        else:
            continue
        # 从种子出发, 用最大堆式贪心吸收邻居
        frontier = sorted(adj[a] + adj[b], key=lambda t: -t[0])
        while frontier:
            w, v = frontier.pop(0)
            if w < thr:
                break
            if owner[v] >= 0:
                continue
            owner[v] = seed
            frontier = sorted(frontier + adj[v], key=lambda t: -t[0])
    return owner


def score_components(lab, ei, s):
    """每个分量的平均内部分数, 用于 oracle@k 的排序。"""
    out = {}
    for (a, b), v in zip(ei.T, s):
        ca, cb = lab[int(a)], lab[int(b)]
        if ca == cb and ca >= 0:
            out.setdefault(int(ca), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in out.items()}


def evaluate(files, thr):
    stats = {name: dict(n_truth=0, exact=0, found=0, frag=0, clean=0)
             for name in ("cc", "forest", "grow")}
    ork = {k: dict(n_truth=0, hit=0) for k in (1, 2, 4, 8)}
    for p in files:
        try:
            z = np.load(p)
            ei, lca, ch = z["edges"].astype(np.int64), z["lca"].astype(np.float64), z["chain"]
        except Exception:
            continue          # 导出时多线程写同一文件名会造成截断, 跳过坏文件
        n = int(ch.shape[0])
        if ei.shape[1] == 0:
            continue
        s = 1.0 - lca[:, 0]
        truths = {int(c): np.nonzero(ch == c)[0] for c in np.unique(ch) if int(c) >= 0}
        for name, fn in (("cc", dec_cc), ("forest", dec_forest), ("grow", dec_grow)):
            lab = fn(n, ei, s, thr)
            cs = comps(lab, n)
            st = stats[name]
            for c, nodes in truths.items():
                st["n_truth"] += 1
                ts = set(nodes.tolist())
                hit_exact, hit_found, hit_frag = False, False, 0
                for _, mem in cs.items():
                    ms = set(mem)
                    if ms == ts:
                        hit_exact = True
                    if ts <= ms:
                        hit_found = True
                    if ms & ts:
                        hit_frag += 1
                st["exact"] += int(hit_exact)
                st["found"] += int(hit_exact or (hit_found and not hit_exact))
                st["frag"] += int(hit_frag > 1)
                st["clean"] += int(not any(ts <= set(m) and set(m) != ts for m in cs.values()))
        # oracle@k: 用 cc 分量按平均分排序
        lab = dec_cc(n, ei, s, thr)
        cs = comps(lab, n)
        sc = score_components(lab, ei, s)
        ranked = sorted((c for c in cs if len(cs[c]) >= 2), key=lambda c: -sc.get(c, 0.0))
        for k in ork:
            top = set(ranked[:k])
            for c in truths:
                ork[k]["n_truth"] += 1
                if any(set(cs[t]) == set(truths[c].tolist()) for t in top):
                    ork[k]["hit"] += 1
    return stats, ork


def main():
    files = sorted(glob.glob(os.path.join(DUMP, "*.npz")))[:MAXEV]
    if not files:
        print(f"没有找到 dump: {DUMP}/*.npz")
        return
    stats, ork = evaluate(files, THR)
    print(f"dump = {DUMP}   events = {len(files)}   thr = {THR}")
    print(f"{'decoder':8s} {'exact%':>8s} {'found%':>8s} {'frag%':>7s} {'clean%':>7s}")
    for name, st in stats.items():
        nt = max(st["n_truth"], 1)
        print(f"{name:8s} {100*st['exact']/nt:8.2f} {100*st['found']/nt:8.2f} "
              f"{100*st['frag']/nt:7.2f} {100*st['clean']/nt:7.2f}")
    print()
    print("oracle@k (cc 分量按链内平均分排序)")
    for k, v in ork.items():
        nt = max(v["n_truth"], 1)
        print(f"  top-{k:<2d}  {100*v['hit']/nt:6.2f}%")


if __name__ == "__main__":
    main()
