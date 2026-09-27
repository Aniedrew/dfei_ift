"""逐事件阈值 oracle: 按事件复杂度分组, 量出"事件自适应阈值"相对"全局单一阈值"的成功率上界。

复用 thr_scan_metric.py 的逐事件分数收集与判据 (nw / ew / ft + truth_clusters / reco_clusters /
judge), **不修改任何已有文件**。对每个事件在阈值网格上重新剪枝+解码, 得到

    A_e(thr) = 该事件在 thr 下"成功重建 (AllParticles)"的真值链数
    P_e(thr) = 同口径的 PerfectReco 链数

事件的真值链数 K_e = 在**完整(未剪枝)图**上真值解码出的链数, 与 thr_scan_metric 里固定分母
N_total 的计数口径完全一致 (N_total = sum_e K_e)。

分组: (1) 按 K_e (1 / 2 / >=3); (2) 按径迹数三分位; (3) 按 PV 数三分位 (PV 数 = 图里 pvs 节点数)。

输出三件事:
  (a) 每组"全局单一最优 thr"下的成功率 vs "该组自己的最优 thr"下的成功率;
  (b) oracle 上界: 每个事件都取"所属组最优 thr"时的全样本 All_fix, 与全局单一 thr 对比 (增益 pp);
  (c) 每组的最优 thr。

结果写入 report_figs/per_event_thr_oracle.csv (含分组 / thr / 成功率 / 样本数)。

用法:
  python per_event_thr_oracle.py --config config_files/eval_v601_inc51_0904.yaml \
      --version 601 --max-events 300 --tag v601_0904
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation          # noqa: E402
from wmpgnn.analysis.load_module import load_module                            # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader                 # noqa: E402
from wmpgnn.data_loader.weights_calculator import transform_pos_weight         # noqa: E402
from wmpgnn.util.pruners import true_node_pruning, edge_pruning                # noqa: E402
from thr_scan_metric import truth_clusters, reco_clusters, judge               # noqa: E402

TT = ('tracks', 'to', 'tracks')
GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)] + [0.995]                # 0.50..0.99 + 0.995


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--version", type=int, default=601)
    p.add_argument("--max-events", type=int, default=300)
    p.add_argument("--tag", default="v601_0904")
    p.add_argument("--out", default="report_figs")
    return p.parse_args()


# ---------------------------------------------------------------- 逐事件分数收集
def collect_events(args):
    """与 thr_scan_metric.py 相同的收集逻辑: 一次前向, 逐事件切出 nw / ew / ft / y 与图。

    唯一修正: LCA 走 op_trafo 输出 `out[("tracks","to","tracks")].edges` (4 类 logits),
    与官方 shared_step / thr_scan_official.py / oracle_decompose.py 一致。
    (thr_scan_metric.py 里取的是 batch 上的边表征, 在当前模型里是 16 维 decoder 输出而非 4 类 logits。)
    """
    cfg = yaml.safe_load(open(args.config))
    cfg["settings"]["model"] = args.version
    cfg = adjust_config_evaluation(cfg)
    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    _, _, chunkloader = load_tst_loader(cfg)

    evts = []
    n_evt = 0
    for batch in chunkloader.test_dataloader():
        batch = batch.to(dev)
        tt_ei_in = batch[TT].edge_index.detach().cpu().clone()
        tb = batch["tracks"].batch.cpu().clone()
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            out = module.model(batch)
        blk = module.model._blocks[-1]
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1)
        ew = blk.edge_weights[TT].detach().cpu().squeeze(-1)
        lca_logits = out[TT].edges.detach().cpu()            # op_trafo 输出的 4 类 logits
        graphs = out.to_data_list()
        ng = int(tb.max().item()) + 1 if tb.numel() else 0
        for gid in range(ng):
            if n_evt >= args.max_events:
                break
            n_evt += 1
            tm = tb == gid
            em = tm[tt_ei_in[0]] & tm[tt_ei_in[1]]
            g = graphs[gid].clone()
            g[TT].lca = lca_logits[em].clone()
            evts.append(dict(nw=nw[tm].clone(), ew=ew[em].clone(),
                             ft=g["tracks"].ft.numpy().astype(int),
                             y=g[TT].y.numpy(), g=g))
        if n_evt >= args.max_events:
            break
    return module, evts


def scan_event(d, grid):
    """对单个事件在阈值网格上剪枝+解码。

    返回三条曲线: A_e (成功重建链数) / P_e (完美重建链数) / S_e (剪枝后存活真值链数)。
    """
    n_t = len(grid)
    a_arr = np.zeros(n_t, dtype=int)
    p_arr = np.zeros(n_t, dtype=int)
    s_arr = np.zeros(n_t, dtype=int)
    nw, ew = d["nw"], d["ew"]
    for ti, thr in enumerate(grid):
        g = d["g"].clone()
        node_selbool = nw > thr
        edge_mask = true_node_pruning(node_selbool, g, "tracks", [TT])
        edge_selbool = ew[edge_mask] > thr
        edge_pruning(edge_selbool, g, TT)
        tc = truth_clusters(g)
        rc = reco_clusters(g)
        a, p, _, _, _ = judge(tc, rc)
        a_arr[ti] = a
        p_arr[ti] = p
        s_arr[ti] = len(tc)
    return a_arr, p_arr, s_arr


def tercile_split(vals):
    q1, q2 = np.percentile(vals, [100.0 / 3, 200.0 / 3])
    labels = np.where(vals <= q1, 0, np.where(vals <= q2, 1, 2))
    return labels, q1, q2


def best_ti_smooth(gr, win=3):
    """3 点滑动平均后再取 argmax —— 抑制极端剪枝处的单点尖刺 (平凡坍塌伪影)。"""
    pad = win // 2
    gs = np.convolve(np.pad(gr, pad, mode="edge"), np.ones(win) / win, mode="valid")
    return int(np.argmax(gs))


def build_groups(K, kft, ntr, npv):
    """返回 {分组方案: [(组名, 事件索引数组), ...]}, 每种方案都覆盖全部事件。

    K_ft    : 事件内的 B 链数 = ft 里非背景 (ft!=1) 的 distinct 取值个数 (0=b 2=bbar);
    K_chain : 事件的真值解码链数 = 完整图上 truth_clusters 的条数 (固定分母口径的链)。
    """
    schemes = {}

    fl = np.clip(kft, 0, 3)
    fn = {0: "B=0(无B)", 1: "1-B", 2: "2-B", 3: ">=3-B"}
    schemes["K_ft"] = [(fn[v], np.nonzero(fl == v)[0]) for v in range(4)]

    cl = np.clip(K, 0, 3)
    cn = {0: "K=0(无真值链)", 1: "K=1", 2: "K=2", 3: "K>=3"}
    schemes["K_chain"] = [(cn[v], np.nonzero(cl == v)[0]) for v in range(4)]

    tl, tq1, tq2 = tercile_split(ntr)
    tn = {0: f"径迹数 low(<= {tq1:.0f})", 1: "径迹数 mid", 2: f"径迹数 high(> {tq2:.0f})"}
    schemes["ntracks"] = [(tn[v], np.nonzero(tl == v)[0]) for v in range(3)]

    pl, pq1, pq2 = tercile_split(npv)
    pn = {0: f"PV数 low(<= {pq1:.0f})", 1: "PV数 mid", 2: f"PV数 high(> {pq2:.0f})"}
    schemes["npv"] = [(pn[v], np.nonzero(pl == v)[0]) for v in range(3)]
    return schemes


def main():
    a = parse_args()
    t0 = time.time()
    module, evts = collect_events(a)
    nev = len(evts)
    print(f"\n=== {a.tag} · v{a.version} · 载入事件 {nev} 个 (用时 {time.time() - t0:.0f}s)", flush=True)

    # ---- 事件级特征: 真值链数 K (完整图) / 径迹数 / PV 数 / ft 链数 ----
    K = np.zeros(nev, dtype=int)
    ntr = np.zeros(nev, dtype=int)
    npv = np.zeros(nev, dtype=int)
    kft = np.zeros(nev, dtype=int)
    for ei, d in enumerate(evts):
        K[ei] = len(truth_clusters(d["g"]))
        ntr[ei] = int(d["nw"].numel())
        npv[ei] = int(d["g"]["pvs"].x.shape[0])
        kft[ei] = len(set(d["ft"].tolist()) - {1})       # ft 里非背景的 B 链数 (0=b 2=bbar)
    N_total = int(K.sum())
    print(f"  真值链总数 N_total = {N_total}  (K 分布 {dict(zip(*np.unique(K, return_counts=True)))};"
          f"  ft 链数分布 {dict(zip(*np.unique(kft, return_counts=True)))})", flush=True)

    # ---- 逐事件 × 逐阈值 解码 ----
    A = np.zeros((nev, len(GRID)), dtype=int)
    P = np.zeros((nev, len(GRID)), dtype=int)
    S = np.zeros((nev, len(GRID)), dtype=int)
    for ei, d in enumerate(evts):
        A[ei], P[ei], S[ei] = scan_event(d, GRID)
        if (ei + 1) % 25 == 0 or ei == nev - 1:
            print(f"    scanned {ei + 1}/{nev} 事件 ({time.time() - t0:.0f}s)", flush=True)

    # ---- 全局单阈值曲线 ----
    def agg(idxs, ti):
        return (int(A[idxs, ti].sum()), int(P[idxs, ti].sum()),
                int(K[idxs].sum()), int(S[idxs, ti].sum()))

    def rate(idxs, ti):
        a_i, _, k_i, _ = agg(idxs, ti)
        return 100.0 * a_i / max(1, k_i)

    all_idx = np.arange(nev)
    glob_rate = np.array([rate(all_idx, ti) for ti in range(len(GRID))])
    ti_global = int(np.argmax(glob_rate))
    thr_global = GRID[ti_global]

    print(f"\n=== 全局单阈值扫描 (所有 {nev} 事件, 固定分母 N_total={N_total}) ===")
    print(f"{'thr':>7}{'All#':>8}{'All_fix%':>10}{'Perf_fix%':>11}{'存活链数':>10}{'覆盖率%':>9}")
    for ti in range(len(GRID)):
        a_i, p_i, k_i, s_i = agg(all_idx, ti)
        print(f"{GRID[ti]:>7.3f}{a_i:>8}{100.0 * a_i / max(1, N_total):>10.2f}"
              f"{100.0 * p_i / max(1, N_total):>11.2f}{s_i:>10}{100.0 * s_i / max(1, N_total):>9.2f}")
    A_glob, P_glob, _, _ = agg(all_idx, ti_global)
    print(f"  → 全局单一最优 thr = {thr_global:.3f}: All# {A_glob} / All_fix "
          f"{100.0 * A_glob / max(1, N_total):.2f}% / Perf_fix {100.0 * P_glob / max(1, N_total):.2f}%")

    # ---- 分组: 每组最优 thr ----
    schemes = build_groups(K, kft, ntr, npv)
    rows = []

    def add_rows(scheme, gname, idxs, ti_best, ti_sm, r_best, r_glob):
        for ti in range(len(GRID)):
            a_i, p_i, k_i, s_i = agg(idxs, ti)
            rows.append(dict(group_type=scheme, group=gname, n_events=int(len(idxs)), K_total=int(k_i),
                             thr=GRID[ti], A=int(a_i), Perf_n=int(p_i),
                             success_rate=round(100.0 * a_i / max(1, k_i), 2),
                             all_fix=round(100.0 * a_i / max(1, N_total), 2),
                             coverage=round(100.0 * s_i / max(1, N_total), 2),
                             best_thr=GRID[ti_best], best_thr_smooth=GRID[ti_sm], best_rate=round(r_best, 2),
                             rate_at_global_thr=round(r_glob, 2), global_thr=thr_global,
                             is_best=int(ti == ti_best)))

    group_best = {}          # (scheme, group) -> (best_thr, best_rate, rate@global, n_evt, K_tot)
    print(f"\n=== 分组成功率 (每组最优 thr 为 in-sample oracle; N_total={N_total}) ===")
    print(f"{'方案':<8}{'分组':<18}{'n_evt':>6}{'K_tot':>7}{'最优thr':>9}{'best_rate%':>11}"
          f"{'@全局thr%':>10}{'Δ(pp)':>8}{'平滑thr':>9}{'平滑rate%':>10}{'贡献事件':>8}")
    for scheme, groups in schemes.items():
        for gname, idxs in groups:
            if len(idxs) == 0:
                continue
            gr = np.array([rate(idxs, ti) for ti in range(len(GRID))])
            ti_best = int(np.argmax(gr))
            ti_sm = best_ti_smooth(gr)
            r_best, r_glob = gr[ti_best], gr[ti_global]
            a_best, p_best, k_g, _ = agg(idxs, ti_best)
            n_contrib = int(np.count_nonzero(A[idxs, ti_best]))
            group_best[(scheme, gname)] = (GRID[ti_best], r_best, r_glob, len(idxs), k_g, GRID[ti_sm])
            print(f"{scheme:<8}{gname:<18}{len(idxs):>6}{k_g:>7}{GRID[ti_best]:>9.3f}{r_best:>11.2f}"
                  f"{r_glob:>10.2f}{r_best - r_glob:>+8.2f}{GRID[ti_sm]:>9.3f}{gr[ti_sm]:>10.2f}{n_contrib:>8}")
            add_rows(scheme, gname, idxs, ti_best, ti_sm, r_best, r_glob)
        print("-" * 100)

    # 全局行 (group_type=ALL)
    add_rows("ALL", "all", all_idx, ti_global, best_ti_smooth(glob_rate), glob_rate[ti_global], glob_rate[ti_global])

    # ---- oracle 上界 ----
    print(f"\n=== Oracle 上界 (每事件取所属组最优 thr) vs 全局单一 thr = {thr_global:.3f} ===")
    print(f"{'─ 全局单一 thr':<26}All# {A_glob:>6}   All_fix {100.0 * A_glob / max(1, N_total):>6.2f}%   "
          f"Perf_fix {100.0 * P_glob / max(1, N_total):>6.2f}%")
    oracle = {}
    for scheme, groups in schemes.items():
        A_or = P_or = A_sm = P_sm = 0
        for gname, idxs in groups:
            if len(idxs) == 0:
                continue
            gr = np.array([rate(idxs, ti) for ti in range(len(GRID))])
            a_i, p_i, _, _ = agg(idxs, int(np.argmax(gr)))
            A_or += a_i
            P_or += p_i
            a_s, p_s, _, _ = agg(idxs, best_ti_smooth(gr))
            A_sm += a_s
            P_sm += p_s
        gain = 100.0 * (A_or - A_glob) / max(1, N_total)
        rel = 100.0 * (A_or - A_glob) / max(1, A_glob)
        gain_sm = 100.0 * (A_sm - A_glob) / max(1, N_total)
        oracle[scheme] = (A_or, P_or, gain, rel, gain_sm)
        print(f"{'─ oracle(' + scheme + ')':<26}All# {A_or:>6}   All_fix {100.0 * A_or / max(1, N_total):>6.2f}%   "
              f"Perf_fix {100.0 * P_or / max(1, N_total):>6.2f}%   "
              f"增益 {gain:+.2f} pp (相对 +{rel:.1f}%)")
        print(f"{'   └ 平滑(3点)':<26}All# {A_sm:>6}   All_fix {100.0 * A_sm / max(1, N_total):>6.2f}%   "
              f"Perf_fix {100.0 * P_sm / max(1, N_total):>6.2f}%   "
              f"增益 {gain_sm:+.2f} pp")

    # ---- 关键对照: 1-B vs 2-B ----
    print("\n=== 关键对照 (K_ft 口径 = ft 给出的 B 链数) ===")
    for g in ["1-B", "2-B", ">=3-B"]:
        v = group_best.get(("K_ft", g))
        if v is None:
            print(f"  {g:<8} 无事件")
            continue
        t_b, r_b, r_g, n_ev, k_g, t_sm = v
        print(f"  {g:<8} n_evt={n_ev:<4} K_tot={k_g:<4} 组内最优 thr={t_b:.3f} (平滑 {t_sm:.3f})  成功率 {r_b:.2f}%"
              f"   (全局 thr {thr_global:.3f} 下 {r_g:.2f}%)")
    v1, v2 = group_best.get(("K_ft", "1-B")), group_best.get(("K_ft", "2-B"))
    if v1 and v2:
        print(f"  → Δthr(2-B − 1-B) = {v2[0] - v1[0]:+.3f}  (平滑口径 {v2[5] - v1[5]:+.3f})")
    if "K_ft" in oracle:
        print(f"  → oracle 增益 (K_ft 分组) = {oracle['K_ft'][2]:+.2f} pp (平滑 {oracle['K_ft'][4]:+.2f} pp)")

    os.makedirs(a.out, exist_ok=True)
    p_csv = f"{a.out}/per_event_thr_oracle.csv"
    with open(p_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {p_csv}  (总用时 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
