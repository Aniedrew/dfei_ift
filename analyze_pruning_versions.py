"""分析"剪枝线"各版本: 区分【判别力(AUC)】与【工作点/标定】, 并给出链存活曲线。

输入: report_figs/pruning_scores/v<ver>.npz (dump_pruning_scores.py)
      logs/fixed_denominator_metrics_inclusive_00342442.csv (该线的最终 eval 结果)
产出: report_figs/pruning_line_analysis.png / .csv
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
FIG = BASE + "/report_figs"
SC = FIG + "/pruning_scores"
CSV = BASE + "/logs/fixed_denominator_metrics_inclusive_00342442.csv"
THRS = np.round(np.arange(0.05, 0.995, 0.025), 4)

# version -> (短标签, 该版唯一的改动, 颜色)
VER = {
    550: ("v550 lr1e-4 base", "lr probe, no prune change", "tab:grey"),
    551: ("v551 lr3e-4", "lr 3e-4", "tab:grey"),
    552: ("v552 node w=5", "node_prune_weight 5 (lr 3e-5)", "tab:blue"),
    553: ("v553 edge w=3", "edge_prune_weight 3", "tab:red"),
    554: ("v554 node5+focal", "focal gamma 2", "tab:purple"),
    555: ("v555 chain-recall", "chain min-pooling recall", "tab:brown"),
    556: ("v556 stack", "node5 + edge3 + lr3e-4", "tab:cyan"),
    557: ("v557 node5 lr1e-4", "SOTA: node w=5 + lr 1e-4", "tab:green"),
    559: ("v559 control", "control (same protocol as 557)", "tab:olive"),
    562: ("v562 node5+edge3", "node5 + edge3 (repeat)", "tab:pink"),
    563: ("v563 v557 repeat", "reproduction of v557", "tab:green"),
    564: ("v564 node w=10", "node_prune_weight 10", "tab:orange"),
    565: ("v565 node w=20", "node_prune_weight 20", "darkorange"),
}
TAGS = {550: "lrprobe_v507_lr1e4", 551: "lrprobe_v507_lr3e4", 552: "prune_node5", 553: "prune_edge3",
        554: "prune_node5focal", 555: "prune_chainrec", 556: "stack_v551", 557: "hilr_node5",
        559: "hilr_ctrl", 563: "hilr_node5_rep"}


def auc(score, label):
    o = np.argsort(score); r = np.empty(len(o), float); r[o] = np.arange(1, len(o) + 1)
    p = label.sum(); n = len(label) - p
    return (r[label == 1].sum() - p * (p + 1) / 2) / (p * n) if p and n else np.nan


def load(ver):
    p = f"{SC}/v{ver}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    ns, nl = z["node_scores"], z["node_labels"]
    es, el = z["edge_scores"], z["edge_labels"]
    nnode, nedge = z["nnode"], z["nedge"]
    noff = np.concatenate([[0], np.cumsum(nnode)])[:-1]
    eoff = np.concatenate([[0], np.cumsum(nedge)])[:-1]
    cl, ce = z["chain_lens"], z["chain_elens"]
    cloff = np.concatenate([[0], np.cumsum(cl)])[:-1]
    ceoff = np.concatenate([[0], np.cumsum(ce)])[:-1]
    # 每条真值链 -> (全局节点索引, 全局边索引)
    chains = []
    for i in range(len(cl)):
        evt = np.searchsorted(np.cumsum(nnode), i, side="left")   # 该链属于哪个事件
        evt = int(np.searchsorted(np.cumsum(z["nchain"]), i, side="left"))
        nodes = z["chain_nodes"][cloff[i]:cloff[i] + cl[i]] + noff[evt]
        edgs = z["chain_edges"][ceoff[i]:ceoff[i] + ce[i]] + eoff[evt]
        chains.append((nodes, edgs))
    return dict(ns=ns, nl=nl, es=es, el=el, chains=chains, nevt=len(nnode))


def metrics(d):
    ns, nl, es, el = d["ns"], d["nl"], d["es"], d["el"]
    out = dict(n_node=len(ns), n_edge=len(es), n_chain=len(d["chains"]),
               auc_node=auc(ns, nl), auc_edge=auc(es, el),
               sig_frac_node=100 * nl.mean())
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97):
        kn = ns > thr
        rec = 100 * kn[nl == 1].mean() if (nl == 1).any() else np.nan
        bkg = 100 * kn[nl == 0].mean() if (nl == 0).any() else np.nan
        prec = 100 * nl[kn].mean() if kn.any() else np.nan
        ke = es > thr
        erec = 100 * ke[el == 1].mean() if (el == 1).any() else np.nan
        surv = np.mean([np.all(kn[n]) and np.all(ke[e]) for n, e in d["chains"]]) if d["chains"] else np.nan
        out.update({f"noderec_{thr}": rec, f"nodebkg_{thr}": bkg, f"nodeprec_{thr}": prec,
                    f"edgerec_{thr}": erec, f"chainsurv_{thr}": 100 * surv})
    # 信号/本底的分数分布 (标定漂移)
    out["sig_med"] = float(np.median(ns[nl == 1])); out["bkg_med"] = float(np.median(ns[nl == 0]))
    out["sig_gt09"] = 100 * float((ns[nl == 1] > 0.9).mean()); out["bkg_gt09"] = 100 * float((ns[nl == 0] > 0.9).mean())
    return out


def main():
    rows = {}
    curves = {}
    for ver in VER:
        d = load(ver)
        if d is None:
            print(f"缺 v{ver} 的 npz, 跳过")
            continue
        rows[ver] = metrics(d)
        cs = []
        for thr in THRS:
            kn = d["ns"] > thr; ke = d["es"] > thr
            cs.append(100 * np.mean([np.all(kn[n]) and np.all(ke[e]) for n, e in d["chains"]]))
        curves[ver] = np.array(cs)
        print(f"[v{ver}] AUC(node)={rows[ver]['auc_node']:.4f} AUC(edge)={rows[ver]['auc_edge']:.4f} "
              f"| @0.9 recall={rows[ver]['noderec_0.9']:.1f}% bkg={rows[ver]['nodebkg_0.9']:.2f}% "
              f"chain_surv={rows[ver]['chainsurv_0.9']:.1f}%")

    # 链存活随 thr 单调下降 -> 用"降到 0.5 能多买多少覆盖率、多带多少本底"量化权衡
    for ver, cs in curves.items():
        d = load(ver); j = int(np.argmin(np.abs(THRS - 0.5)))
        kn = d["ns"] > 0.5
        rows[ver]["surv_at_050"] = float(cs[j])
        rows[ver]["bkg_at_050"] = 100 * float(kn[d["nl"] == 0].mean())
        rows[ver]["surv_gain_090_to_050"] = float(cs[j]) - rows[ver]["chainsurv_0.9"]
        a = int(np.argmin(np.abs(THRS - 0.9)))
        rows[ver]["surv_drop_050_to_090"] = float(cs[0]) - float(cs[a])
        # 达到 80.6% 信号召回(=v559@0.9) 所需的阈值: 各版本"标定等效阈值"
        grid = np.round(np.arange(0.30, 1.0, 0.005), 4)
        recs = np.array([(d["ns"][d["nl"] == 1] > t).mean() * 100 for t in grid])
        i = int(np.argmin(np.abs(recs - 80.6)))
        rows[ver]["thr_for_recall80"] = float(grid[i])

    df = pd.DataFrame(rows).T
    df.index.name = "version"
    df["label"] = [VER[v][0] for v in df.index]
    df["lever"] = [VER[v][1] for v in df.index]
    # 与 eval 结果对账
    if os.path.exists(CSV):
        e = pd.read_csv(CSV)
        e = e.drop_duplicates(subset=["version", "tag"], keep="last")
        m = {v: e[(e["version"] == v) & (e["tag"] == TAGS.get(v, ""))] for v in df.index}
        df["eval_N"] = [float(m[v]["N"].iloc[0]) if len(m[v]) else np.nan for v in df.index]
        df["All_fix"] = [float(m[v]["All_fix%"].iloc[0]) if len(m[v]) else np.nan for v in df.index]
        df["Perf_fix"] = [float(m[v]["Perf_fix%"].iloc[0]) if len(m[v]) else np.nan for v in df.index]
    surv = df["chainsurv_0.9"] / 100 * 17700            # 该事件子集上"链存活数"的估计 (与 eval N 同量级)
    df["surv_est_N"] = surv
    df.to_csv(f"{FIG}/pruning_line_analysis.csv")
    print("\n写出", f"{FIG}/pruning_line_analysis.csv")

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    ax = axes[0, 0]
    for ver, (lab, lev, c) in VER.items():
        if ver not in rows:
            continue
        d = load(ver)
        o = np.argsort(-d["ns"]); s, l = d["ns"][o], d["nl"][o]
        tp = np.cumsum(l) / l.sum(); fp = np.cumsum(1 - l) / (len(l) - l.sum())
        ax.plot(np.concatenate([[0], fp]), np.concatenate([[0], tp]), lw=1.6, color=c,
                label=f"{lab} (AUC {rows[ver]['auc_node']:.4f})")
    ax.set_xscale("log"); ax.set_xlabel("background retention"); ax.set_ylabel("signal recall")
    ax.set_title("node pruning ROC (same 250 events per version)"); ax.grid(alpha=.25)
    ax.legend(fontsize=6, loc="lower right")

    ax = axes[0, 1]
    for ver in (550, 552, 553, 557, 559, 564, 565, 554):
        if ver not in rows:
            continue
        lab, lev, c = VER[ver]
        d = load(ver)
        sv = np.sort(d["ns"][d["nl"] == 1])
        cdf = np.arange(1, len(sv) + 1) / len(sv) * 100
        ax.plot(sv, cdf, lw=1.8, color=c, label=lab)
    ax.axvline(0.9, color="k", ls=":", lw=1); ax.text(0.905, 20, "thr 0.9", fontsize=8)
    ax.set_xlim(0.5, 1.0); ax.set_ylim(50, 100)
    ax.set_xlabel("node score (signal tracks)"); ax.set_ylabel("cumulative % of signal tracks")
    ax.set_title("calibration shift: how much signal passes thr 0.9"); ax.legend(fontsize=6); ax.grid(alpha=.25)

    ax = axes[0, 2]
    for ver, (lab, lev, c) in VER.items():
        if ver not in rows:
            continue
        d = load(ver)
        rec = [(d["ns"][d["nl"] == 1] > t).mean() * 100 for t in THRS]
        ax.plot(THRS, rec, lw=1.7, color=c, label=lab)
    ax.axvline(0.9, color="k", ls=":", lw=1); ax.text(0.905, 20, "thr 0.9", fontsize=8)
    ax.set_xlabel("node threshold"); ax.set_ylabel("signal-track recall %")
    ax.set_title("recall vs threshold (how many truth tracks survive)"); ax.grid(alpha=.25)

    ax = axes[1, 0]
    for ver, (lab, lev, c) in VER.items():
        if ver not in curves:
            continue
        ax.plot(THRS, curves[ver], lw=1.7, color=c, label=lab)
    ax.axvline(0.9, color="k", ls=":", lw=1); ax.text(0.905, 5, "thr 0.9", fontsize=8)
    ax.set_xlabel("threshold"); ax.set_ylabel("% of truth chains fully surviving")
    ax.set_title("chain survival vs threshold (monotone: lower cut = more chains, more junk)"); ax.grid(alpha=.25)
    ax.legend(fontsize=6)

    ax = axes[1, 1]
    for ver, (lab, lev, c) in VER.items():
        if ver not in rows:
            continue
        ax.scatter(rows[ver]["noderec_0.9"], rows[ver]["nodebkg_0.9"], color=c, s=45)
        ax.annotate(f"v{ver}", (rows[ver]["noderec_0.9"], rows[ver]["nodebkg_0.9"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("signal-track recall @ thr 0.9  [%]"); ax.set_ylabel("background retention @ 0.9  [%]")
    ax.set_title("operating point at the historical threshold"); ax.grid(alpha=.25)

    ax = axes[1, 2]
    sub = df.dropna(subset=["All_fix"])
    for ver, r in sub.iterrows():
        c = VER[ver][2] if ver in VER else "grey"
        ax.scatter(r["chainsurv_0.9"], r["All_fix"], color=c, s=50)
        ax.annotate(f"v{ver}", (r["chainsurv_0.9"], r["All_fix"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xlabel("chain survival @ 0.9 (this subset) [%]")
    ax.set_ylabel("All_fix % (official eval, 20 files)")
    ax.set_title("does coverage explain the metric?"); ax.grid(alpha=.25)
    fig.suptitle("Pruning line (July data): discrimination vs operating point", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{FIG}/pruning_line_analysis.png", dpi=130)
    print("写出", f"{FIG}/pruning_line_analysis.png")

    cols = ["label", "lever", "auc_node", "auc_edge", "noderec_0.9", "nodebkg_0.9", "nodeprec_0.9",
            "chainsurv_0.9", "thr_for_recall80", "surv_at_050", "bkg_at_050", "surv_gain_090_to_050",
            "sig_gt09", "bkg_gt09", "eval_N", "All_fix", "Perf_fix"]
    print("\n", df[cols].round(3).to_string())


if __name__ == "__main__":
    main()
