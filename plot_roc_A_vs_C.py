"""把 arm A (0702) 与 arm C (0904, 归一化对齐) 的 pruning ROC 画在一张图上。
左: 节点剪枝 (track 是否属于真值 b 链); 右: tt 边剪枝 (是否真值结构边)。
同时标出 thr=0.9 / 0.7 的工作点。
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "report_figs"
ARMS = [("armA_0702", "A: 0702 (July, bugged)", "tab:red"),
        ("armC_0904", "C: 0904 (fixed, renorm)", "tab:blue")]
PAIRS = [("node", "Node pruning (track in truth b chain)"), ("edge", "tt edge pruning (truth structural edge)")]
MARKS = [0.9, 0.7]


def load(tag, kind):
    p = f"{OUT}/roc_{tag}_v557_{kind}.npy"
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return a[0], a[1].astype(int)      # score, label


def roc_curve_point(scores, labels):
    """返回 (fpr, tpr) 曲线点 + 各条阈值点。"""
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    P, N = l.sum(), len(l) - l.sum()
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    tpr = tp / max(1, P)
    fpr = fp / max(1, N)
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr]), s, fpr, tpr


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (kind, title) in zip(axes, PAIRS):
        for tag, label, color in ARMS:
            got = load(tag, kind)
            if got is None:
                print("缺文件:", tag, kind); continue
            sc, lb = got
            x, y, s_sorted, fpr_s, tpr_s = roc_curve_point(sc, lb)
            # AUC
            order = np.argsort(sc); r = np.empty(len(sc), float); r[order] = np.arange(1, len(sc) + 1)
            P = lb.sum(); N = len(lb) - P
            auc = (r[lb == 1].sum() - P * (P + 1) / 2) / (P * N)
            ax.plot(x, y, color=color, lw=1.8, label=f"{label}  AUC={auc:.4f}  (n={len(sc)})")
            # 标出工作点
            for thr in MARKS:
                keep = s_sorted > thr
                if keep.any():
                    ep = keep.sum() / max(1, P)        # TPR
                    fb = (~keep).sum() / max(1, N)     # FPR
                    ax.plot([fb], [ep], marker="o" if thr == MARKS[0] else "^", ms=8, color=color,
                            mfc="white", mew=1.6)
                    ax.annotate(f"thr={thr}", (fb, ep), textcoords="offset points",
                                xytext=(6, -12 if thr == MARKS[0] else 6), fontsize=8, color=color)
        ax.plot([1e-6, 1], [1e-6, 1], "--", color="grey", lw=1, label="random")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\epsilon_{bkg}$ (FPR)")
        ax.set_ylabel(r"$\epsilon_{s}$ (TPR)")
        ax.set_title(title)
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Pruning ROC: arm A (0702, July data) vs arm C (0904 fixed, renorm) · v557 · 300 events", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{OUT}/roc_pruning_A_vs_C_v557.png"
    fig.savefig(p, dpi=130)
    print("写出", p)


if __name__ == "__main__":
    main()
