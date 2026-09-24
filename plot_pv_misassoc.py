"""PV 错配的"方向性"与"物理依赖"分析 —— 用 dump_pv_tracks.py 的逐径迹表。

核心量 (逐径迹, 已按管线口径只在 pv_filter 通过的边里取 argmin/argmax):
  ip_rank : 真值 PV 在"按 log_minIP 升序"里的名次 (0 = 真值 PV 恰好是 IP 最小的)
  ip_gap  : log_minIP(真值 PV) − min(log_minIP)      (rank>0 时为正; 越大越"不像")
  Δz      : z(判定的 PV) − z(真值 PV), 已用归一化常数反归一化到 cm (带符号)
产出:
  report_figs/pv_physics_dependence.png
  report_figs/pv_direction_delta.png
  report_figs/pv_direction_summary.csv
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/report_figs"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NORM = {"new": "/lzufs/user/guoqingxiang/DFEI_IFT_20260904/dfei_repo/preprocessing/normalization_dict.pt",
        "old": "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/dfei_repo/preprocessing/old_norm_ported.pt"}
SETS = [("v601 / 0904 incl", "v601_0904incl", "new", "tab:blue"),
        ("v557 / 0702 July", "v557_armA", "old", "tab:red"),
        ("v557 / 0904 renorm", "v557_armC", "old", "tab:green")]


def load(tag, norm):
    p = f"{FIG}/pv_tracks_{tag}.csv"
    if not os.path.exists(p):
        return None
    import torch
    d = pd.read_csv(p)
    d = d[d["true_pv"] >= 0].copy()
    nd = torch.load(NORM[norm], map_location="cpu")
    zc, zs = float(nd["center"]["zPV_reco"]), float(nd["scale"]["zPV_reco"])
    d["dz_pred_cm"] = d["dz_pred"] * zs
    d["dz_minip_cm"] = d["dz_minip"] * zs
    d["absdz_ours_cm"] = d["dz_pred_cm"].abs()
    d["absdz_minip_cm"] = d["dz_minip_cm"].abs()
    d["correct_ours"] = (d["pred_pv"] == d["true_pv"]).astype(int)
    d["correct_minip"] = (d["minip_pv"] == d["true_pv"]).astype(int)
    d["wrong_ours"] = 1 - d["correct_ours"]
    d["wrong_minip"] = 1 - d["correct_minip"]
    d["eta_abs"] = d["eta"].abs()
    d["chain_pos_frac"] = d["chain_pos"] / d["chain_len"].clip(lower=1)
    d["pv_mult"] = d.groupby(["evt", "true_pv"])["true_pv"].transform("size")
    return d


def binned(ax, d, col, nbins=8, logx=False, prefix="", c="tab:blue", both=True, ymax=100):
    x = d[col].astype(float)
    m = np.isfinite(x)
    d, x = d[m], x[m]
    if logx:
        x = np.log10(x.clip(lower=max(1e-6, np.nanpercentile(x, 1))))
    try:
        bins = pd.qcut(x, nbins, duplicates="drop")
    except Exception:
        return None
    g = d.groupby(bins, observed=True)
    mid = np.asarray(g.apply(lambda s: np.nanmedian(s[col]))).ravel()
    n = np.asarray(g.size()).ravel()
    K = min(len(mid), len(n))
    if K < 2:
        return None
    mid, n = mid[:K], n[:K]
    wo = np.asarray(100 * g["wrong_ours"].mean()).ravel()[:K]
    wm = np.asarray(100 * g["wrong_minip"].mean()).ravel()[:K]
    eo = 100 * np.sqrt(np.clip(wo / 100 * (1 - wo / 100) / n, 0, None))
    em = 100 * np.sqrt(np.clip(wm / 100 * (1 - wm / 100) / n, 0, None))
    ax.errorbar(mid, wo, yerr=eo, fmt="o-", ms=4, lw=1.9, capsize=2, color=c, label=f"{prefix}我们")
    if both:
        ax.errorbar(mid, wm, yerr=em, fmt="s--", ms=4, lw=1.5, capsize=2, color=c, alpha=.7,
                    label=f"{prefix}minIP")
    if logx:
        ax.set_xscale("log")
    out = pd.DataFrame(dict(x=mid, n=n, wrong_ours=wo, wrong_minip=wm))
    return out


def prep(ax, title, xlab, ymax=100):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel("错配率 %", fontsize=10)
    ax.set_ylim(0, ymax)
    ax.grid(alpha=.25)


def main():
    data = {}
    for lab, tag, norm, col in SETS:
        d = load(tag, norm)
        if d is not None:
            data[lab] = (d, col)
            hard = d[d.ip_rank > 0]
            print(f"[{lab}] n={len(d)}  错配: 我们 {100*d.wrong_ours.mean():.2f}%  minIP {100*d.wrong_minip.mean():.2f}%"
                  f" | rank==0 占 {100*(d.ip_rank==0).mean():.1f}% | rank>0 上我们错配 {100*hard.wrong_ours.mean():.1f}%"
                  f" (minIP 100%)")

    # ================= 图 1: 物理依赖 + 战场分解 =================
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    ax = axes[0, 0]
    for lab, (d, c) in data.items():
        binned(ax, d, "pt", nbins=8, logx=True, prefix=lab.split(" / ")[1] + " ", c=c)
    prep(ax, "错配率 vs 横动量 pT", "pT [MeV]（对数轴）")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for lab, (d, c) in data.items():
        binned(ax, d, "eta_abs", nbins=8, prefix=lab.split(" / ")[1] + " ", c=c)
    prep(ax, "错配率 vs 赝快度 |η|", "|η|   （LHCb 前向区：η 越大越靠前向）")

    ax = axes[0, 2]
    for lab, (d, c) in data.items():
        binned(ax, d, "ghost", nbins=8, prefix=lab.split(" / ")[1] + " ", c=c)
    prep(ax, "错配率 vs ghost 概率（可否重建径迹）", "Prob_ghost")
    ax.legend(fontsize=7, ncol=2)

    # ---- 核心面板: 战场分解 ----
    ax = axes[1, 0]
    labels, xs = [], np.arange(len(data) * 2)
    W = 0.34
    for i, (lab, (d, c)) in enumerate(data.items()):
        hard = d[d.ip_rank > 0]
        vals = [100 * d.wrong_ours.mean(), 100 * hard.wrong_ours.mean()]
        mins = [100 * d.wrong_minip.mean(), 100 * hard.wrong_minip.mean()]
        ax.bar(i * 2 - W / 2, vals[0], W, fc=c, ec="white", label="我们" if i == 0 else None)
        ax.bar(i * 2 + W / 2, mins[0], W, fc=c, ec="white", alpha=.45, hatch="//",
               label="minIP" if i == 0 else None)
        ax.text(i * 2 - W / 2, vals[0] + 1.5, f"{vals[0]:.0f}", ha="center", fontsize=8, color=c)
        ax.text(i * 2 + W / 2, mins[0] + 1.5, f"{mins[0]:.0f}", ha="center", fontsize=8, color=c)
        ax.bar(3 + i * 2 - W / 2, vals[1], W, fc=c, ec="white", hatch="..")
        ax.bar(3 + i * 2 + W / 2, mins[1], W, fc=c, ec="white", alpha=.45, hatch="//")
        ax.text(3 + i * 2 - W / 2, vals[1] + 1.5, f"{vals[1]:.0f}", ha="center", fontsize=8, color=c)
        ax.text(3 + i * 2 + W / 2, mins[1] + 1.5, f"{mins[1]:.0f}", ha="center", fontsize=8, color=c)
        labels += [lab.split(" / ")[0], ""]
    ax.set_xticks(list(np.arange(len(data) * 2)))
    ax.set_xticklabels([l if l else "" for l in labels], fontsize=8, rotation=20)
    ax.axvline(2.5, color="grey", ls=":", lw=1)
    ax.text(1, 108, "全体径迹", ha="center", fontsize=9, color=GREY)
    ax.text(5, 108, "硬子集：真值 PV ≠ 最近 IP 的 PV（占 10-12%）", ha="center", fontsize=9, color=RED)
    ax.set_ylim(0, 118)
    prep(ax, "★ 战场分解：整体差距几乎全部来自这 10%", "")
    ax.set_ylabel("错配率 %")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.25, axis="y")

    ax = axes[1, 1]
    for lab, (d, c) in data.items():
        g = d[d.ip_rank >= 0].groupby("ip_rank")
        n = g.size()
        mid = np.asarray(g.ip_rank.median())
        wo = 100 * g["wrong_ours"].mean()
        wm = 100 * g["wrong_minip"].mean()
        K = (n >= 20).sum()
        ax.plot(mid[:K], np.asarray(wo)[:K], "o-", color=c, lw=1.9, label=f"{lab.split(' / ')[1]} 我们")
        ax.plot(mid[:K], np.asarray(wm)[:K], "s--", color=c, lw=1.4, alpha=.7,
                label=f"{lab.split(' / ')[1]} minIP")
    prep(ax, "错配率 vs 真值 PV 的 IP 名次（精确分组）", "真值 PV 的 IP 名次 0/1/2/3/4（0 = IP 最小）")
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1, 2]
    for lab, (d, c) in data.items():
        binned(ax, d, "npvs", nbins=6, prefix=lab.split(" / ")[1] + " ", c=c)
    prep(ax, "错配率 vs 事件 PV 数（堆积）", "事件内 PV 数")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("PV 错配对物理量的依赖（实心=我们，虚线=minIP；误差棒为 √(p(1−p)/n)）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p1 = f"{FIG}/pv_physics_dependence.png"
    fig.savefig(p1, dpi=130); plt.close(fig)
    print("写出", p1)

    # ================= 图 2: 方向性 =================
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    # (1) 带符号 Δz 分布
    ax = axes[0, 0]
    for lab, (d, c) in data.items():
        wm = d[d.wrong_minip == 1]["dz_minip_cm"]
        wo = d[d.wrong_ours == 1]["dz_pred_cm"]
        ax.hist(wm, bins=60, range=(-30, 30), histtype="step", lw=1.8, color=c,
                label=f"{lab.split(' / ')[1]} minIP")
        ax.hist(wo, bins=60, range=(-30, 30), histtype="stepfilled", lw=1.2, color=c, alpha=.25,
                label=f"{lab.split(' / ')[1]} 我们")
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("Δz = z(判定的 PV) − z(真值 PV)   [cm]", fontsize=10)
    ax.set_ylabel("错配径迹数", fontsize=10)
    ax.set_title("错配的方向（带符号 Δz，只统计错配径迹；|Δz|>30cm 未画）", fontsize=11)
    ax.legend(fontsize=6.5); ax.grid(alpha=.25)

    # (2) 错配"错多远"：|Δz| 的分布
    ax = axes[0, 1]
    labs, w_med, m_med, w_far, m_far = [], [], [], [], []
    for lab, (d, c) in data.items():
        wo = d[d.wrong_ours == 1]["absdz_ours_cm"]
        wm = d[d.wrong_minip == 1]["absdz_minip_cm"]
        labs.append(lab.split(" / ")[0]); w_med.append(wo.median()); m_med.append(wm.median())
        w_far.append(100 * (wo > 10).mean()); m_far.append(100 * (wm > 10).mean())
    x = np.arange(len(labs))
    ax.bar(x - 0.2, w_med, 0.4, color="tab:blue", label="我们")
    ax.bar(x + 0.2, m_med, 0.4, color="tab:red", alpha=.6, hatch="//", label="minIP")
    for i in range(len(labs)):
        ax.text(i - 0.2, w_med[i] + .2, f"{w_med[i]:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, m_med[i] + .2, f"{m_med[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8, rotation=12)
    ax.set_ylabel("错配时 |Δz| 的中位数 [cm]", fontsize=10)
    ax.set_title("错配'错多远'：我们错在邻近 PV，minIP 错得更远", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    # (3) 救援率 vs IP 间隙
    ax = axes[0, 2]
    for lab, (d, c) in data.items():
        hard = d[(d.ip_rank > 0) & np.isfinite(d.ip_gap)]
        if len(hard) < 30:
            continue
        b = pd.qcut(hard["ip_gap"], 6, duplicates="drop")
        g = hard.groupby(b, observed=True)
        mid = np.asarray(g.ip_gap.median()); rec = 100 * np.asarray(g["correct_ours"].mean())
        err = 100 * np.sqrt(rec / 100 * (1 - rec / 100) / np.asarray(g.size()))
        ax.errorbar(mid, rec, yerr=err, fmt="o-", color=c, capsize=2, lw=1.9,
                    label=f"{lab.split(' / ')[1]}（n={len(hard)}）")
    ax.axhline(0, color="grey", lw=.8, ls=":")
    ax.text(0.02, 3, "minIP 在这里恒为 0（必错）", fontsize=8, color=GREY)
    ax.set_xlabel("IP 间隙 Δlog(minIP)：真值 PV 比“最近”差多少（越大越难）", fontsize=10)
    ax.set_ylabel("我们救回的比例 %", fontsize=10)
    ax.set_title("在 minIP 必错的径迹上，我们救回多少", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25)

    # (4) 错配率 vs 真值 PV 的径迹多重数
    ax = axes[1, 0]
    for lab, (d, c) in data.items():
        binned(ax, d, "pv_mult", nbins=8, prefix=lab.split(" / ")[1] + " ", c=c)
    prep(ax, "错配率 vs 真值 PV 上的径迹多重数", "该 PV 关联的真值径迹数")
    ax.legend(fontsize=7, ncol=2)

    # (5) 救援率 vs IP 名次（难度分级）
    ax = axes[1, 1]
    buckets = [(1, 1), (2, 2), (3, 99)]
    names = ["名次 1", "名次 2", "名次 ≥3"]
    xb = np.arange(len(buckets)); W2 = 0.26
    for j, (lab, (d, c)) in enumerate(data.items()):
        rec = []
        for lo, hi in buckets:
            sub = d[(d.ip_rank >= lo) & (d.ip_rank <= hi)]
            rec.append(100 * sub.correct_ours.mean() if len(sub) >= 20 else np.nan)
        ax.bar(xb + (j - 1) * W2, rec, W2, color=c, label=lab.split(" / ")[0])
        for i, v in enumerate(rec):
            if np.isfinite(v):
                ax.text(xb[i] + (j - 1) * W2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(xb); ax.set_xticklabels(names)
    ax.set_ylim(0, 100); ax.set_ylabel("我们救回的比例 %", fontsize=10)
    ax.set_title("救援率随难度下降（minIP 在这三档全为 0）", fontsize=11)
    ax.legend(fontsize=7.5); ax.grid(alpha=.25, axis="y")

    # (6) 错配率 vs 真值 PV 索引
    ax = axes[1, 2]
    for lab, (d, c) in data.items():
        dd = d[d["true_pv"].between(0, 5)]
        binned(ax, dd, "true_pv", nbins=6, prefix=lab.split(" / ")[1] + " ", c=c)
    prep(ax, "错配率 vs 真值 PV 的索引（前位 / 后位）", "真值 PV 索引（事件内排序）")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("PV 错配的方向性与结构依赖（Δz 已用归一化常数反归一化到 cm）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p2 = f"{FIG}/pv_direction_delta.png"
    fig.savefig(p2, dpi=130); plt.close(fig)
    print("写出", p2)

    # ================= 汇总表 =================
    rows = []
    for lab, (d, c) in data.items():
        hard = d[d.ip_rank > 0]
        wo = d[d.wrong_ours == 1]; wm = d[d.wrong_minip == 1]
        rows.append(dict(dataset=lab, n_tracks=len(d),
                         mis_ours=100 * d.wrong_ours.mean(), mis_minip=100 * d.wrong_minip.mean(),
                         mis_ours_rank0=100 * d[d.ip_rank == 0].wrong_ours.mean(),
                         mis_ours_hard=100 * hard.wrong_ours.mean(),
                         hard_frac=100 * (d.ip_rank > 0).mean(),
                         rescue_on_hard=100 * hard.correct_ours.mean(), n_hard=len(hard),
                         dz_ours_mean_cm=wo.dz_pred_cm.mean(), dz_ours_med_cm=wo.dz_pred_cm.median(),
                         dz_minip_mean_cm=wm.dz_minip_cm.mean(), dz_minip_med_cm=wm.dz_minip_cm.median(),
                         absdz_ours_med_cm=wo.absdz_ours_cm.median(), absdz_minip_med_cm=wm.absdz_minip_cm.median(),
                         ours_frac_absdz_gt10cm=100 * (wo.absdz_ours_cm > 10).mean(),
                         minip_frac_absdz_gt10cm=100 * (wm.absdz_minip_cm > 10).mean(),
                         minip_err_frac_dz_negative=100 * (wm.dz_minip_cm < 0).mean(),
                         ours_err_frac_dz_negative=100 * (wo.dz_pred_cm < 0).mean()))
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{FIG}/pv_direction_summary.csv", index=False)
    print()
    print(summ.round(2).to_string(index=False))
    print("\n写出", f"{FIG}/pv_direction_summary.csv")


GREY = "#7F7F7F"
RED = "#C0392B"

if __name__ == "__main__":
    main()
