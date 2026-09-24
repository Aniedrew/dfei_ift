"""English version of the PV-association figures (same data, English labels).
Produces:
  report_figs/pv_bias_occupancy_en.png    (occupancy + accuracy vs true PV index)
  report_figs/pv_bias_chain_en.png        (chain agreement, difference decomposition, pile-up)
  report_figs/pv_physics_dependence_en.png(mis-association vs pT/|eta|/ghost/battlefield/IP-rank/nPVs)
  report_figs/pv_direction_delta_en.png   (signed dz, |dz|, rescue, rescue vs rank, multiplicity, PV index)
Inputs: report_figs/pv_tracks_<tag>.csv (from dump_pv_tracks.py) and the eval signal_reco_df CSVs.
"""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
FIG = BASE + "/report_figs"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F4E79"; BLUE = "#2E75B6"; RED = "#C0392B"; GREEN = "#1E8449"; GREY = "#7F7F7F"
NORM = {"new": "/lzufs/user/guoqingxiang/DFEI_IFT_20260904/dfei_repo/preprocessing/normalization_dict.pt",
        "old": "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/dfei_repo/preprocessing/old_norm_ported.pt"}
TRACKS = [("0904 new-data model (v601)", "v601_0904incl", "new", "tab:blue"),
          ("0702 July (v557, arm A)", "v557_armA", "old", "tab:red"),
          ("0904 degraded (v557, arm C)", "v557_armC", "old", "tab:green")]
# chain-level CSV per (dataset, model, csv, colour)
CHAINS = [("v601 / 0904 incl (new-MC trained)",
           f"{BASE}/LHCb_logs/DFEI/version_601/signal_reco_df_inclusive_00342451__v601_0904.csv", "tab:blue"),
          ("v557 / 0702 July (arm A)",
           f"{BASE}/LHCb_logs/DFEI/version_557/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armA_0702.csv", "tab:red"),
          ("v557 / 0904 renorm (arm C)",
           f"{BASE}/LHCb_logs/DFEI/version_557/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armC_renorm.csv", "tab:green")]


# ----------------------------------------------------------------- helpers
def load_tracks(tag, norm):
    p = f"{FIG}/pv_tracks_{tag}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d["true_pv"] >= 0].copy()
    nd = torch.load(NORM[norm], map_location="cpu")
    zs = float(nd["scale"]["zPV_reco"])
    d["dz_ours_cm"] = d["dz_pred"] * zs
    d["dz_minip_cm"] = d["dz_minip"] * zs
    d["absdz_ours_cm"] = d["dz_ours_cm"].abs()
    d["absdz_minip_cm"] = d["dz_minip_cm"].abs()
    d["ours_ok"] = (d["pred_pv"] == d["true_pv"]).astype(int)
    d["minip_ok"] = (d["minip_pv"] == d["true_pv"]).astype(int)
    d["wrong_ours"] = 1 - d["ours_ok"]; d["wrong_minip"] = 1 - d["minip_ok"]
    d["eta_abs"] = d["eta"].abs()
    d["pv_mult"] = d.groupby(["evt", "true_pv"])["true_pv"].transform("size")
    return d


def curve(ax, d, col, nbins=8, logx=False, label="", c="tab:blue", both=True):
    x = d[col].astype(float)
    m = np.isfinite(x); d, x = d[m], x[m]
    if logx:
        x = np.log10(x.clip(lower=max(1e-6, np.nanpercentile(x, 1))))
    bins = pd.qcut(x, nbins, duplicates="drop")
    g = d.groupby(bins, observed=True)
    mid = np.asarray(g.apply(lambda s: np.nanmedian(s[col]))).ravel()
    n = np.asarray(g.size()).ravel(); K = min(len(mid), len(n))
    if K < 2:
        return None
    mid, n = mid[:K], n[:K]
    wo = np.asarray(100 * g["wrong_ours"].mean()).ravel()[:K]
    wm = np.asarray(100 * g["wrong_minip"].mean()).ravel()[:K]
    eo = 100 * np.sqrt(np.clip(wo / 100 * (1 - wo / 100) / n, 0, None))
    em = 100 * np.sqrt(np.clip(wm / 100 * (1 - wm / 100) / n, 0, None))
    ax.errorbar(mid, wo, yerr=eo, fmt="o-", ms=4, lw=1.9, capsize=2, color=c, label=f"{label} ours")
    if both:
        ax.errorbar(mid, wm, yerr=em, fmt="s--", ms=4, lw=1.5, capsize=2, color=c, alpha=.7,
                    label=f"{label} minIP")
    if logx:
        ax.set_xscale("log")
    return pd.DataFrame(dict(x=mid, n=n, wrong_ours=wo, wrong_minip=wm))


def prep(ax, title, xlab, ylab="mis-association %"):
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10); ax.set_ylim(0, 100); ax.grid(alpha=.25)


# ----------------------------------------------------------------- figs
def fig_physics():
    data = {lab: (load_tracks(tag, nrm), c) for lab, tag, nrm, c in TRACKS}
    data = {k: v for k, v in data.items() if v[0] is not None}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    ax = axes[0, 0]
    for lab, (d, c) in data.items():
        curve(ax, d, "pt", logx=True, label=lab.split(" (")[0], c=c)
    prep(ax, "mis-association vs transverse momentum", "pT [MeV] (log axis)"); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for lab, (d, c) in data.items():
        curve(ax, d, "eta_abs", label=lab.split(" (")[0], c=c)
    prep(ax, "mis-association vs pseudo-rapidity", "|eta|  (large = forward region)")

    ax = axes[0, 2]
    for lab, (d, c) in data.items():
        curve(ax, d, "ghost", label=lab.split(" (")[0], c=c)
    prep(ax, "mis-association vs ghost probability", "Prob_ghost"); ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    W = 0.34
    for i, (lab, (d, c)) in enumerate(data.items()):
        hard = d[d.ip_rank > 0]
        v = [100 * d.wrong_ours.mean(), 100 * hard.wrong_ours.mean()]
        w = [100 * d.wrong_minip.mean(), 100 * hard.wrong_minip.mean()]
        for j, off in enumerate([0, 3]):
            ax.bar(i * 2 + off - W / 2, v[j], W, fc=c, ec="white",
                   label="ours" if (i == 0 and j == 0) else None)
            ax.bar(i * 2 + off + W / 2, w[j], W, fc=c, ec="white", alpha=.45, hatch="//",
                   label="minIP" if (i == 0 and j == 0) else None)
            ax.text(i * 2 + off - W / 2, v[j] + 1.5, f"{v[j]:.0f}", ha="center", fontsize=8, color=c)
            ax.text(i * 2 + off + W / 2, w[j] + 1.5, f"{w[j]:.0f}", ha="center", fontsize=8, color=c)
    ax.set_xticks(np.arange(len(data) * 2))
    ax.set_xticklabels(sum([[lab.split(" (")[0], ""] for lab in data], []), fontsize=7.5, rotation=15)
    ax.axvline(2.5, color="grey", ls=":", lw=1)
    ax.text(1, 105, "all tracks", ha="center", fontsize=9, color=GREY)
    ax.text(5, 105, "hard subset: truth PV is NOT the closest-IP PV (10-12%)", ha="center", fontsize=9, color=RED)
    ax.set_ylim(0, 118)
    prep(ax, "* the whole difference lives in these 10-12%", ""); ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.25, axis="y")

    ax = axes[1, 1]
    for lab, (d, c) in data.items():
        g = d[d.ip_rank >= 0].groupby("ip_rank")
        n = g.size(); K = (n >= 20).sum()
        ax.plot(np.asarray(g.ip_rank.median())[:K], np.asarray(100 * g["wrong_ours"].mean())[:K], "o-",
                color=c, lw=1.9, label=f"{lab.split(' (')[0]} ours")
        ax.plot(np.asarray(g.ip_rank.median())[:K], np.asarray(100 * g["wrong_minip"].mean())[:K], "s--",
                color=c, lw=1.4, alpha=.7, label=f"{lab.split(' (')[0]} minIP")
    prep(ax, "mis-association vs IP rank of the true PV", "IP rank of the true PV (0 = smallest IP)")
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1, 2]
    for lab, (d, c) in data.items():
        curve(ax, d, "npvs", nbins=6, label=lab.split(" (")[0], c=c)
    prep(ax, "mis-association vs number of PVs (pile-up)", "nPVs in event"); ax.legend(fontsize=7, ncol=2)

    fig.suptitle("PV mis-association vs physics variables (solid = ours, dashed = minIP)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{FIG}/pv_physics_dependence_en.png", dpi=130); plt.close(fig)
    print("wrote pv_physics_dependence_en.png")


def fig_direction():
    data = {lab: (load_tracks(tag, nrm), c) for lab, tag, nrm, c in TRACKS}
    data = {k: v for k, v in data.items() if v[0] is not None}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    ax = axes[0, 0]
    for lab, (d, c) in data.items():
        ax.hist(d[d.wrong_minip == 1].dz_minip_cm, bins=60, range=(-30, 30), histtype="step", lw=1.8,
                color=c, label=f"{lab.split(' (')[0]} minIP")
        ax.hist(d[d.wrong_ours == 1].dz_ours_cm, bins=60, range=(-30, 30), histtype="stepfilled", lw=1.2,
                color=c, alpha=.25, label=f"{lab.split(' (')[0]} ours")
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("dz = z(assigned PV) - z(true PV)   [cm]", fontsize=10)
    ax.set_ylabel("number of mis-assigned tracks", fontsize=10)
    ax.set_title("direction of errors (signed dz, wrong tracks only)", fontsize=11)
    ax.legend(fontsize=6.5); ax.grid(alpha=.25)

    ax = axes[0, 1]
    labs, wm_, mm_ = [], [], []
    for lab, (d, c) in data.items():
        labs.append(lab.split(" (")[0])
        wm_.append(d[d.wrong_ours == 1].absdz_ours_cm.median())
        mm_.append(d[d.wrong_minip == 1].absdz_minip_cm.median())
    x = np.arange(len(labs))
    ax.bar(x - 0.2, wm_, 0.4, color="tab:blue", label="ours")
    ax.bar(x + 0.2, mm_, 0.4, color="tab:red", alpha=.6, hatch="//", label="minIP")
    for i in range(len(labs)):
        ax.text(i - 0.2, wm_[i] + .15, f"{wm_[i]:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, mm_[i] + .15, f"{mm_[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8, rotation=12)
    ax.set_ylabel("median |dz| when wrong  [cm]", fontsize=10)
    ax.set_title("how far the error lands: ours nearby, minIP further", fontsize=11)
    ax.set_ylim(0, max(mm_) * 1.25); ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    ax = axes[0, 2]
    for lab, (d, c) in data.items():
        hard = d[(d.ip_rank > 0) & np.isfinite(d.ip_gap)]
        if len(hard) < 30:
            continue
        b = pd.qcut(hard["ip_gap"], 6, duplicates="drop")
        g = hard.groupby(b, observed=True)
        rec = 100 * np.asarray(g["ours_ok"].mean())
        err = 100 * np.sqrt(rec / 100 * (1 - rec / 100) / np.asarray(g.size()))
        ax.errorbar(np.asarray(g.ip_gap.median()), rec, yerr=err, fmt="o-", color=c, capsize=2, lw=1.9,
                    label=f"{lab.split(' (')[0]} (n={len(hard)})")
    ax.text(0.02, 3, "minIP is exactly 0 here (it must fail)", fontsize=8, color=GREY)
    ax.set_xlabel("IP gap  dlog(minIP): how much less 'IP-like' the true PV is", fontsize=10)
    ax.set_ylabel("we rescue [%]", fontsize=10); ax.set_ylim(0, 100)
    ax.set_title("rescue rate on the tracks minIP must get wrong", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25)

    ax = axes[1, 0]
    buckets = [(1, 1), (2, 2), (3, 99)]
    xb = np.arange(3); W2 = 0.26
    for j, (lab, (d, c)) in enumerate(data.items()):
        rec = [100 * d[(d.ip_rank >= lo) & (d.ip_rank <= hi)].ours_ok.mean()
               if len(d[(d.ip_rank >= lo) & (d.ip_rank <= hi)]) >= 20 else np.nan for lo, hi in buckets]
        ax.bar(xb + (j - 1) * W2, rec, W2, color=c, label=lab.split(" (")[0])
        for i, v in enumerate(rec):
            if np.isfinite(v):
                ax.text(xb[i] + (j - 1) * W2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(xb); ax.set_xticklabels(["rank 1", "rank 2", "rank >=3"])
    ax.set_ylim(0, 100); ax.set_ylabel("we rescue [%]", fontsize=10)
    ax.set_title("rescue rate vs difficulty (minIP = 0 in all three)", fontsize=11)
    ax.legend(fontsize=7.5); ax.grid(alpha=.25, axis="y")

    ax = axes[1, 1]
    for lab, (d, c) in data.items():
        curve(ax, d, "pv_mult", label=lab.split(" (")[0], c=c)
    prep(ax, "mis-association vs truth multiplicity of that PV", "truth tracks on that PV")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 2]
    for lab, (d, c) in data.items():
        curve(ax, d[d["true_pv"].between(0, 5)], "true_pv", nbins=6, label=lab.split(" (")[0], c=c)
    prep(ax, "mis-association vs the true PV index", "true PV index (order inside event)")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("direction and structure of PV mis-assignment (dz converted back to cm)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{FIG}/pv_direction_delta_en.png", dpi=130); plt.close(fig)
    print("wrote pv_direction_delta_en.png")

    rows = []
    for lab, (d, c) in data.items():
        hard = d[d.ip_rank > 0]; wo = d[d.wrong_ours == 1]; wm = d[d.wrong_minip == 1]
        rows.append(dict(dataset=lab, n_tracks=len(d), mis_ours=100 * d.wrong_ours.mean(),
                         mis_minip=100 * d.wrong_minip.mean(), mis_ours_easy=100 * d[d.ip_rank == 0].wrong_ours.mean(),
                         hard_frac=100 * (d.ip_rank > 0).mean(), mis_ours_hard=100 * hard.wrong_ours.mean(),
                         rescue_on_hard=100 * hard.ours_ok.mean(), n_hard=len(hard),
                         dz_ours_mean=wo.dz_ours_cm.mean(), dz_minip_mean=wm.dz_minip_cm.mean(),
                         absdz_ours_med=wo.absdz_ours_cm.median(), absdz_minip_med=wm.absdz_minip_cm.median(),
                         frac_neg_ours=100 * (wo.dz_ours_cm < 0).mean(), frac_neg_minip=100 * (wm.dz_minip_cm < 0).mean()))
    pd.DataFrame(rows).round(3).to_csv(f"{FIG}/pv_direction_summary_en.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def fig_bias():
    """occupancy + accuracy-vs-true-index and the chain-level panels, from the eval CSVs"""
    import yaml  # noqa
    def parse(s):
        return np.array([int(x) for x in str(s).split("_")], dtype=int)

    def load_chains(path):
        d = pd.read_csv(path)
        rows = []
        for ci, r in enumerate(d.itertuples(index=False)):
            t, p, m = parse(r.true_pv), parse(r.pred_pv), parse(r.minIP_pv)
            if not (len(t) == len(p) == len(m)):
                continue
            for k in range(len(t)):
                if t[k] < 0:
                    continue
                rows.append((ci, t[k], p[k], m[k], int(r.npvs)))
        df = pd.DataFrame(rows, columns=["chain", "true", "ours", "minip", "npvs"])
        df["minip_vote"] = df.groupby("chain")["minip"].transform(lambda s: s.mode().iloc[0])
        return df

    dfs = [(lab, load_chains(p), c) for lab, p, c in CHAINS if os.path.exists(p)]
    # ---- occupancy ----
    fig, axes = plt.subplots(2, len(dfs), figsize=(5.6 * len(dfs), 9))
    if len(dfs) == 1:
        axes = axes.reshape(2, 1)
    for j, (lab, d, c) in enumerate(dfs):
        ax = axes[0, j]
        idx = np.arange(0, 8); w = 0.27
        occ = {k: 100 * d[k].value_counts(normalize=True) / 1.0 for k in ("true", "ours", "minip")}
        ax.bar(idx - w, [occ["true"].get(i, 0) for i in idx], w, label="truth", color="grey")
        ax.bar(idx, [occ["ours"].get(i, 0) for i in idx], w, label="ours (HGNN)", color=c)
        ax.bar(idx + w, [occ["minip"].get(i, 0) for i in idx], w, label="minIP", color=c, alpha=.45, hatch="//")
        ax.set_title(f"{lab}\nPV index occupancy", fontsize=9)
        ax.set_xlabel("PV index"); ax.set_ylabel("% of track slots"); ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")
        ax = axes[1, j]
        idx = np.sort(d["true"].unique())
        for col, style, nm, cc in [("ours", "o-", "ours (HGNN)", c), ("minip", "s--", "minIP", c),
                                   ("minip_vote", "^:", "minIP + chain vote", "black")]:
            acc = [100 * (d.loc[d["true"] == i, col] == i).mean() for i in idx]
            ax.plot(idx, acc, style, color=cc, alpha=.75, label=nm, lw=1.7, ms=4)
        ax.set_title("accuracy vs TRUE PV index", fontsize=9)
        ax.set_xlabel("true PV index"); ax.set_ylabel("accuracy %")
        ax.set_ylim(0, 102); ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.suptitle("PV association bias: occupancy and conditional accuracy (signal-chain tracks)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{FIG}/pv_bias_occupancy_en.png", dpi=130); plt.close(fig)
    print("wrote pv_bias_occupancy_en.png")

    # ---- chain-level ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    labs = [lab for lab, _, _ in dfs]; x = np.arange(len(labs)); w = 0.2
    def chain_stat(d, col, stat):
        g = d.groupby("chain")
        if stat == "allok":
            return 100 * g.apply(lambda s, cc=col: (s[cc] == s["true"]).all()).mean()
        return 100 * g.apply(lambda s, cc=col: s[cc].nunique() == 1).mean()
    ax = axes[0]
    for k, (col, nm, cc) in enumerate([("ours", "ours (per-track)", "tab:blue"),
                                       ("minip", "minIP", "tab:red"),
                                       ("minip_vote", "minIP + chain vote", "tab:orange")]):
        ax.bar(x + (k - 1) * w, [chain_stat(d, col, "allok") for _, d, _ in dfs], w, label=nm, color=cc)
    ax.set_xticks(x); ax.set_xticklabels([l.split(" / ")[0] + "\n" + l.split(" / ")[1] for l in labs], fontsize=7)
    ax.set_ylabel("% of truth chains with ALL daughters correct"); ax.set_ylim(0, 100)
    ax.set_title("Chain-level agreement with truth"); ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    ok_o, ok_m, gain, loss = [], [], [], []
    for _, d, _ in dfs:
        o = (d["ours"] == d["true"]); m = (d["minip"] == d["true"])
        ok_o.append(100 * o.mean()); ok_m.append(100 * m.mean())
        gain.append(100 * (o & ~m).mean()); loss.append(100 * (~o & m).mean())
    both_wrong = [100 - ok_o[i] - gain[i] for i in range(len(dfs))]
    ax.bar(x, both_wrong, 0.5, color="tab:grey", label="both wrong")
    ax.bar(x, loss, 0.5, bottom=both_wrong, color="tab:red", label="minIP only")
    ax.bar(x, gain, 0.5, bottom=[both_wrong[i] + loss[i] for i in range(len(dfs))],
           color="tab:blue", label="ours only")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7, rotation=10)
    ax.set_ylim(0, 100); ax.set_ylabel("% of track slots")
    ax.set_title("Where the difference comes from"); ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    conc = []
    for _, d, _ in dfs:
        conc.append([chain_stat(d, "ours", "unan"), chain_stat(d, "minip", "unan")])
    ax.bar(x - 0.2, [c[0] for c in conc], 0.4, color="tab:blue", label="ours (HGNN)")
    ax.bar(x + 0.2, [c[1] for c in conc], 0.4, color="tab:red", alpha=.6, hatch="//", label="minIP")
    for i in range(len(dfs)):
        ax.text(i - 0.2, conc[i][0] + 1, f"{conc[i][0]:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, conc[i][1] + 1, f"{conc[i][1]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7, rotation=10)
    ax.set_ylim(0, 100); ax.set_ylabel("% of chains whose daughters share one PV")
    ax.set_title("Chain concordance (physics needs ~100%)"); ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    fig.suptitle("chain-level view of PV association (all truth-chain daughters)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/pv_bias_chain_en.png", dpi=130); plt.close(fig)
    print("wrote pv_bias_chain_en.png")


if __name__ == "__main__":
    fig_physics()
    fig_direction()
    fig_bias()
