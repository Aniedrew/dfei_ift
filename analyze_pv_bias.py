"""PV association bias: where does our HGNN systematically differ from traditional minIP?"

数据来源: 各次 eval 保存的 signal_reco_df_*.csv (逐真值链一行, 内含逐子径迹的
          true_pv / pred_pv / minIP_pv / pred_pv_b_lvl)。

比较的四种决策:
  1. ours      = HGNN 逐径迹 argmax (tt-pv 边权重)
  2. minIP     = 传统方法: 逐径迹独立取 log_minIP 最小的 PV
  3. minIP+vote= 传统方法 + 链内多数票 (给传统方法补上"同一条 B 链应共享 PV"的约束)
  4. ours(chain)= 我们模型的链级决策 (对链内所有径迹的分数求和再 argmax)

输出的偏置量:
  * 占用偏置: 各 PV index 的占用率 (真值 / ours / minIP) —— 谁把径迹推向低 index / PV0
  * 条件正确率: 按**真值 PV index**分组 (minIP 是否在高 index 更差?)
  * 堆积依赖: 按事件 PV 数分组
  * 误差方向: 错配径迹的 Δ = pred - true 分布 (向低 index 偏还是向高 index 偏)
  * 链级一致性: 一条链是否被全部指到同一个 PV; 以及"全对"的比例
  * 增益/损失分解: ours 对而 minIP 错 / 反之

用法: python analyze_pv_bias.py [--allp]   (--allp: 只用 AllParticles=True 的链)
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "report_figs"
NPV_BINS = [0, 4, 5, 6, 7, 8, 99]

# (标签, csv, 颜色, 线型)
CONFIGS = [
    ("v601 / 0904 incl (new-MC trained)",
     "LHCb_logs/DFEI/version_601/signal_reco_df_inclusive_00342451__v601_0904.csv", "tab:blue", "-"),
    ("v557 / 0702 July (arm A)",
     "LHCb_logs/DFEI/version_557/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armA_0702.csv", "tab:red", "-"),
    ("v557 / 0904 renorm (arm C)",
     "LHCb_logs/DFEI/version_557/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armC_renorm.csv", "tab:green", "-"),
]
EXTRA = [
    ("v47 / 0702 July (arm A)",
     "LHCb_logs/DFEI/version_47/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armA_0702.csv"),
    ("v47 / 0904 renorm (arm C)",
     "LHCb_logs/DFEI/version_47/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armC_renorm.csv"),
    ("v38 / 0702 July (arm A)",
     "LHCb_logs/DFEI/version_38/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armA_0702.csv"),
    ("v38 / 0904 renorm (arm C)",
     "LHCb_logs/DFEI/version_38/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armC_renorm.csv"),
    ("v31 / 0702 July (arm A)",
     "LHCb_logs/DFEI/version_31/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armA_0702.csv"),
    ("v31 / 0904 renorm (arm C)",
     "LHCb_logs/DFEI/version_31/signal_reco_df_inclusive_BsToJpsiPhi_00342629__armC_renorm.csv"),
]


def parse(s):
    return np.array([int(x) for x in str(s).split("_")], dtype=int)


def load_tracks(path, allp_only):
    """展开成逐(链,径迹)记录: true, ours, minip, blvl, npvs, chain_id, chain_len"""
    d = pd.read_csv(path)
    if allp_only and "AllParticles" in d:
        d = d[d["AllParticles"] == 1]
    rows = []
    for ci, r in enumerate(d.itertuples(index=False)):
        t, p, m = parse(r.true_pv), parse(r.pred_pv), parse(r.minIP_pv)
        if not (len(t) == len(p) == len(m)) or len(t) == 0:
            continue
        bl = int(r.pred_pv_b_lvl)
        # 链级决策 = b_lvl: 所有径迹都给同一个 PV
        b = np.full(len(t), bl)
        for k in range(len(t)):
            rows.append((ci, k, t[k], p[k], m[k], b[k], int(r.npvs), len(t), int(r.AllParticles)))
    df = pd.DataFrame(rows, columns=["chain", "pos", "true", "ours", "minip", "blvl", "npvs", "chain_len", "allp"])
    # 传统方法 + 链内多数票 (给 minIP 补上"同一条 B 链共享 PV"的约束)
    df["minip_vote"] = df.groupby("chain")["minip"].transform(lambda s: s.mode().iloc[0])
    df["ours_vote"] = df.groupby("chain")["ours"].transform(lambda s: s.mode().iloc[0])
    return df


def chain_vote(df, col):
    """链内多数票 -> 每径迹的链级决策"""
    return df.groupby("chain")[col].transform(lambda s: s.mode().iloc[0]).to_numpy()


def summarize(df, tag):
    df = df.copy()
    df["minip_vote"] = chain_vote(df, "minip")
    df["ours_vote"] = chain_vote(df, "ours")
    nt = len(df)
    nch = df["chain"].nunique()

    def acc(col):
        return 100 * (df[col] == df["true"]).mean()

    dfv0 = df[df["true"] >= 0]              # 剔除"无真值 PV"哨兵 (占用/方向统计只用这个子集)
    occ_true = dfv0["true"].value_counts(normalize=True).sort_index()
    occ_ours = dfv0["ours"].value_counts(normalize=True).sort_index()
    occ_mip = dfv0["minip"].value_counts(normalize=True).sort_index()

    out = {
        "tag": tag, "n_tracks": nt, "n_chains": nch,
        "acc_ours": acc("ours"), "acc_minip": acc("minip"),
        "acc_minip_vote": acc("minip_vote"), "acc_ours_blvl": acc("blvl"),
        "p_true_pv0": 100 * occ_true.get(0, 0), "p_ours_pv0": 100 * occ_ours.get(0, 0),
        "p_minip_pv0": 100 * occ_mip.get(0, 0),
        "meanidx_true": dfv0["true"].mean(), "meanidx_ours": dfv0["ours"].mean(),
        "meanidx_minip": dfv0["minip"].mean(),
    }
    # 误差方向 (只在错配的径迹上; 用剔除哨兵后的子集, 否则 minIP 的 -1 会污染 Δ)
    wo = dfv0[dfv0["ours"] != dfv0["true"]]
    wm = dfv0[dfv0["minip"] != dfv0["true"]]
    out["ourswrong_dmean"] = (wo["ours"] - wo["true"]).mean() if len(wo) else np.nan
    out["minipwrong_dmean"] = (wm["minip"] - wm["true"]).mean() if len(wm) else np.nan
    out["ourswrong_pct_lower"] = 100 * ((wo["ours"] < wo["true"]).mean()) if len(wo) else np.nan
    out["minipwrong_pct_lower"] = 100 * ((wm["minip"] < wm["true"]).mean()) if len(wm) else np.nan
    # ---- 弃权(哨兵)口径: true = -1 表示"该径迹没有真值 PV"(未关联/假径迹)。
    #      minIP 基线经 pv_filter 后可以给出 -1(弃权), 我们的模型只会 argmax -> 永远弃权不了。
    sent = df["true"] < 0
    out["sentinel_pct"] = 100 * sent.mean()
    dfv = df[~sent]
    if len(dfv):
        for col in ["ours", "minip", "minip_vote", "blvl"]:
            out[f"accv_{col}"] = 100 * (dfv[col] == dfv["true"]).mean()
        out["accv_n_tracks"] = len(dfv)
    # 增益/损失
    out["gain_ours_only"] = 100 * ((df["ours"] == df["true"]) & (df["minip"] != df["true"])).mean()
    out["loss_minip_only"] = 100 * ((df["ours"] != df["true"]) & (df["minip"] == df["true"])).mean()
    # 链级: 全对 / 全体一致
    for col, nm in [("ours", "ours"), ("minip", "minip"), ("minip_vote", "minip_vote"), ("blvl", "ours_chain")]:
        allok = df.groupby("chain").apply(lambda g, c=col: (g[c] == g["true"]).all())
        unan = df.groupby("chain").apply(lambda g, c=col: g[c].nunique() == 1)
        out[f"chain_allok_{nm}"] = 100 * allok.mean()
        out[f"chain_unan_{nm}"] = 100 * unan.mean()
    unan_true = df.groupby("chain").apply(lambda g: g["true"].nunique() == 1)
    out["chain_unan_truth"] = 100 * unan_true.mean()
    return out, occ_true, occ_ours, occ_mip


def acc_by(df, col, key, bins=None):
    tmp = pd.DataFrame({"v": df[col].to_numpy(), "t": df["true"].to_numpy(), "k": df[key].to_numpy()})
    return tmp.groupby("k").apply(lambda s: 100 * (s["v"] == s["t"]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allp", action="store_true", help="只用 AllParticles 链")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows, occs, bys = [], {}, {}
    for label, path, color, ls in CONFIGS:
        if not os.path.exists(path):
            print("缺文件:", path)
            continue
        df = load_tracks(path, a.allp)
        s, ot, oo, om = summarize(df, label)
        rows.append(s)
        occs[label] = (ot, oo, om)
        dfv = df[df["true"] >= 0]           # 去掉"无真值 PV"的哨兵径迹 (曲线用)
        y = acc_by(dfv, "ours", "true").rename("ours")
        z = acc_by(dfv, "minip", "true").rename("minip")
        yv = acc_by(dfv, "minip_vote", "true").rename("minip_vote")
        bys[label] = pd.concat([y, z, yv], axis=1)
        print(f"\n[{label}] tracks={s['n_tracks']} chains={s['n_chains']}")
        print(f"  正确率: ours {s['acc_ours']:.2f} | minIP {s['acc_minip']:.2f} | minIP+多数票 {s['acc_minip_vote']:.2f} "
              f"| ours链级 {s['acc_ours_blvl']:.2f}")
        print(f"  弃权哨兵占比 {s['sentinel_pct']:.2f}% | 剔除哨兵后: ours {s.get('accv_ours', float('nan')):.2f} "
              f"| minIP {s.get('accv_minip', float('nan')):.2f} | minIP+多数票 {s.get('accv_minip_vote', float('nan')):.2f} "
              f"| ours链级 {s.get('accv_blvl', float('nan')):.2f}")
        print(f"  PV0 占用: 真值 {s['p_true_pv0']:.2f}% ours {s['p_ours_pv0']:.2f}% minIP {s['p_minip_pv0']:.2f}%")
        print(f"  平均 index: 真值 {s['meanidx_true']:.3f} ours {s['meanidx_ours']:.3f} minIP {s['meanidx_minip']:.3f}")
        print(f"  错配方向 mean(Δ): ours {s['ourswrong_dmean']:+.3f} (向低 index {s['ourswrong_pct_lower']:.1f}%) | "
              f"minIP {s['minipwrong_dmean']:+.3f} (向低 {s['minipwrong_pct_lower']:.1f}%)")
        print(f"  链级全对: ours {s['chain_allok_ours']:.1f} | minIP {s['chain_allok_minip']:.1f} "
              f"| minIP+多数票 {s['chain_allok_minip_vote']:.1f} | ours链级 {s['chain_allok_ours_chain']:.1f}")
        print(f"  链内一致: 真值 {s['chain_unan_truth']:.1f} | ours {s['chain_unan_ours']:.1f} | minIP {s['chain_unan_minip']:.1f}")

    for label, path in EXTRA:
        if os.path.exists(path):
            rows.append(summarize(load_tracks(path, a.allp), label)[0])

    summ = pd.DataFrame(rows)
    p = f"{OUT}/pv_bias_summary{'_allp' if a.allp else ''}.csv"
    summ.to_csv(p, index=False)
    print("\n写出", p)

    # ---------- 图 1: 占用偏置 + 条件正确率 ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for j, (label, _, color, ls) in enumerate(CONFIGS):
        if label not in occs:
            continue
        ot, oo, om = occs[label]
        ax = axes[0, j]
        idx = np.arange(0, 8)
        w = 0.27
        ax.bar(idx - w, [100 * ot.get(i, 0) for i in idx], w, label="truth", color="grey")
        ax.bar(idx, [100 * oo.get(i, 0) for i in idx], w, label="ours (HGNN)", color=color)
        ax.bar(idx + w, [100 * om.get(i, 0) for i in idx], w, label="minIP", color=color, alpha=.45, hatch="//")
        ax.set_title(f"{label}\nPV index occupancy", fontsize=9)
        ax.set_xlabel("PV index"); ax.set_ylabel("% of track-slots")
        ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

        by = bys[label]
        ax = axes[1, j]
        ax.plot(by.index, by["ours"], "o-", color=color, label="ours (HGNN)")
        ax.plot(by.index, by["minip"], "s--", color=color, alpha=.5, label="minIP")
        ax.plot(by.index, by["minip_vote"], "^:", color="black", alpha=.6, label="minIP + chain vote")
        ax.set_title("accuracy vs TRUE PV index", fontsize=9)
        ax.set_xlabel("true PV index"); ax.set_ylabel("accuracy %")
        ax.set_ylim(0, 102); ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.suptitle("PV association bias: ours vs traditional minIP   (signal-chain tracks)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = f"{OUT}/pv_bias_occupancy{'_allp' if a.allp else ''}_en.png"
    fig.savefig(p1, dpi=130)
    print("写出", p1)

    # ---------- 图 2: 链级一致性 + 增益/损失 + 堆积依赖 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = [l for l, *_ in CONFIGS if l in occs]
    x = np.arange(len(labels))
    s = summ.set_index("tag")
    ax = axes[0]
    w = 0.2
    for k, (col, nm, c) in enumerate([("chain_allok_ours", "ours (per-track)", "tab:blue"),
                                      ("chain_allok_ours_chain", "ours (chain-level)", "tab:cyan"),
                                      ("chain_allok_minip", "minIP", "tab:red"),
                                      ("chain_allok_minip_vote", "minIP + chain vote", "tab:orange")]):
        ax.bar(x + (k - 1.5) * w, [s.loc[l, col] for l in labels], w, label=nm, color=c)
    ax.set_xticks(x); ax.set_xticklabels([l.split(" / ")[0] + "\n" + l.split(" / ")[1] for l in labels], fontsize=7)
    ax.set_ylabel("% of truth chains with ALL daughters correct")
    ax.set_title("Chain-level agreement with truth"); ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    ok_t = [s.loc[l, "acc_ours"] for l in labels]
    ok_m = [s.loc[l, "acc_minip"] for l in labels]
    gain = [s.loc[l, "gain_ours_only"] for l in labels]
    loss = [s.loc[l, "loss_minip_only"] for l in labels]
    bot = [ok_m[i] - loss[i] for i in range(len(labels))]
    both_wrong = [100 - ok_m[i] - gain[i] for i in range(len(labels))]
    ax.bar(x, bot, color="tab:grey", label="both wrong")
    ax.bar(x, loss, bottom=bot, color="tab:red", label="minIP only")
    ax.bar(x, gain, bottom=[bot[i] + loss[i] for i in range(len(labels))], color="tab:blue", label="ours only")
    ax.set_xticks(x); ax.set_xticklabels([l.split(" / ")[0] for l in labels], fontsize=8)
    ax.set_ylabel("% of track-slots"); ax.set_title("Where the difference comes from")
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    for label, path, color, ls in CONFIGS:
        if not os.path.exists(path):
            continue
        df = load_tracks(path, a.allp)
        df["npvbin"] = pd.cut(df["npvs"], bins=NPV_BINS)
        y = df.groupby("npvbin", observed=True).apply(lambda g: pd.Series({
            "ours": 100 * (g["ours"] == g["true"]).mean(),
            "minip": 100 * (g["minip"] == g["true"]).mean()}))
        ax.plot([str(i) for i in y.index], y["ours"], "o-", color=color, label=label.split(" (")[0] + " ours")
        ax.plot([str(i) for i in y.index], y["minip"], "s--", color=color, alpha=.5, label=label.split(" (")[0] + " minIP")
    ax.set_xlabel("nPVs in event"); ax.set_ylabel("accuracy %")
    ax.set_title("Pile-up dependence"); ax.legend(fontsize=6); ax.grid(alpha=.25)
    plt.setp(ax.get_xticklabels(), rotation=20, fontsize=7)
    fig.tight_layout()
    p2 = f"{OUT}/pv_bias_chain{'_allp' if a.allp else ''}_en.png"
    fig.savefig(p2, dpi=130)
    print("写出", p2)


if __name__ == "__main__":
    main()
