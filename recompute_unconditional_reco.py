"""无条件（固定分母）重建指标 —— 由本人重写。

背景（为什么要这个脚本）
------------------------
上游 `wmpgnn/analysis/reco_accuracy.py::calculate_accuracy`：
    all_particles_ratio = (#AllParticles) / len(df) * 100
df = `signal_reco_df_*.csv` 的**逐真值链行**，它是在**已剪枝的图**上解的
（reconstruction.py L722-728: particle_keys = get_truth_part_keys(graph) 取剪枝后幸存径迹，
 再由 reconstruct_decay(true_LCA, ...) 得 tc_dict，每条目一行）。于是：
  * 分母 = "剪枝后仍存在（且已被截断）的真值链数"，**随模型/阈值变化**
    -> 剪得越狠、分母越小、百分比越虚高（实测 0904 线 v651 N=6089 vs v655 N=9516）；
  * 分子也宽松：rc.node_keys == tc.node_keys 比的是"幸存部分"，
    一条被剪掉若干径迹的链只要把剩下的部分拼出来，照样算 AllParticles=1。

本脚本给**无条件版本**（与上游 recompute_fixed_denominator.py 同口径）：
    All_fix%  = (#AllParticles) / N_total
    Perf_fix% = (#PerfectReco) / N_total
N_total = **完整图（不剪枝）**上的真值链总数，与模型无关的常数，
由 `config_files/eval_CERN_noprune.yaml`（0702）/ `eval_CERN_0904_noprune.yaml`（0904）产出。
被剪掉、或事件整个没进评估的链，一律计失败。另补两项稳健诊断：
  * 分层固定分母：按事件真值链数 1B / 2B / >2B（用真实 RUN/EVENT 号跨版本对齐）；
  * 真值径迹存活率 track_surv% = ΣNumSignalParticles(版本) / ΣNumSignalParticles(noprune)
    —— 这是"被剪枝截断了多少真值内容"的无偏度量；All_fix% 天然是它的上界。

用法
----
    python3 recompute_unconditional_reco.py --line 0904   # 非泄漏数据（结论以此为准）
    python3 recompute_unconditional_reco.py --line 0702   # 含顺序泄漏，仅对照
"""
import argparse
import glob
import os
import re

import pandas as pd

ROOT = "LHCb_logs/DFEI"
LINES = {
    "0904": dict(sample="00342451", known_n_total=17561 * 0),   # N_total 由 noprune 运行给出
    "0702": dict(sample="00342442", known_n_total=17561),
}
SKIP = re.compile(r"smoke|small|paper|noprune|ckpt-scan")


def load_reference(cfg):
    """参照（完整图真值链）优先取本人 count_truth_chains.py 的产物；
    没有则回退到 noprune 评估的 signal_reco_df。两者在 0702 上互相校验一致。"""
    sample = cfg["sample"]
    mine = f"report_figs/truth_chain_total_inclusive_{sample}.csv"
    if os.path.exists(mine):
        ref = pd.read_csv(mine)
        path = mine
    else:
        pats = [f"{ROOT}/version_*/signal_reco_df_*{sample}*noprune*.csv"]
        cands = sorted({c for p in pats for c in glob.glob(p)})
        if not cands:
            raise SystemExit(
                f"[{sample}] 缺参照。请先跑 count_truth_chains.py，或运行:\n"
                f"  python3 -u wmpgnn/analysis/evaluate.py --config config_files/"
                f"eval_CERN_{'0904_' if sample == '00342451' else ''}noprune.yaml"
            )
        path = cands[0]
        ref = pd.read_csv(path)
    if "RUNNUMBER" not in ref.columns or "EVENTNUMBER" not in ref.columns:
        raise SystemExit(f"参照文件缺 RUNNUMBER/EVENTNUMBER: {path}")
    ref = ref.assign(_k=list(zip(ref["RUNNUMBER"].astype("int64"),
                                 ref["EVENTNUMBER"].astype("int64"))))
    per_evt = ref.groupby("_k")["n_truth_chains"].sum() if "n_truth_chains" in ref.columns \
        else ref.groupby("_k").size()
    ref_tracks = float(ref["n_truth_tracks"].sum()) if "n_truth_tracks" in ref.columns \
        else float(ref["NumSignalParticles"].sum())
    return path, per_evt, ref_tracks, ref


def strat(n):
    return "1B" if n == 1 else ("2B" if n == 2 else ">2B")


def recompute(cfg):
    path, per_evt, ref_tracks, ref = load_reference(cfg)
    # 参照文件两种格式: 本人产物(逐事件, 需 sum) / noprune 评估(逐真值链, 行数即链数)
    n_total = int(ref["n_truth_chains"].sum()) if "n_truth_chains" in ref.columns else len(ref)
    print(f"noprune 参照: {os.path.basename(path)}")
    print(f"N_total (完整图真值链总数 = 固定分母) = {n_total}")
    if cfg.get("known_n_total"):
        ok = "✓ 与史料一致" if cfg["known_n_total"] == n_total else f"⚠ 史料={cfg['known_n_total']}"
        print(f"  交叉核对: {ok}")
    s_evt = per_evt.map(lambda n: "1B" if n == 1 else ("2B" if n == 2 else (">2B" if n > 2 else "0链")))
    d = per_evt.groupby(s_evt).sum().to_dict()      # 每层"真值链总数" = 该层分母
    denom = {k: int(d.get(k, 0)) for k in ("1B", "2B", ">2B")}
    print(f"分层分母(真值链数): 1B={denom['1B']}  2B={denom['2B']}  >2B={denom['>2B']}")
    print(f"完整图真值径迹总数 = {ref_tracks:.0f}\n")

    rows = []
    for csv in sorted(glob.glob(f"{ROOT}/version_*/signal_reco_df_*{cfg['sample']}*.csv")):
        tag = os.path.basename(csv).split("__", 1)[-1][:-4]
        if SKIP.search(tag):
            continue
        m = re.search(r"version_(\d+)", csv)
        if not m:
            continue
        version = int(m.group(1))
        df = pd.read_csv(csv)
        if len(df) == 0 or "RUNNUMBER" not in df.columns:
            continue
        df = df.assign(_k=list(zip(df["RUNNUMBER"].astype("int64"),
                                   df["EVENTNUMBER"].astype("int64"))))
        nb = df["_k"].map(per_evt)
        df = df.assign(_str=nb.map(lambda x: strat(x) if pd.notna(x) else "?"))
        ap, pr = int(df["AllParticles"].sum()), int(df["PerfectReco"].sum())
        rec = {
            "version": version, "tag": tag,
            "events": df["_k"].nunique(), "events_total": len(per_evt),
            "rows(=评估内真值链)": len(df), "N_total": n_total,
            "All#": ap, "Perfect#": pr,
            "All_orig%": round(100 * ap / len(df), 2),
            "Perf_orig%": round(100 * pr / len(df), 2),
            "All_fix%": round(100 * ap / n_total, 2),
            "Perf_fix%": round(100 * pr / n_total, 2),
            "track_surv%": round(100 * df["NumSignalParticles"].sum() / ref_tracks, 1),
        }
        for s in ("1B", "2B", ">2B"):
            sub = df[df["_str"] == s]
            rec[f"All#{s}"] = int(sub["AllParticles"].sum())
            rec[f"All_fix%_{s}"] = round(100 * rec[f"All#{s}"] / denom[s], 2) if denom[s] else float("nan")
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("version").reset_index(drop=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(out.to_string(index=False))
    op = f"report_figs/unconditional_reco_{cfg['sample']}.csv"
    out.to_csv(op, index=False)
    print(f"\n已写出: {op}")
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--line", default="0904", choices=list(LINES))
    recompute(LINES[a.parse_args().line])
