"""阶段对比: 从 yukai 原始版 -> GNN 优化 -> bug 修复 -> pruning 优化 -> 输出优化。

每个阶段选代表性版本, 给"尽可能全面"的指标。所有指标都在**各自所属数据线**的口径下计算:
  0702 线(含 delta_z0 泄漏) N_total=17561 ; 0904 线(泄漏已修) N_total=13255
=> 两条线之间只能看"趋势", 不能直接相减(数据集变更 + 泄漏修复)。
在本线内部, 固定分母的 all_fix% 与 ε_found% 是可跨版本比较的。

输出: report_figs/stage_comparison.csv  +  控制台表格
用法: python3 make_stage_comparison.py
"""
import glob
import os
import re

import pandas as pd

ROOT = "LHCb_logs/DFEI"
LINES = {"0702": dict(sample="00342442", n_total=17561),
         "0904": dict(sample="00342451", n_total=13255)}

# 阶段 -> 代表版本
STAGES = [
    ("S1 原始(yukai 论文版)", ["8", "31"]),
    ("S2 我们优化 GNN", ["47", "557"]),
    ("S3 修 bug(种子+DOCA)", ["643"]),
    ("S4 优化 pruning", ["648", "651"]),
    ("S5 优化输出(当前)", ["654", "670", "667", "1201"]),
]
NOTE = {
    "8": "最早版本", "31": "论文基线(旧线的参照)",
    "47": "v31→v47: 图结构与特征", "557": "上次组会时的冠军",
    "643": "首个 0904 + 种子/DOCA 全修复",
    "648": "竞争/排他上下文(硬池 AP 最高)",
    "651": "线图注意力+几何 bias(干净端最好)",
    "654": "chain_recall w=5(召回旋钮)",
    "670": "context_prune 注意力上下文(+6.91)",
    "667": "edge_rank ×2(+4.99)",
    "1201": "blend_a = 654+ctx+rank(+7.63, 当前最好)",
}


def pick(ver, sample):
    """选代表性评估文件: 0702 线优先'无 __ 后缀'的朴素评估; 0904 线优先 v<ver>_ 前缀。"""
    files = sorted(glob.glob(f"{ROOT}/version_{ver}/signal_reco_df_*{sample}*.csv"))
    if not files:
        return None, None
    plain = [f for f in files if "__" not in os.path.basename(f)]
    if plain:
        f = plain[0]
    else:
        pref = [f for f in files if os.path.basename(f).split("__")[-1].startswith(f"v{ver}_")]
        f = pref[0] if pref else files[0]
    tag = os.path.basename(f).split("__")[-1][:-4]
    return f, tag


def pv_and_lcag(ver, tag, sample):
    """从同 tag 的 info_*_reco.txt 里取 PV 关联与 LCAG 对角。
    旧版本(0702 线)的评估文件常无 __tag 后缀 -> 回退到该目录下该样本的任意 info 文件。"""
    fs = glob.glob(f"{ROOT}/version_{ver}/info_*__{tag}_reco.txt")
    if not fs:
        fs = glob.glob(f"{ROOT}/version_{ver}/info_*{sample}*reco.txt")
    if not fs:
        return {}, {}
    t = open(fs[0], errors="ignore").read()
    pv = {}
    seg = t.split("PV association performance")[-1]
    cur = None
    for line in seg.split("\n"):
        s = line.strip()
        if s.endswith(":") and "association" not in s:
            cur = s[:-1]
        m = re.match(r"(HGNN|minIP) association\s*:\s*([\d.]+)", s)
        if m and cur:
            pv.setdefault(cur, {})[m.group(1)] = float(m.group(2))
    lc = {}
    for i in range(4):
        m = re.search(rf"LCA_class{i} \(n=([\d.]+)\):\s*pred0=([\d.]+)%\s*pred1=([\d.]+)%"
                      rf"\s*pred2=([\d.]+)%\s*pred3=([\d.]+)%", t)
        if m:
            lc[i] = float(m.group(i + 2))
    return pv, lc


def ref_tracks(line, sample):
    f = f"report_figs/truth_chain_total_inclusive_{sample}.csv"
    if os.path.exists(f):
        return float(pd.read_csv(f)["n_truth_tracks"].sum())
    return float("nan")


def main():
    bench = {}
    for r in pd.read_csv("report_figs/bench_auc_v3_seed.csv").to_dict("records"):
        m = re.search(r"v(\d+)", str(r.get("tag", "")))
        if m and str(r.get("events")) == "3000":
            bench.setdefault(int(m.group(1)), (r.get("ap_hardpool"), r.get("chain_surv")))
    rows = []
    for stage, vers in STAGES:
        for ver in vers:
            line = "0702" if int(ver) < 601 else "0904"
            cfg = LINES[line]
            f, tag = pick(ver, cfg["sample"])
            if f is None:
                rows.append(dict(stage=stage, ver=ver, note=NOTE.get(ver, ""), status="无评估"));
                continue
            d = pd.read_csv(f)
            A = int(d["AllParticles"].sum()); N = len(d)
            F = A + int(d["NoneIso"].sum())
            pv, lc = pv_and_lcag(ver, tag, cfg["sample"])
            b = bench.get(int(ver), (None, None))
            rows.append(dict(
                stage=stage, ver=ver, note=NOTE.get(ver, ""), line=line, status="ok",
                N_raw=N, all_raw_pct=round(100 * A / N, 2),
                all_fix_pct=round(100 * A / cfg["n_total"], 2),
                found_pct=round(100 * F / cfg["n_total"], 2),
                ap_hardpool=b[0], chain_surv=b[1],
                pv_hgnn=pv.get("all_tracks", {}).get("HGNN"),
                pv_minip=pv.get("all_tracks", {}).get("minIP"),
                lcag_c1=lc.get(1), lcag_c2=lc.get(2),
                track_surv_pct=round(100 * float(d["NumSignalParticles"].sum()) / ref_tracks(line, cfg["sample"]), 1),
            ))
    df = pd.DataFrame(rows)
    df.to_csv("report_figs/stage_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["stage", "ver", "note", "line", "N_raw", "all_raw_pct", "all_fix_pct", "found_pct",
            "ap_hardpool", "chain_surv", "pv_hgnn", "pv_minip", "lcag_c1", "lcag_c2"]
    print(df[cols].to_string(index=False))
    print("\n注意: 0702/0904 之间不可直接相减(数据集变更+泄漏修复); 线内 all_fix% 与 found% 可横比。")


if __name__ == "__main__":
    main()
