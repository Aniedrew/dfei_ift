"""构建"全版本 × 全 benchmark"的统一评估矩阵 (只读, 只新增产物).

产出: report_figs/eval_matrix_all_versions.csv

每行 = 一个训练版本 (LHCb_logs/DFEI/version_*), 列为:
  * 身份: version / tag / reco_file / dataset / line / completion_date / cross_break
  * 旧 benchmark (论文口径, 来自 info_*_reco.txt 的 Reconstruction performance 段):
      all_particles / perfect_reco / none_iso / part_reco  (整体 + 1B/2B/>2B 分层, 含 ± 误差)
  * PV 关联: HGNN vs minIP (整体 all_tracks + sig_AllParticles), 及两者差值
  * LCAG 混淆: class0..3 的对角线正确率 (从 4x4 表算)
  * 新 benchmark (report_figs/bench_auc_v3_seed.csv, 阈值无关): ap_hardpool / p_at_r90_hardpool /
      chain_surv / chain_surv_2 / chain_surv_3 / per_link_p_2 —— 按事件数 (200/500/3000) 分列
  * 端到端: chain_full (report_figs/seqrec_offline_v654_thr0.9_all.csv; 只 v654 有)

缺口 / 口径不确定性一律留空, 不编造.

用法:
  export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
  python build_eval_matrix.py
"""
import os
import re
import glob
import csv
from datetime import datetime

import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(REPO, "LHCb_logs", "DFEI")
OUT = os.path.join(REPO, "report_figs", "eval_matrix_all_versions.csv")
BENCH = os.path.join(REPO, "report_figs", "bench_auc_v3_seed.csv")
E2E = os.path.join(REPO, "report_figs", "seqrec_offline_v654_thr0.9_all.csv")

# ---------------------------------------------------------------------------
# 评估口径的三个断点 (来源见 EVALUATION_PROTOCOL.md / build 报告)
#   B1 = 数据集切换 0702 -> 0904 (同时修掉 delta_z0 顺序泄漏), 边界版本 v601
#   B2 = 评测事件抽样加种子 (跨版本同池可比), 边界版本 v642 (commit 317ca5b, 2026-10-05)
#   B3 = 本地"修正 DOCA 几何"派生量 (绕开上游 calculate_doca 的 t1 符号 bug), 边界版本 v643
# ---------------------------------------------------------------------------
BREAKPOINTS = {
    "B1_dataset_0702to0904": dict(version=601, date="2026-09-20",
                                  label="dataset switch 0702->0904 (leak fix)"),
    "B2_eval_seed_fix": dict(version=642, date="2026-10-05",
                             label="eval event-sampling seed fix"),
    "B3_doca_geom_fix": dict(version=643, date="2026-10-05",
                             label="local corrected DOCA geometry"),
}

# 非"主输出"tag 记号: 出现在变体/重评/探针文件名里, 优先排除
NONPRIMARY = ["smoke", "ctrl", "perm", "lrprobe", "join", "stack", "drift",
              "ckptscan", "record", "reeval", "public", "renorm", "oracle",
              "truth", "off", "noprune", "node0", "node1", "node2",
              "edge0", "edge1", "armA", "armB", "armC", "iso_", "cpub", "topk",
              "_batch_", "pvcluster", "_da_", "overlap", "_mb"]

RECO_RE = re.compile(
    r"^info_(?:inclusive_(?P<ds1>\d{8})|(?P<ds2>\d{8})_inclusive)"
    r"(?:__(?P<tag>.+?))?_reco\.txt$")

NUM_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*([-\d.]+|nan|inf)\s*\+/-\s*([-\d.]+|nan|inf)\s*$")


def _num(s):
    if s is None:
        return None
    s = s.strip()
    if s in ("nan", "inf", "-inf", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_reco(path):
    """解析一份 info_*_reco.txt -> dict."""
    res = {"events": None,
           "all_particles": None, "all_particles_err": None,
           "perfect_reco": None, "perfect_reco_err": None,
           "none_iso": None, "none_iso_err": None,
           "part_reco": None, "part_reco_err": None}
    tiers = {"1B": dict(n=None, all_particles=None, all_particles_err=None,
                        perfect_reco=None, perfect_reco_err=None,
                        none_iso=None, none_iso_err=None),
             "2B": dict(n=None, all_particles=None, all_particles_err=None,
                        perfect_reco=None, perfect_reco_err=None,
                        none_iso=None, none_iso_err=None),
             "gt2B": dict(n=None, all_particles=None, all_particles_err=None,
                          perfect_reco=None, perfect_reco_err=None,
                          none_iso=None, none_iso_err=None)}

    with open(path) as f:
        lines = f.readlines()

    cur = None          # 当前 Reconstruction efficiency 分层
    pv_group = None     # 当前 PV 关联分组
    sections = {"reco": False, "lcag": False, "pv": False}

    def set_reco_field(store, key, val, err):
        store[key] = val
        store[key + "_err"] = err

    for ln in lines:
        st = ln.strip()
        if "Reconstruction performance" in st:
            sections["reco"] = True
            continue
        if "LCAG classification accuracy" in st:
            sections["lcag"] = True
            sections["reco"] = False
            continue
        if "PV association performance" in st:
            sections["pv"] = True
            sections["lcag"] = False
            continue

        m = re.match(r"^Reconstruction efficiency(?:\s+(.+?))?\s*\((\d+)\)\s*:?\s*$", st)
        if m:
            band = (m.group(1) or "").replace("only ", "").strip()
            n = int(m.group(2))
            if band == "":
                cur = "overall"
                res["events"] = n
            elif "more than" in band:
                cur = "gt2B"
                tiers["gt2B"]["n"] = n
            elif "1 B" in band:
                cur = "1B"
                tiers["1B"]["n"] = n
            elif "2 B" in band:
                cur = "2B"
                tiers["2B"]["n"] = n
            else:
                cur = "other"     # negative id / positive id
            continue

        # LCA 分类表
        ml = re.match(r"^LCA_class(\d)\s*\(n=([\d.]+)\):\s*(.*)$", st)
        if ml:
            cls = int(ml.group(1))
            n = _num(ml.group(2))
            preds = {}
            for pm in re.finditer(r"pred(\d)=([\d.]+)%", ml.group(3)):
                preds[int(pm.group(1))] = float(pm.group(2))
            res[f"lcag_n_class{cls}"] = n
            res[f"lcag_acc_class{cls}"] = preds.get(cls)
            continue

        # PV 关联分组头
        mp = re.match(r"^(all_tracks|sig_[A-Za-z_]+):\s*$", st)
        if mp:
            pv_group = mp.group(1)
            continue
        if sections["pv"] and pv_group:
            mh = re.match(r"^(HGNN|minIP) association\s*:\s*(.*)$", st)
            if mh:
                who = "hgnn" if mh.group(1) == "HGNN" else "minip"
                rn = re.match(r"^\s*([-\d.]+|nan)\s*\+/-\s*([-\d.]+|nan)\s*$", mh.group(2))
                if rn:
                    res[f"pv_{who}_{pv_group}"] = _num(rn.group(1))
                    res[f"pv_{who}_{pv_group}_err"] = _num(rn.group(2))
                continue

        if sections["reco"] and cur and cur != "other":
            mn = NUM_RE.match(ln)
            if mn:
                name, val, err = mn.group(1), _num(mn.group(2)), _num(mn.group(3))
                if cur == "overall":
                    set_reco_field(res, name, val, err)
                else:
                    set_reco_field(tiers[cur], name, val, err)

    for k, t in tiers.items():
        for kk, vv in t.items():
            res[f"{kk}_{k}"] = vv
    return res


def pick_canonical_reco(files, version):
    """挑一个版本的代表性 reco 文件.

    规则 (按优先级, 全部只读):
      1) 少含"非主输出"记号 (smoke/ctrl/perm/oracle/thr/iso/topk/...);
      2) 最早写入的 (≈ 训练结束时的自带 test 输出);
      3) 再按文件名稳定排序.
    """
    def key(p):
        name = os.path.basename(p)
        low = name.lower()
        penalty = sum(1 for tok in NONPRIMARY if tok in low)
        return (penalty, os.path.getmtime(p), name)
    return sorted(files, key=key)[0]


def classify_line(version, tag, dataset):
    blob = f"{tag or ''}"
    if re.search(r"\bpart\b|part_", blob, re.I):
        return "direction-1 part"
    if "setpred" in blob.lower():
        return "direction-2 setpred"
    if "seqrec" in blob.lower():
        return "direction-3 seqrec"
    if "geomcand" in blob.lower():
        return "direction-4 geomcand"
    if version >= 900000 or "ckptscan" in blob.lower():
        return "ckpt-scan"
    if dataset == "0904":
        return "0904 (post-leakfix)"
    if dataset == "0702":
        return "0702 (leaky)"
    return "unknown"


def cross_break(version, dataset):
    hits = []
    if dataset == "0904" or version >= BREAKPOINTS["B1_dataset_0702to0904"]["version"]:
        hits.append("B1")
    if version >= BREAKPOINTS["B2_eval_seed_fix"]["version"]:
        hits.append("B2")
    if version >= BREAKPOINTS["B3_doca_geom_fix"]["version"]:
        hits.append("B3")
    return "+".join(hits) if hits else ""


def load_bench():
    """bench_auc_v3_seed.csv -> {version(str): {events(int): row-dict}} (去重)."""
    out = {}
    if not os.path.exists(BENCH):
        return out
    df = pd.read_csv(BENCH)
    df = df.drop_duplicates(subset=["tag", "version", "events"], keep="first")
    for _, r in df.iterrows():
        ev = int(r["events"])
        out.setdefault(str(int(r["version"])), {})[ev] = r
    return out


def load_e2e():
    """seqrec 离线端到端 -> {version: dict(chain_full, decoder, events)}."""
    if not os.path.exists(E2E):
        return {}
    df = pd.read_csv(E2E)
    # 取"learned"解码器里 chain_full 最高者为主, cc 为基线
    learned = df[df["decoder"].str.startswith("learned")].sort_values("chain_full", ascending=False)
    cc = df[df["decoder"] == "cc"]
    n_evt = int(df["n_evt"].iloc[0]) if len(df) else None
    d = {}
    if len(learned):
        d["e2e_chain_full"] = float(learned["chain_full"].iloc[0])
        d["e2e_decoder"] = str(learned["decoder"].iloc[0])
        d["e2e_events"] = n_evt
    if len(cc):
        d["e2e_chain_full_cc"] = float(cc["chain_full"].iloc[0])
    return {"654": d}


def main():
    bench = load_bench()
    e2e = load_e2e()

    rows = []
    version_dirs = sorted(glob.glob(os.path.join(LOGS, "version_*")))
    for d in version_dirs:
        base = os.path.basename(d)
        mv = re.match(r"version_(\d+)", base)
        if not mv:
            continue
        version = int(mv.group(1))
        files = sorted(glob.glob(os.path.join(d, "info_*reco*.txt")))
        if not files:
            continue
        canon = pick_canonical_reco(files, version)
        cname = os.path.basename(canon)
        m = RECO_RE.match(cname)
        if m:
            ds = m.group("ds1") or m.group("ds2")
            tag = m.group("tag") or "default"
        else:
            ds, tag = None, "unknown"
        dataset = {"00342442": "0702", "00342451": "0904"}.get(ds, "other")
        cdate = datetime.fromtimestamp(os.path.getmtime(canon)).strftime("%Y-%m-%d")

        rec = parse_reco(canon)
        row = {
            "version": version,
            "version_dir": base,
            "tag": tag,
            "reco_file": cname,
            "n_reco_files": len(files),
            "dataset": dataset,
            "line": classify_line(version, tag, dataset),
            "completion_date": cdate,
            "cross_break": cross_break(version, dataset),
        }
        row.update(rec)

        # PV 差值
        for grp in ("all_tracks", "sig_AllParticles"):
            h = rec.get(f"pv_hgnn_{grp}")
            mn = rec.get(f"pv_minip_{grp}")
            row[f"pv_diff_{grp}"] = (h - mn) if (h is not None and mn is not None) else None

        # 新 benchmark, 按事件数分列
        vbench = bench.get(str(version), {})
        row["bench_events"] = ",".join(str(e) for e in sorted(vbench)) if vbench else ""
        cols = {"ap_hardpool": "ap_hardpool", "p_at_r90_hardpool": "p_at_r90_hardpool",
                "chain_surv": "chain_surv", "chain_surv_2": "chain_surv_2",
                "chain_surv_3": "chain_surv_3", "per_link_p_2": "per_link_p_2",
                "ap_hardpool_lo": "ap_hardpool_lo", "ap_hardpool_hi": "ap_hardpool_hi",
                "hardpool_n": "hardpool_n", "thr": "thr"}
        for ev, r in vbench.items():
            for outk, src in cols.items():
                row[f"{outk}_{ev}"] = r.get(src)

        # 端到端 (只 654 有)
        ve2e = e2e.get(str(version), {})
        for k in ("e2e_chain_full", "e2e_chain_full_cc", "e2e_decoder", "e2e_events"):
            row[k] = ve2e.get(k)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["completion_date", "version"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  rows={len(df)}  cols={len(df.columns)}")
    print("versions with reco txt:", len(df))
    print("by dataset:\n", df["dataset"].value_counts())
    print("by line:\n", df["line"].value_counts())
    print("with bench data:", df["bench_events"].astype(bool).sum())
    print("with e2e:", df["e2e_chain_full"].notna().sum())


if __name__ == "__main__":
    main()
