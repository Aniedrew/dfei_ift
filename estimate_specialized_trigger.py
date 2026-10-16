"""基于现有数据估算"特化（单 B / 单一末态）触发"的预期效率。

思路（为什么这几个数能估出特化后的水平）
----------------------------------------
上游 `signal_reco_df` 每条真值链有 4 个互斥标志（`reconstruction.py` L776-822）：
  AllParticles = 重建链径迹集合 == 真值链 **完全相等**      -> 找到且干净
  NoneIso      = 真值径迹**全在一个重建链里**，但该链**多了背景径迹** -> **找到了，但有污染**
  PartReco     = 只找到一部分
  NotFound     = 没找到
因此：
  ε_clean = AllParticles / N_total        （我们一直在报的"严格"数）
  **ε_found = (AllParticles + NoneIso) / N_total**  ← "把这条链的径迹作为一个对象找到了"（不要求纯度）

**触发需要的正是 ε_found**：触发只需"这个事件里有我要的候选"并接受它，
纯度/污染由**输出率**去管（切紧一点即可）。所以 ε_found 是"特化触发可达效率"的
**下界式估计**——而且它用的是**通用**模型、**全链**目标、**全部（含不可重建）真值链**做分母，
三个方面都还会被特化进一步改善：
  1) 分母换成"接受度内、径迹可重建的信号候选"（排除根本找不回的 B）；
  2) 目标从"整条链"降到"最后一步顶点"（径迹数少得多）；
  3) 专项训练去掉多任务稀释；且单 B 场景下**多链竞争这个结构性缺陷不存在**。

用法: python3 estimate_specialized_trigger.py [sample]
"""
import sys

import pandas as pd

SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "00342451"
ROOT = "LHCb_logs/DFEI"
KEYS = ["AllParticles", "NoneIso", "PartReco", "NotFound"]


def main():
    ref = pd.read_csv(f"report_figs/truth_chain_total_inclusive_{SAMPLE}.csv")
    n_total = int(ref["n_truth_chains"].sum())
    ref = ref.assign(_k=list(zip(ref["RUNNUMBER"].astype("int64"), ref["EVENTNUMBER"].astype("int64"))))
    per_evt = ref.groupby("_k")["n_truth_chains"].sum()
    oneb = set(per_evt[per_evt == 1].index)                       # 单 B 事件

    import glob, os, re
    rows = []
    for csv in sorted(glob.glob(f"{ROOT}/version_*/signal_reco_df_*{SAMPLE}*.csv")):
        tag = os.path.basename(csv).split("__", 1)[-1][:-4]
        if re.search(r"smoke|small|paper|noprune|ckpt-scan|cpub", tag):
            continue
        m = re.search(r"version_(\d+)", csv)
        if not m or "RUNNUMBER" not in open(csv).readline():
            continue
        v = int(m.group(1))
        # [防护] 只接受属于该版本目录的评估文件 (tag 以 v<ver>_ 开头, 或等于该目录的 over_write),
        # 防止"配置里 model 键没改"导致别的臂把结果写进本目录(2026-10-16 combo 事故)
        _ow = set()
        _ic = f"{ROOT}/version_%d/input_config.yaml" % int(re.search(r"version_(\d+)", csv).group(1))
        try:
            import yaml as _y
            _c = _y.safe_load(open(_ic))
            for _sec in ("evaluate", "inference", "settings"):
                if isinstance(_c.get(_sec), dict) and _c[_sec].get("over_write"):
                    _ow.add(str(_c[_sec]["over_write"]))
        except Exception:
            pass
        _vnum = re.search(r"version_(\d+)", csv).group(1)
        if not (tag.startswith(f"v{_vnum}_") or tag in _ow):
            continue
        d = pd.read_csv(csv)
        if len(d) == 0:
            continue
        d = d.assign(_k=list(zip(d["RUNNUMBER"].astype("int64"), d["EVENTNUMBER"].astype("int64"))))
        c = {k: int(d[k].sum()) for k in KEYS}
        d1 = d[d["_k"].isin(oneb)]
        c1 = {k: int(d1[k].sum()) for k in KEYS}
        n1 = len(oneb)                                            # 单B事件数 == 单B真值链数
        rows.append({
            "ver": v, "tag": tag,
            "ε_clean%": round(100 * c["AllParticles"] / n_total, 2),
            "ε_found%": round(100 * (c["AllParticles"] + c["NoneIso"]) / n_total, 2),
            "ε_found%_1B": round(100 * (c1["AllParticles"] + c1["NoneIso"]) / n1, 2),
            "ε_clean%_1B": round(100 * c1["AllParticles"] / n1, 2),
            "NotFound#": c["NotFound"],
        })
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"N_total={n_total}   单B事件/链={len(oneb)}")
    print("ε_clean = 整链精确一致;  ε_found = 径迹全聚成一个对象(允许污染) —— 触发的相关量\n")
    print(t.sort_values("ε_found%_1B", ascending=False)
          [["ver", "tag", "ε_clean%", "ε_found%", "ε_clean%_1B", "ε_found%_1B"]].to_string(index=False))

    # ---- 链大小依赖: 末态径迹越少越容易(特化到小末态的直接证据) ----
    print("\n=== 恢复率 vs 该真值链在评估内的径迹数 (取 v651 / v648) ===")
    for v in (651, 648):
        fl = glob.glob(f"{ROOT}/version_{v}/signal_reco_df_*{SAMPLE}*.csv")
        fl = [f for f in fl if re.search(rf"v{v}_", f)] or fl
        if not fl:
            continue
        d = pd.read_csv(fl[0])
        d = d[d["_k"].isin(oneb)] if "_k" in d else d
        d = d.assign(_k=list(zip(d["RUNNUMBER"].astype("int64"), d["EVENTNUMBER"].astype("int64"))))
        d = d[d["_k"].isin(oneb)]
        g = d.groupby("NumSignalParticles").agg(n=("AllParticles", "size"),
                                                clean=("AllParticles", "sum"),
                                                found=("AllParticles", "sum"))
        d["_found"] = d["AllParticles"] + d["NoneIso"]
        g["found"] = d.groupby("NumSignalParticles")["_found"].sum()
        g["ε_clean%"] = 100 * g["clean"] / g["n"]
        g["ε_found%"] = 100 * g["found"] / g["n"]
        print(f"\n  v{v} (单B事件):")
        print(g[["n", "ε_clean%", "ε_found%"]].round(1).to_string())


if __name__ == "__main__":
    main()
