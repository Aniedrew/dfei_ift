"""固定分母重算: 把各版本(可变分母)的重建指标换算到统一口径。

问题
----
推理剪枝在**真值解码之前**就原地删节点:
  reconstruction.py L193-196  true_node_pruning(node_selbool, graph, "tracks", ...)
  pruners.py        true_node_pruning 会删除 graph["tracks"] 的**所有逐节点属性**,
                    包括 sig_keys(真值末态键) -> 被剪掉的径迹连同真值标签一起消失
  reco_helper.py    lca_truth_matrix 只用**幸存的** sig_keys 建真值 LCA
  reconstruction.py reconstruct_single_evt 在**已剪枝的图**上解出 tc_dict
=> 分母 = "剪枝后仍能拼出真值链的 B 数", 随阈值/模型变化:
   阈值越高 / 剪枝 MLP 越严 -> 删掉的信号径迹越多 -> 分母越小 -> 百分比虚高。

修法
----
分母固定为**完整图(不剪枝)上的真值链总数 N_total**,
被剪掉或对不上的 B 记为失败(计数 0)。
N_total 由 config_files/eval_CERN_noprune.yaml 产出(阈值取负 -> 全保留),
它与模型无关(不剪枝时真值解码只依赖数据), 故一个常数可服务所有版本。

用法
----
    python3 wmpgnn/analysis/recompute_fixed_denominator.py [N_total] [signal]

不给 N_total 时, 自动从 LHCb_logs/DFEI/version_*/info_*__noprune_reco.txt 读取。
signal 默认 CERN 的 inclusive_00342442; 公开数据传 00342442_inclusive。
结果同时写出 logs/fixed_denominator_metrics.csv。
"""
import glob
import os
import re
import sys

import pandas as pd

ROOT = "LHCb_logs/DFEI"
# 跳过非 20 文件/非 CERN 标准口径的评估
SKIP_TAG = re.compile(r"smoke|small|paper")


def read_n_total():
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    cands = glob.glob(f"{ROOT}/version_*/info_*__noprune_reco.txt")
    if not cands:
        raise SystemExit(
            "未找到 noprune 结果。请先跑 config_files/eval_CERN_noprune.yaml, "
            "或手动传入 N_total: python3 ...py <N_total>"
        )
    for path in cands:
        with open(path) as f:
            for line in f:
                if line.startswith("Reconstruction efficiency ("):
                    return int(re.search(r"\((\d+)\)", line).group(1))
    raise SystemExit(f"noprune 文件里没有效率条目: {cands}")


def main():
    n_total = read_n_total()
    signal = sys.argv[2] if len(sys.argv) > 2 else "inclusive_00342442"
    signal_glob = f"{ROOT}/version_*/signal_reco_df_{signal}*.csv"
    rows = []
    for csv in sorted(glob.glob(signal_glob)):
        base = os.path.basename(csv)
        tag = base.split("__", 1)[1][:-4] if "__" in base else "default"
        if SKIP_TAG.search(tag):
            continue
        version = int(re.search(r"version_(\d+)", csv).group(1))
        df = pd.read_csv(csv)
        n = len(df)
        if n == 0:
            continue
        ap = int(df["AllParticles"].sum())
        pr = int(df["PerfectReco"].sum())
        ni = int(df["NoneIso"].sum())
        rows.append({
            "version": version, "tag": tag, "N": n,
            "All#": ap, "Perfect#": pr,
            "All_orig%": round(100 * ap / n, 2),
            "Perf_orig%": round(100 * pr / n, 2),
            "All_fix%": round(100 * ap / n_total, 2),
            "Perf_fix%": round(100 * pr / n_total, 2),
        })

    df = pd.DataFrame(rows).sort_values(["version", "tag"]).reset_index(drop=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(f"N_total (完整图真值链数, 固定分母) = {n_total}")
    print(f"丢失/未命中一律计 0  ->  All_fix% = All# / {n_total}\n")
    print(df.to_string(index=False))
    out = f"logs/fixed_denominator_metrics_{signal}.csv"
    df.to_csv(out, index=False)
    print(f"\n已写出: {out}")


if __name__ == "__main__":
    main()
