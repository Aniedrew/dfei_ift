"""
从 signal_reco_df CSV 计算论文风格的重建类别占比
- per-chain (官方 calculate_accuracy 口径, 互斥化): perfect / complete / notiso / partial / notfound
- per-event (严格: 事件内所有链同类才归入该类)
用法: python3 compare_reco_metrics.py <signal_reco_df.csv> [label]
"""
import sys

import numpy as np
import pandas as pd


def wilson_err(k, n):
    if n == 0 or k == 0:
        return np.nan
    p = k / n
    return p * np.sqrt(1 / k + 1 / n) * 100  # 与论文/官方相同的泊松误差


def per_chain_stats(df):
    n = len(df)
    print(f"\n=== per-chain (n={n}) ===")
    # 官方重叠口径
    allp = (df["AllParticles"] == 1).sum()
    perf = (df["PerfectReco"] == 1).sum()
    niso = (df["NoneIso"] == 1).sum()
    part = (df["PartReco"] == 1).sum()
    notf = (df["NotFound"] == 1).sum()
    # 互斥口径
    complete = allp - perf  # AllParticles 且非 perfect
    rows = [
        ("perfect", perf), ("complete (excl)", complete), ("not isolated", niso),
        ("partial", part), ("notfound", notf),
    ]
    for name, k in rows:
        print(f"  {name:16s}: {k/n*100:6.2f}% ± {wilson_err(k, n):.2f}")
    return rows


def per_event_stats(df):
    """事件级: 事件内所有链都是某类才归入该类 (严格)"""
    def evt_status(rows):
        if all(rows["PerfectReco"] == 1):
            return "perfect"
        if all((rows["PerfectReco"] == 1) | (rows["AllParticles"] == 1)):
            return "complete"
        if any((rows["PerfectReco"] == 1) | (rows["AllParticles"] == 1) |
               (rows["PartReco"] == 1) | (rows["NoneIso"] == 1)):
            return "partial"
        return "notfound"

    evt = df.groupby("EventNumber", as_index=False).agg(n=("EventNumber", "size"))
    evt["status"] = df.groupby("EventNumber").apply(
        evt_status, include_groups=False).reset_index(name="status")["status"]
    n = len(evt)
    print(f"\n=== per-event (n={n}) ===")
    for name in ["perfect", "complete", "partial", "notfound"]:
        k = (evt["status"] == name).sum()
        print(f"  {name:12s}: {k/n*100:6.2f}% ± {wilson_err(k, n):.2f}  (n={k})")
    return evt


if __name__ == "__main__":
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    df = pd.read_csv(path)
    print(f"===== {label} | {path} =====")
    print(f"总链数(rows): {len(df)}, 唯一事件数: {df['EventNumber'].nunique()}")
    print(f"每事件链数分布: {df.groupby('EventNumber').size().value_counts().sort_index().to_dict()}")
    per_chain_stats(df)
    per_event_stats(df)
