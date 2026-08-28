#!/usr/bin/env python3
"""F 评估完成后自动叠加 chain_lca_filter 的多阈值对比脚本。

流程:
  1. 等待 F 评估 (方案F, 无 LCA 记录) 的 signal_reco_df CSV 生成;
  2. 提交/等待 chain_lca_record 评估 (记录每条链的 LCA 置信度, 不过滤);
  3. 读 record CSV, 对每个 conf_thr 重放过滤 (剔除 conf<thr 的重建链),
     重新计算 per-chain / per-event 指标, 输出对比表。

用法:
  python3 compare_chain_lca_thr.py --base-csv <F评估signal_csv> \
      --record-csv <record评估signal_csv> --thr 0.5,0.6,0.7 \
      [--record-config eval_CERN_v31_f_lca.yaml]   # record CSV 缺失时自动提交评估
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = f"{BASE}/LHCb_logs/DFEI/version_31"


def wait_for_csv(path, timeout_min=180, poll_s=60):
    """轮询等待 CSV 出现 (评估完成写出 signal_reco_df)。"""
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout_min * 60:
            raise TimeoutError(f"等待超时: {path}")
        print(f"[wait] {path} 未生成, {poll_s}s 后重试...")
        time.sleep(poll_s)
    print(f"[ok] CSV 已生成: {path}")


def submit_and_wait(record_config, record_csv, timeout_min=240):
    """提交 record 评估作业并等待其 CSV。"""
    print(f"[submit] 提交 chain_lca_record 评估: {record_config}")
    out = subprocess.run(
        ["hep_sub", f"{BASE}/submit_eval.sh", "-argu", record_config,
         "-g", "ghigh", "-gpu", "1", "-cpu", "4", "-m", "32000", "-wt", "mid",
         "-o", f"{BASE}/logs/eval_{os.path.splitext(record_config)[0]}.out",
         "-e", f"{BASE}/logs/eval_{os.path.splitext(record_config)[0]}.err"],
        capture_output=True, text=True)
    print(f"[submit] {out.stdout.strip()} {out.stderr.strip()}")
    wait_for_csv(record_csv, timeout_min=timeout_min, poll_s=120)


def replay_threshold(df, thr):
    """按 conf 阈值重放链级 LCA 过滤。

    规则: 原匹配成功 (chain_lca_conf >= 0) 且 conf < thr 的 truth 链行
    -> 其重建链被剔除 -> 该行变为 NotFound。
    """
    d = df.copy()
    drop = (d["chain_lca_conf"] >= 0) & (d["chain_lca_conf"] < thr)
    d.loc[drop, ["PerfectReco", "AllParticles", "NoneIso", "PartReco"]] = 0
    d.loc[drop, "NotFound"] = 1
    return d


def metrics(d):
    """per-chain 分类 + per-event Perfect。"""
    n = len(d)
    chain = {
        "Perfect": d["PerfectReco"].mean() * 100,
        "AllParticles": d["AllParticles"].mean() * 100,
        "NoneIso": d["NoneIso"].mean() * 100,
        "PartReco": d["PartReco"].mean() * 100,
        "NotFound": d["NotFound"].mean() * 100,
    }
    # per-event Perfect: 事件内所有 truth 链都 PerfectReco (用 collect_results 插入的 EventNumber 列)
    evt_perfect = d.groupby("EventNumber")["PerfectReco"].agg(lambda s: float(s.all()))
    return chain, evt_perfect.mean() * 100 if len(evt_perfect) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", default=f"{LOG_DIR}/signal_reco_df_inclusive_00342442__v31_f_thr09_k12_h3_s05.csv")
    ap.add_argument("--record-csv", default=f"{LOG_DIR}/signal_reco_df_inclusive_00342442__v31_f_lca_record_thr09.csv")
    ap.add_argument("--record-config", default="eval_CERN_v31_f_lca.yaml")
    ap.add_argument("--thr", default="0.5,0.6,0.7,0.8")
    ap.add_argument("--submit", action="store_true",
                    help="record CSV 缺失时自动提交 record 评估 (需要 hep_sub + GPU)")
    args = ap.parse_args()
    thrs = [float(x) for x in args.thr.split(",")]

    # 1. 等 F 评估完成 (base CSV 出现, 作为方案F结果记录)
    wait_for_csv(args.base_csv, timeout_min=120, poll_s=60)

    # 2. 等/提 record 评估
    if not os.path.exists(args.record_csv):
        if not args.submit:
            print(f"[提示] record CSV 不存在: {args.record_csv}")
            print("       用 --submit 自动提交 record 评估, 或先手动跑: "
                  f"hep_sub submit_eval.sh -argu {args.record_config} ...")
            sys.exit(2)
        submit_and_wait(args.record_config, args.record_csv)
    else:
        print(f"[ok] record CSV 已存在: {args.record_csv}")

    # 3. 读 CSV + 多阈值重放
    df = pd.read_csv(args.record_csv)
    print(f"\nrecord CSV: {len(df)} 行 (truth 链), {df['EVENTNUMBER'].nunique()} 事件")
    print(f"chain_lca_conf 分布: min={df['chain_lca_conf'].min():.3f} "
          f"med={df['chain_lca_conf'].median():.3f} max={df['chain_lca_conf'].max():.3f}")

    rows = []
    for label, thr in [("无过滤(F)", -1.0)] + [(f"conf≥{thr}", thr) for thr in thrs]:
        d = replay_threshold(df, thr)
        chain, evt_perf = metrics(d)
        rows.append((label, thr, chain, evt_perf))

    # 输出对比表
    lines = []
    header = f"{'阈值':<12}{'Per-chain Perfect':>18}{'AllParticles':>14}{'NoneIso':>10}{'PartReco':>10}{'NotFound':>10}{'Per-event Perfect':>18}"
    print("\n" + "=" * 92)
    print(header)
    lines.append("## chain_lca_filter 阈值对比 (方案F + 链级LCA物理判据)\n")
    lines.append("| 阈值 | Per-chain Perfect | AllParticles | NoneIso | PartReco | NotFound | Per-event Perfect |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, thr, chain, evt_perf in rows:
        print(f"{label:<12}{chain['Perfect']:>17.2f}%{chain['AllParticles']:>13.2f}%"
              f"{chain['NoneIso']:>9.2f}%{chain['PartReco']:>9.2f}%{chain['NotFound']:>9.2f}%"
              f"{evt_perf:>17.2f}%")
        lines.append(f"| {label} | {chain['Perfect']:.2f}% | {chain['AllParticles']:.2f}% | "
                     f"{chain['NoneIso']:.2f}% | {chain['PartReco']:.2f}% | {chain['NotFound']:.2f}% | "
                     f"{evt_perf:.2f}% |")
    print("=" * 92)

    out_md = f"{LOG_DIR}/chain_lca_thr_compare.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write(f"\n> 来源: {args.record_csv}\n> conf_thr 扫描: {args.thr}\n")
    print(f"\n[已保存] 对比表: {out_md}")


if __name__ == "__main__":
    main()
