"""阈值 -> 最终指标 扫描（走管线自己的 EventReconstruction，保证口径与 eval 完全一致）。

做法: 事件只加载一次(留在内存), 对每个阈值改 EventReconstruction.configs 里的
      node_prune_thr/edge_prune_thr, 重跑 reconstruct_heavyhadrons, 收集 sig_df,
      再按 **固定分母 N_total(未剪枝真值链总数, 本事件子集)** 换算成 All_fix/Perf_fix。

用法: python thr_scan_official.py --config config_files/eval_armC_renorm_v557.yaml \
        --max-events 300 --tag armC_0904
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation          # noqa: E402
from wmpgnn.analysis.load_module import load_module                            # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader                 # noqa: E402
from wmpgnn.data_loader.weights_calculator import transform_pos_weight         # noqa: E402
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, get_truth_part_keys,  # noqa: E402
                                               get_truth_part_ids, reconstruct_decay, particle_name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--version", type=int, default=557)
    p.add_argument("--max-events", type=int, default=300)
    p.add_argument("--tag", default="scan")
    p.add_argument("--out", default="report_figs")
    p.add_argument("--thrs", default="0.5,0.6,0.7,0.75,0.8,0.85,0.88,0.9,0.92,0.95")
    return p.parse_args()


def main():
    a = parse_args()
    grid = [float(x) for x in a.thrs.split(",")]
    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    _, _, chunkloader = load_tst_loader(cfg)

    # ---- 事件只加载一次, 存到内存 (含模型输出; 保留 batch 形式供管线使用) ----
    batches = []
    graphs_all = []
    n = 0
    for batch in chunkloader.test_dataloader():
        batch = batch.to(dev)
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            out = module.model(batch)
        blk = module.model._blocks[-1]
        out["node_weights"] = blk.node_weights["tracks"].squeeze()
        out["edge_weights"] = blk.edge_weights[("tracks", "to", "tracks")].squeeze()
        out[("tracks", "to", "tracks")].lca = out[("tracks", "to", "tracks")].edges   # 同 shared_step
        gs = out.to_data_list()
        graphs_all += gs
        n += len(gs)
        batches.append(out)
        if n >= a.max_events:
            break
    print(f"\n=== {a.tag} v{a.version}: 载入 {len(graphs_all)} 事件 ({len(batches)} 个 batch)")

    # ---- N_total: 未剪枝图上的真值链总数 (固定分母) ----
    n_total = 0
    for g in graphs_all:
        try:
            tl = lca_truth_matrix(g)
            keys = get_truth_part_keys(g).tolist()
            ids = list(map(particle_name, get_truth_part_ids(g).numpy()))
            tc, _, _ = reconstruct_decay(tl, keys, particle_ids=ids, truth_level_simulation=1)
            n_total += len(tc)
        except Exception:
            pass
    print(f"  未剪枝真值链总数 N_total = {n_total}")

    print(f"\n{'thr':>6}{'剪枝后链数':>11}{'覆盖率%':>9}{'All#':>7}{'All_fix%':>10}{'Perf_fix%':>10}"
          f"{'NoneIso%':>10}{'Part%':>7}{'NotFound%':>10}")
    rows = []
    for thr in grid:
        module.evt_reco.configs["node_prune_thr"] = thr
        module.evt_reco.configs["edge_prune_thr"] = thr
        module.evt_reco.sig_df = []
        module.evt_reco.evt_df = []
        module.evt_reco.evt_counter = 0
        module.evt_reco.log = {"pv_corr_ml": {}, "pv_corr_ip": {}, "pv_total": {}, "npvs": {}}
        module.evt_reco.device = dev
        for b in batches:
            module.evt_reco.reconstruct_heavyhadrons(b)
        df, _ = module.evt_reco.collect_results()
        if len(df) == 0:
            print(f"{thr:>6.2f}   空"); continue
        A = int(df["AllParticles"].sum()); P = int(df["PerfectReco"].sum())
        NI = int(df["NoneIso"].sum()); PR = int(df["PartReco"].sum()); NF = int(df["NotFound"].sum())
        N = len(df); Nt = max(1, n_total)
        row = dict(thr=thr, n_surv=N, coverage=100 * N / Nt, All_n=A, All_fix=100 * A / Nt,
                   Perf_fix=100 * P / Nt, NoneIso=100 * NI / Nt, Part=100 * PR / Nt, NotFound=100 * NF / Nt)
        rows.append(row)
        print(f"{thr:>6.2f}{N:>11}{row['coverage']:>9.2f}{A:>7}{row['All_fix']:>10.2f}{row['Perf_fix']:>10.2f}"
              f"{row['NoneIso']:>10.2f}{row['Part']:>7.2f}{row['NotFound']:>10.2f}")
    if rows:
        best = max(rows, key=lambda r: r["All_fix"])
        print(f"\nAll_fix 最大: thr={best['thr']:.2f} → All_fix {best['All_fix']:.2f}% / Perf_fix {best['Perf_fix']:.2f}% "
              f"(覆盖率 {best['coverage']:.2f}%)")
        import csv
        os.makedirs(a.out, exist_ok=True)
        p = f"{a.out}/thr_metric_official_{a.tag}_v{a.version}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("写出", p)


if __name__ == "__main__":
    main()
