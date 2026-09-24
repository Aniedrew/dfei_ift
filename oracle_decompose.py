"""Oracle 分解 (在修复后的 0904 生产上重做): 一次前向, 多组推理期干预, 量出剩余天花板。

干预 (通过管线自带的 _oracle_rewrite 钩子, 与 7 月那次同一套接口):
  baseline            原样 (模型剪枝 + 模型解码)
  node_add_tp         只补回被误删的真值径迹      -> 点侧 recall 的价值
  node_remove_fp      只删掉误留的背景径迹        -> 点侧 precision 的价值
  node_both           点剪枝 = 真值
  edge_add_tp / edge_remove_fp / edge_both        -> 边侧 recall / precision / 整体
  both_both           点+边都完美 = 剪枝天花板 (仍用模型的 LCAG 解码)
  node_drop_tp        反向对照 (按比例删真阳性, 应当显著变差, 验证实验有效)

指标: 固定分母 (未剪枝真值链总数 N_total), 与官方 eval 完全同口径 -> 可直接和 v601 的
      All_fix 14.5 / Perf_fix 9.5 对齐; 同时输出 PV 关联与剪枝 precision/recall。
用法: python oracle_decompose.py --config config_files/eval_v601_inc51_0904.yaml --version 601 --max-events 300
"""
import argparse
import csv
import os
import sys

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

ARMS = [
    ("baseline",          None,          None),
    ("both_both",         "both",        "both"),
    ("node_both",         "both",        None),
    ("edge_both",         None,          "both"),
    ("node_add_tp",       "add_tp",      None),
    ("node_remove_fp",    "remove_fp",   None),
    ("edge_add_tp",       None,          "add_tp"),
    ("edge_remove_fp",    None,          "remove_fp"),
    ("node_drop_tp",      "drop_tp",     None),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--version", type=int, default=601)
    p.add_argument("--max-events", type=int, default=300)
    p.add_argument("--tag", default="v601_0904")
    p.add_argument("--out", default="report_figs")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    _, _, ckl = load_tst_loader(cfg)

    batches, graphs_all, n = [], [], 0
    for batch in ckl.test_dataloader():
        batch = batch.to(dev)
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            out = module.model(batch)
        blk = module.model._blocks[-1]
        out["node_weights"] = blk.node_weights["tracks"].squeeze()
        out["edge_weights"] = blk.edge_weights[("tracks", "to", "tracks")].squeeze()
        out[("tracks", "to", "tracks")].lca = out[("tracks", "to", "tracks")].edges
        graphs_all += out.to_data_list()
        n += len(out.to_data_list())
        batches.append(out)
        if n >= a.max_events:
            break
    print(f"\n=== {a.tag} (model v{a.version}): {len(graphs_all)} 事件 / {len(batches)} batch", flush=True)

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
    print(f"  N_total (未剪枝真值链) = {n_total}", flush=True)

    rows = []
    for name, o_n, o_e in ARMS:
        c = module.evt_reco.configs
        c.pop("oracle_node", None); c.pop("oracle_edge", None)
        if o_n:
            c["oracle_node"] = o_n; c["oracle_node_frac"] = 1.0
        if o_e:
            c["oracle_edge"] = o_e; c["oracle_edge_frac"] = 1.0
        module.evt_reco._oracle_stat = dict(n_kept=0, n_sig=0, n_kept_sig=0, e_kept=0, e_sig=0, e_kept_sig=0)
        module.evt_reco.sig_df = []
        module.evt_reco.evt_df = []
        module.evt_reco.evt_counter = 0
        module.evt_reco.log = {"pv_corr_ml": {}, "pv_corr_ip": {}, "pv_total": {}, "npvs": {}}
        module.evt_reco.device = dev
        for b in batches:
            module.evt_reco.reconstruct_heavyhadrons(b)
        st = module.evt_reco._oracle_stat
        try:
            df, _ = module.evt_reco.collect_results()
        except ValueError:
            print(f"{name:>16}  空 (无重建链) | 点 p/r={100*st['n_kept_sig']/max(1,st['n_kept']):.1f}/"
                  f"{100*st['n_kept_sig']/max(1,st['n_sig']):.1f} 边 p/r="
                  f"{100*st['e_kept_sig']/max(1,st['e_kept']):.1f}/{100*st['e_kept_sig']/max(1,st['e_sig']):.1f}", flush=True)
            continue
        prec = 100 * st["n_kept_sig"] / max(1, st["n_kept"])
        rec = 100 * st["n_kept_sig"] / max(1, st["n_sig"])
        eprec = 100 * st["e_kept_sig"] / max(1, st["e_kept"])
        erec = 100 * st["e_kept_sig"] / max(1, st["e_sig"])
        if len(df) == 0:
            print(f"{name:>16}  空", flush=True); continue
        A = int(df["AllParticles"].sum()); P = int(df["PerfectReco"].sum())
        NI = int(df["NoneIso"].sum()); PR = int(df["PartReco"].sum()); NF = int(df["NotFound"].sum())
        N = len(df); Nt = max(1, n_total)
        r = dict(arm=name, oracle_node=o_n or "-", oracle_edge=o_e or "-", n_surv=N,
                 coverage=round(100 * N / Nt, 2), All_n=A,
                 All_fix=round(100 * A / Nt, 2), Perf_fix=round(100 * P / Nt, 2),
                 NoneIso=round(100 * NI / Nt, 2), Part=round(100 * PR / Nt, 2),
                 NotFound=round(100 * NF / Nt, 2),
                 node_prec=round(prec, 2), node_rec=round(rec, 2),
                 edge_prec=round(eprec, 2), edge_rec=round(erec, 2))
        rows.append(r)
        print(f"{name:>16}  N={N:>5}  cov={r['coverage']:>6.2f}  All_fix={r['All_fix']:>6.2f}  "
              f"Perf_fix={r['Perf_fix']:>6.2f}  NoneIso={r['NoneIso']:>6.2f}  "
              f"| 点 p/r={r['node_prec']:>6.2f}/{r['node_rec']:>6.2f}  边 p/r={r['edge_prec']:>6.2f}/{r['edge_rec']:>6.2f}",
              flush=True)

    os.makedirs(a.out, exist_ok=True)
    p = f"{a.out}/oracle_decompose_{a.tag}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    base = next((r for r in rows if r["arm"] == "baseline"), None)
    both = next((r for r in rows if r["arm"] == "both_both"), None)
    if base and both:
        print(f"\n剪枝天花板: All_fix {base['All_fix']:.2f} -> {both['All_fix']:.2f} "
              f"(+{both['All_fix'] - base['All_fix']:.2f}pp), Perf_fix {base['Perf_fix']:.2f} -> {both['Perf_fix']:.2f}")
    print("写出", p)


if __name__ == "__main__":
    main()
