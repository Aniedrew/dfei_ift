"""用管线自己的剪枝函数重算阈值曲线（对账用，避免手写掩码出错）。
对每个阈值: 走 reconstruction.py 的真实路径 (true_node_pruning -> edge_pruning),
然后在剪枝后的图上做真值解码, 得到"存活真值链数" —— 与 eval 的 N 口径一致。
用法: python thr_scan_pipeline.py --config <cfg> --max-events 200 --tag x
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
from wmpgnn.util.pruners import true_node_pruning, edge_pruning                # noqa: E402
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, get_truth_part_keys,  # noqa: E402
                                               get_truth_part_ids, reconstruct_decay, particle_name)

GRID = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.97, 0.99]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--version", type=int, default=557)
    p.add_argument("--max-events", type=int, default=200)
    p.add_argument("--tag", default="scan")
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
    _, _, chunkloader = load_tst_loader(cfg)

    # 收集: 每事件 (节点分数/标签, 边分数/标签) —— 用与管线一致的打标
    data = []
    n_evt = 0
    for batch in chunkloader.test_dataloader():
        batch = batch.to(dev)
        tt_ei_in = batch[("tracks", "to", "tracks")].edge_index.detach().cpu().clone()
        tb = batch["tracks"].batch.cpu().clone()
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            module.model(batch)
        blk = module.model._blocks[-1]
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1)
        ew = blk.edge_weights[("tracks", "to", "tracks")].detach().cpu().squeeze(-1)
        graphs = batch.to_data_list()
        for gid in range(int(tb.max().item()) + 1 if tb.numel() else 0):
            if n_evt >= a.max_events:
                break
            n_evt += 1
            tm = tb == gid
            em = tm[tt_ei_in[0]] & tm[tt_ei_in[1]]
            g = graphs[gid]
            # 一致性自检: 输入掩码选出的边数 vs 该事件图里的边数
            assert int(em.sum()) == int(g[("tracks", "to", "tracks")].edge_index.shape[1]), \
                (int(em.sum()), int(g[("tracks", "to", "tracks")].edge_index.shape[1]))
            data.append(dict(nw=nw[tm].clone(), ew=ew[em].clone(),
                             ft=g["tracks"].ft.numpy().astype(int),
                             y=g[("tracks", "to", "tracks")].y.numpy(),
                             g=g.clone()))
        if n_evt >= a.max_events:
            break

    n_node_sig = sum(int((d["ft"] != 1).sum()) for d in data)
    n_edge_sig = sum(int((d["y"] > 0).sum()) for d in data)
    print(f"\n=== {a.tag} v{a.version}: 事件 {len(data)}  信号节点 {n_node_sig}  结构边(双向) {n_edge_sig}")

    # 未剪枝的真值链总数 (固定分母, 本测试集口径)
    n_total = 0
    for d in data:
        try:
            tl = lca_truth_matrix(d["g"])
            keys = get_truth_part_keys(d["g"]).tolist()
            ids = list(map(particle_name, get_truth_part_ids(d["g"]).numpy()))
            tc, _, _ = reconstruct_decay(tl, keys, particle_ids=ids, truth_level_simulation=1)
            n_total += len(tc)
        except Exception:
            pass
    print(f"  未剪枝真值链总数 N_total(本 {len(data)} 事件) = {n_total}")

    print(f"\n{'thr':>6}{'节点召回%':>10}{'背景保留%':>10}{'边召回%':>9}{'存活链数':>9}{'覆盖率%':>9}")
    rows = []
    for thr in GRID:
        tot_chain = 0
        keep_n_recall = keep_n_bkg = keep_e_recall = 0.0
        k_tot = b_tot = e_tot = 0
        for d in data:
            g = d["g"].clone()
            nw = d["nw"]; ew = d["ew"]
            nlabel = d["ft"] != 1
            elabel = d["y"] > 0
            node_selbool = nw > thr
            keep_n_recall += int((node_selbool & torch.tensor(nlabel)).sum())
            keep_n_bkg += int((node_selbool & torch.tensor(~nlabel)).sum())
            k_tot += int(nlabel.sum()); b_tot += int((~nlabel).sum())
            edge_mask = true_node_pruning(node_selbool, g, "tracks", [('tracks', 'to', 'tracks')])
            edge_selbool = ew[edge_mask] > thr
            keep_e_recall += int((edge_selbool & torch.tensor(elabel)[edge_mask]).sum())
            e_tot += int(elabel.sum())
            edge_pruning(edge_selbool, g, ('tracks', 'to', 'tracks'))
            try:
                tl = lca_truth_matrix(g)
                keys = get_truth_part_keys(g).tolist()
                ids = list(map(particle_name, get_truth_part_ids(g).numpy()))
                tc, _, _ = reconstruct_decay(tl, keys, particle_ids=ids, truth_level_simulation=1)
                tot_chain += len(tc)
            except Exception:
                pass
        rows.append((thr, 100 * keep_n_recall / max(1, k_tot), 100 * keep_n_bkg / max(1, b_tot),
                     100 * keep_e_recall / max(1, e_tot), tot_chain, 100 * tot_chain / max(1, n_total)))
    for r in rows:
        print(f"{r[0]:>6.2f}{r[1]:>10.2f}{r[2]:>10.2f}{r[3]:>9.2f}{r[4]:>9d}{r[5]:>9.2f}")
    import csv
    os.makedirs(a.out, exist_ok=True)
    p = f"{a.out}/thr_scan_pipeline_{a.tag}_v{a.version}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["thr", "node_recall", "node_bkg_keep", "edge_recall", "n_chain_surv", "coverage"])
        for r in rows:
            w.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in r])
    print("写出", p)


if __name__ == "__main__":
    main()
