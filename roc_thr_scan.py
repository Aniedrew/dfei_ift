"""剪枝阈值扫描 + ROC（本地 CPU 跑，不排队列）。
对每个阈值 thr 计算:
  - 节点: 信号径迹召回 / 背景保留 / 节点污染率
  - 边  : 结构边召回
  - 链  : 真值链"全节点+全结构边都存活"的比例 (= 覆盖率上限, 即剪枝侧能保住多少条链)
并输出 ROC 数据点 (node/edge) 与推荐阈值。

用法: python roc_thr_scan.py --config config_files/eval_armC_renorm_v557.yaml \
        --max-events 200 --tag armC_0904 [--thr-edge 0.5]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation          # noqa: E402
from wmpgnn.analysis.load_module import load_module                            # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader                 # noqa: E402
from wmpgnn.data_loader.weights_calculator import transform_pos_weight         # noqa: E402
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, get_truth_part_keys,  # noqa: E402
                                               get_truth_part_ids, get_final_keys,
                                               reconstruct_decay, particle_name)

GRID = np.round(np.arange(0.30, 0.995, 0.02), 3)


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

    node_s, node_l, edge_s, edge_l = [], [], [], []
    chains = []      # 每事件: list of dict(node_idx, node_score, edge_idx, edge_score)
    n_evt = 0
    for batch in chunkloader.test_dataloader():
        batch = batch.to(dev)
        tt_ei = batch[("tracks", "to", "tracks")].edge_index.detach().cpu().clone()
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
            em = tm[tt_ei[0]] & tm[tt_ei[1]]
            idx_global = tm.nonzero()[:, 0]
            key2local = {int(k): i for i, k in enumerate(idx_global.tolist())}
            ns = nw[tm].numpy()
            es = ew[em].numpy()
            g = graphs[gid]
            ft = g["tracks"].ft.numpy().astype(int)
            y = g[("tracks", "to", "tracks")].y.numpy()
            local_ei = torch.stack([torch.tensor([key2local[int(v)] for v in tt_ei[0][em].tolist()]),
                                    torch.tensor([key2local[int(v)] for v in tt_ei[1][em].tolist()])]) \
                if em.any() else torch.zeros(2, 0, dtype=torch.long)
            node_s.append(ns); node_l.append((ft != 1).astype(int))
            edge_s.append(es); edge_l.append((y > 0).astype(int))
            # 真值链 (完整图口径)
            try:
                tl = lca_truth_matrix(g)
                keys = get_truth_part_keys(g).tolist()
                ids = list(map(particle_name, get_truth_part_ids(g).numpy()))
                tc, _, _ = reconstruct_decay(tl, keys, particle_ids=ids, truth_level_simulation=1)
                pk = get_final_keys(g).tolist()
                pk2local = {int(k): i for i, k in enumerate(pk)}
                for ck, c in tc.items():
                    nidx = [pk2local[int(k)] for k in c["node_keys"] if int(k) in pk2local]
                    if not nidx:
                        continue
                    s = set(nidx)
                    eidx = [i for i, (u, v) in enumerate(zip(local_ei[0].tolist(), local_ei[1].tolist()))
                            if u in s and v in s and edge_l[-1][i] == 1]
                    chains.append(dict(n_idx=nidx, e_idx=eidx))
            except Exception as e:
                print(f"[warn] evt {n_evt} 真值解码异常: {type(e).__name__}: {e}")
        if n_evt >= a.max_events:
            break

    ns = np.concatenate(node_s); nl = np.concatenate(node_l)
    es = np.concatenate(edge_s); el = np.concatenate(edge_l)
    print(f"\n=== {a.tag} · v{a.version} · 事件 {n_evt} · 节点 {len(ns)}(信号 {int(nl.sum())}) · tt边 {len(es)}(结构 {int(el.sum())}) · 真值链 {len(chains)}")

    # AUC
    def auc(score, label):
        o = np.argsort(score); r = np.empty(len(o), float); r[o] = np.arange(1, len(o) + 1)
        p = label.sum(); n = len(label) - p
        return (r[label == 1].sum() - p * (p + 1) / 2) / (p * n) if p and n else float("nan")
    print(f"  AUC(node) = {auc(ns, nl):.5f}   AUC(edge) = {auc(es, el):.5f}")

    rows = []
    for thr in GRID:
        keep_n = ns > thr
        keep_e = es > thr
        n_recall = keep_n[nl == 1].mean() if (nl == 1).any() else np.nan
        n_bkg = keep_n[nl == 0].mean() if (nl == 0).any() else np.nan
        n_prec = nl[keep_n].mean() if keep_n.any() else np.nan
        e_recall = keep_e[el == 1].mean() if (el == 1).any() else np.nan
        surv = np.mean([all(keep_n[i] for i in c["n_idx"]) and all(keep_e[j] for j in c["e_idx"])
                        for c in chains]) if chains else np.nan
        rows.append(dict(thr=thr, node_recall=100 * n_recall, node_bkg_keep=100 * n_bkg,
                         node_precision=100 * n_prec, edge_recall=100 * e_recall,
                         chain_survival=100 * surv, kept_nodes=100 * keep_n.mean()))
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/thr_scan_{a.tag}_v{a.version}.csv", index=False)
    print(f"\n{'thr':>6}{'节点召回%':>10}{'背景保留%':>10}{'节点精度%':>10}{'边召回%':>9}{'链存活%':>9}{'保留节点%':>10}")
    for r in rows:
        flag = ""
        if r["thr"] in (0.5, 0.7, 0.8, 0.9, 0.95):
            flag = "  ←"
        print(f"{r['thr']:>6.2f}{r['node_recall']:>10.2f}{r['node_bkg_keep']:>10.2f}{r['node_precision']:>10.2f}"
              f"{r['edge_recall']:>9.2f}{r['chain_survival']:>9.2f}{r['kept_nodes']:>10.2f}{flag}")
    best = df.loc[df["chain_survival"].idxmax()]
    print(f"\n链存活最大: thr={best['thr']:.2f} (链存活 {best['chain_survival']:.2f}%, 节点召回 {best['node_recall']:.2f}%, "
          f"背景保留 {best['node_bkg_keep']:.2f}%, 保留节点 {best['kept_nodes']:.2f}%)")
    # 与主口径 comparable: 覆盖率 = 链存活
    np.save(f"{a.out}/roc_scan_{a.tag}_v{a.version}_node.npy", np.stack([ns, nl]))
    np.save(f"{a.out}/roc_scan_{a.tag}_v{a.version}_edge.npy", np.stack([es, el]))


if __name__ == "__main__":
    main()
