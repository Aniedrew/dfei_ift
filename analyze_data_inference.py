#!/usr/bin/env python
"""0904 真实数据(无真值)推理行为统计 —— 只做推理，不用任何真值标签。

背景: 真实碰撞数据 (DFEI_IFT_20260904/data/data_normed/) 没有 tt.y / ft / 真值链，
      现有评估链路 (EventReconstruction) 会在 lca_truth_matrix 处拿不到真值而崩，
      因此效率类指标 (AllParticles/PerfectReco/PV 错联率) 在 data 上无法计算。
      本脚本只报告"模型在真实数据上的行为统计"：
        1. 剪枝: 节点/边在阈值下的存活率、预测分布
        2. PV 关联: 每条径迹的 HGNN 分配 vs minIP 分配的一致率、每 PV 分到的径迹数、
                   无 tr-pv 边的径迹比例
        3. LCAG 解码(真值无关): 用模型 logits 走 reco 路径做 cluster 解码,
                   得到候选链数/链长分布 (与真值无关, 只反映模型声称的结构)

用法:
  python analyze_data_inference.py --config config_files/eval_0904real.yaml \
         --versions 31,38,47,557 [--max-events 200] [--out report_figs]
"""
import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation          # noqa: E402
from wmpgnn.analysis.load_module import load_module, get_bis_model             # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader                 # noqa: E402
from wmpgnn.data_loader.weights_calculator import transform_pos_weight         # noqa: E402
from wmpgnn.reconstruction.reco_helper import reconstruct_decay, get_final_keys  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="评估配置 (data_dir / sample / nfiles)")
    p.add_argument("--versions", default="557", help="逗号分隔的模型版本号, 如 31,38,47,557")
    p.add_argument("--thr-node", type=float, default=None, help="覆盖 node_prune_thr")
    p.add_argument("--thr-edge", type=float, default=None, help="覆盖 edge_prune_thr")
    p.add_argument("--max-events", type=int, default=0, help="只用前 N 个事件 (0=全部; 冒烟用)")
    p.add_argument("--out", default="report_figs", help="输出目录 (CSV)")
    return p.parse_args()


def reco_chains(graph, lca_logits):
    """真值无关的 LCAG 解码: 模型 logits -> argmax -> reconstruct_decay (reco 路径)。"""
    ei = graph[("tracks", "to", "tracks")].edge_index.cpu()
    dec = lca_logits.argmax(dim=-1).cpu()
    keep = ei[0] < ei[1]                      # 双向边只保留一个方向
    mat = pd.DataFrame({"senders": ei[0][keep].numpy(),
                        "receivers": ei[1][keep].numpy(),
                        "LCA_dec": dec[keep].numpy()})
    keys = get_final_keys(graph).cpu().tolist()
    cdict, _, _ = reconstruct_decay(mat, keys)
    return [len(v["node_keys"]) for v in cdict.values()]


def run_version(base_cfg_path, version, thr_node, thr_edge, max_events, rows, per_event):
    cfg = yaml.safe_load(open(base_cfg_path))
    cfg["settings"]["model"] = version
    if thr_node is not None:
        cfg["inference"]["node_prune_thr"] = thr_node
    if thr_edge is not None:
        cfg["inference"]["edge_prune_thr"] = thr_edge
    cfg = adjust_config_evaluation(cfg)
    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    tn = float(cfg["inference"]["node_prune_thr"])
    te = float(cfg["inference"]["edge_prune_thr"])
    _cpt = cfg["DFEI"]["cpt"]
    ckpt = os.path.basename(_cpt) if isinstance(_cpt, str) else os.path.basename(get_bis_model(_cpt, cfg))
    print(f"\n===== v{version}  thr_node={tn} thr_edge={te}  ({ckpt}) =====")

    _, _, chunkloader = load_tst_loader(cfg)
    acc = collections.defaultdict(float)
    n_evt = 0
    for batch in chunkloader.test_dataloader():
        batch = batch.to(dev)
        # 输入侧张量在 forward 前快照 (forward 会改写部分 store 的 edges)
        tt_ei = batch[("tracks", "to", "tracks")].edge_index.detach().cpu().clone()
        tp_ei = batch[("tracks", "to", "pvs")].edge_index.detach().cpu().clone()
        minip = batch[("tracks", "to", "pvs")].edges.detach().cpu().flatten().clone()
        tb = batch["tracks"].batch.cpu().clone()
        pb = batch["pvs"].batch.cpu().clone()
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            out = module.model(batch)
        blk = module.model._blocks[-1]
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1)
        ew = blk.edge_weights[("tracks", "to", "tracks")].detach().cpu().squeeze(-1)
        pw = blk.edge_weights[("tracks", "to", "pvs")].detach().cpu().squeeze(-1)
        lca_logits = out[("tracks", "to", "tracks")].edges.detach().cpu()
        graphs = out.to_data_list()

        for gid in range(int(tb.max().item()) + 1 if tb.numel() else 0):
            if max_events and n_evt >= max_events:
                break
            n_evt += 1
            tm = tb == gid
            pm = pb == gid
            ew_m = tm[tt_ei[0]] & tm[tt_ei[1]]
            tp_m = tm[tp_ei[0]] & pm[tp_ei[1]]
            n_tr = int(tm.sum())
            n_pv = int(pm.sum())
            acc["n_evt"] += 1
            acc["n_tracks"] += n_tr
            acc["n_pvs"] += n_pv
            acc["n_nodes_alive"] += int((nw[tm] > tn).sum())
            acc["n_edges_alive"] += int((ew[ew_m] > te).sum())
            acc["n_edges"] += int(ew_m.sum())
            # --- PV 关联 ---
            if tp_m.any():
                trk_of_edge = torch.searchsorted(tm.nonzero()[:, 0], tp_ei[0][tp_m])
                pv_of_edge = torch.searchsorted(pm.nonzero()[:, 0], tp_ei[1][tp_m])
                e_pred = pw[tp_m]
                e_ip = minip[tp_m]
                n_loc = int(tm.sum())
                pred = np.full(n_loc, -1, dtype=int)
                ip = np.full(n_loc, -1, dtype=int)
                trk_np = trk_of_edge.numpy(); pv_np = pv_of_edge.numpy()
                ep = e_pred.numpy(); eip = e_ip.numpy()
                for t in np.unique(trk_np):
                    s = trk_np == t
                    pred[t] = pv_np[s][np.argmax(ep[s])]
                    ip[t] = pv_np[s][np.argmin(eip[s])]
                has = pred >= 0
                acc["n_trk_with_edge"] += int(has.sum())
                acc["n_agree"] += int((pred[has] == ip[has]).sum())
                cnt_pred = np.bincount(pred[has], minlength=max(1, n_pv)) if has.any() else np.zeros(1)
                acc["max_trk_per_pv"] = max(acc["max_trk_per_pv"], int(cnt_pred.max()))
                acc["n_pv_used"] += int((cnt_pred > 0).sum())
            # --- 预测类别分布 (tt 边) ---
            cls = lca_logits[ew_m].argmax(-1).numpy()
            for c in range(4):
                acc[f"n_tt_class{c}"] += int((cls == c).sum())
            # --- LCAG 解码 (真值无关) ---
            sizes = reco_chains(graphs[gid], lca_logits[ew_m])
            acc["n_chains"] += len(sizes)
            acc["n_chain_nodes"] += int(sum(sizes))
            acc["max_chain"] = max(acc["max_chain"], max(sizes) if sizes else 0)
            per_event.append(dict(version=version, thr_node=tn, thr_edge=te, evt=n_evt,
                                  n_tracks=n_tr, n_pvs=n_pv,
                                  node_alive=int((nw[tm] > tn).sum()),
                                  edge_alive=int((ew[ew_m] > te).sum()), n_edges=int(ew_m.sum()),
                                  n_chains=len(sizes), chain_nodes=int(sum(sizes)),
                                  max_chain=max(sizes) if sizes else 0))
        if max_events and n_evt >= max_events:
            break

    e = max(1, acc["n_evt"])
    row = dict(version=version, thr_node=tn, thr_edge=te, ckpt=ckpt,
               n_evt=int(acc["n_evt"]),
               tracks_per_evt=acc["n_tracks"] / e, pvs_per_evt=acc["n_pvs"] / e,
               node_alive_pct=100 * acc["n_nodes_alive"] / max(1, acc["n_tracks"]),
               edge_alive_pct=100 * acc["n_edges_alive"] / max(1, acc["n_edges"]),
               trk_with_pv_pct=100 * acc["n_trk_with_edge"] / max(1, acc["n_tracks"]),
               hgnn_eq_minip_pct=100 * acc["n_agree"] / max(1, acc["n_trk_with_edge"]),
               max_trk_per_pv=int(acc["max_trk_per_pv"]),
               pv_used_per_evt=acc["n_pv_used"] / e,
               chains_per_evt=acc["n_chains"] / e,
               chain_nodes_per_evt=acc["n_chain_nodes"] / e,
               max_chain=int(acc["max_chain"]),
               tt_class_frac=[round(acc[f"n_tt_class{c}"] / max(1, acc["n_edges"]), 5) for c in range(4)])
    print(f"  事件 {row['n_evt']}  径迹/事件 {row['tracks_per_evt']:.1f}  PV/事件 {row['pvs_per_evt']:.2f}")
    print(f"  剪枝:  节点存活 {row['node_alive_pct']:.1f}%   边存活 {row['edge_alive_pct']:.2f}%")
    print(f"  PV:    有 tr-pv 边的径迹 {row['trk_with_pv_pct']:.1f}%   "
          f"HGNN==minIP {row['hgnn_eq_minip_pct']:.1f}%   单 PV 最多径迹 {row['max_trk_per_pv']}")
    print(f"  链:    {row['chains_per_evt']:.2f} 条/事件  平均 {row['chain_nodes_per_evt'] / max(1e-9, row['chains_per_evt']):.2f} 节点/链  "
          f"最大 {row['max_chain']}")
    print(f"  tt 边类别占比 (0/1/2/3): {row['tt_class_frac']}")
    rows.append(row)
    return row


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    versions = [int(v) for v in str(a.versions).split(",") if v.strip()]
    rows, per_event = [], []
    for v in versions:
        run_version(a.config, v, a.thr_node, a.thr_edge, a.max_events, rows, per_event)
    tag = os.path.splitext(os.path.basename(a.config))[0]
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/data_inference_stats_{tag}.csv", index=False)
    pd.DataFrame(per_event).to_csv(f"{a.out}/data_inference_per_event_{tag}.csv", index=False)
    print(f"\n汇总 -> {a.out}/data_inference_stats_{tag}.csv")
    print(df[["version", "n_evt", "node_alive_pct", "edge_alive_pct", "hgnn_eq_minip_pct",
              "chains_per_evt", "max_chain"]].to_string(index=False))


if __name__ == "__main__":
    main()
