"""把"剪枝线"各版本的逐径迹分数 dump 下来, 供量化分析 (节点/边 ROC、工作点、链存活曲线)。

固定事件子集: 每个版本都取 (RUNNUMBER, EVENTNUMBER) 最小的 N 个事件 -> 各版本严格同一批事件。
每个版本存一个 npz:
  node_scores, node_labels        (每条径迹: 模型点剪枝分数, 是否真值 b 链径迹)
  edge_scores, edge_labels        (每条 tt 边: 模型边剪枝分数, 是否真值结构边)
  chains[evt] = (node_idx, edge_idx)  真值链在**未剪枝**图上的节点/真值结构边索引 (各版本同一真值, 与模型无关)
用法: python dump_pruning_scores.py --version 557 --events 250 [--config config_files/eval_armA_0702_v557.yaml]
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

BASE_CFG = "config_files/eval_armA_0702_v557.yaml"   # July data (arm A), July normalization


def event_id(g):
    for k in ("EVENTNUMBER", "event_id", "EVENTNUMBER_reco"):
        if k in g:
            return int(np.ravel(g[k])[0])
        if k in g["tracks"]:
            return int(np.ravel(g["tracks"][k])[0])
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--config", default=BASE_CFG)
    ap.add_argument("--events", type=int, default=250)
    ap.add_argument("--out", default="report_figs/pruning_scores")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    cfg["settings"]["model"] = a.version
    cfg = adjust_config_evaluation(cfg)
    module = load_module(cfg, transform_pos_weight(None, None, mode="eval"))
    module.eval()
    dev = next(module.parameters()).device
    _, _, ckl = load_tst_loader(cfg)

    evts = []
    for batch in ckl.test_dataloader():
        batch = batch.to(dev)
        ei_in = batch[("tracks", "to", "tracks")].edge_index.detach().cpu().clone()
        tb = batch["tracks"].batch.detach().cpu().clone()
        if module.use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        with torch.no_grad():
            out = module.model(batch)
        blk = module.model._blocks[-1]
        nw = blk.node_weights["tracks"].detach().cpu().squeeze(-1).numpy()
        ew = blk.edge_weights[("tracks", "to", "tracks")].detach().cpu().squeeze(-1).numpy()
        graphs = out.to_data_list()
        ft_all = batch["tracks"].ft.detach().cpu().numpy().astype(int)
        y_all = batch[("tracks", "to", "tracks")].y.detach().cpu().numpy()
        for gid in range(int(tb.max().item()) + 1):
            tm = (tb == gid)
            em = tm[ei_in[0]] & tm[ei_in[1]]
            g = graphs[gid]
            gt = tm.nonzero()[:, 0].numpy()
            # 真值链结构 (未剪枝图)
            try:
                tl = lca_truth_matrix(g)
                keys = get_truth_part_keys(g).tolist()
                ids = list(map(particle_name, get_truth_part_ids(g).numpy()))
                tc, _, _ = reconstruct_decay(tl, keys, particle_ids=ids, truth_level_simulation=1)
            except Exception:
                tc = {}
            pk = g["tracks"].part_keys.numpy().tolist()
            pk2i = {int(k): i for i, k in enumerate(pk)}
            chains = []
            ei_e = ei_in[:, em.numpy()].numpy()
            elab = (y_all[em.numpy()] > 0).astype(int)
            for ck, c in tc.items():
                nidx = [pk2i[int(k)] for k in c["node_keys"] if int(k) in pk2i]
                if len(nidx) < 2:
                    continue
                s = set(nidx)
                eidx = [i for i in range(ei_e.shape[1]) if ei_e[0, i] in s and ei_e[1, i] in s and elab[i] == 1]
                chains.append((np.array(nidx, dtype=np.int32), np.array(eidx, dtype=np.int32)))
            evts.append(dict(eid=event_id(g), evt=gid,
                             ns=nw[tm.numpy()].astype(np.float32),
                             nl=(ft_all[gt] != 1).astype(np.int8),
                             es=ew[em.numpy()].astype(np.float32),
                             el=elab.astype(np.int8),
                             chains=chains))
        if len(evts) >= int(a.events * 1.4):
            break

    evts.sort(key=lambda e: e["eid"])
    evts = evts[: a.events]
    os.makedirs(a.out, exist_ok=True)
    out = f"{a.out}/v{a.version}.npz"
    np.savez_compressed(
        out,
        meta=np.array([len(evts), sum(len(e["ns"]) for e in evts), sum(len(e["es"]) for e in evts),
                       sum(len(e["chains"]) for e in evts)]),
        node_scores=np.concatenate([e["ns"] for e in evts]),
        node_labels=np.concatenate([e["nl"] for e in evts]),
        edge_scores=np.concatenate([e["es"] for e in evts]),
        edge_labels=np.concatenate([e["el"] for e in evts]),
        eids=np.array([e["eid"] for e in evts]),
        nnode=np.array([len(e["ns"]) for e in evts]),
        nedge=np.array([len(e["es"]) for e in evts]),
        nchain=np.array([len(e["chains"]) for e in evts]),
        chain_nodes=np.concatenate([c[0] for e in evts for c in e["chains"]]) if any(e["chains"] for e in evts) else np.zeros(0, np.int32),
        chain_edges=np.concatenate([c[1] for e in evts for c in e["chains"]]) if any(e["chains"] for e in evts) else np.zeros(0, np.int32),
        chain_lens=np.array([len(c[0]) for e in evts for c in e["chains"]]),
        chain_elens=np.array([len(c[1]) for e in evts for c in e["chains"]]),
    )
    m = np.load(out)["meta"]
    print(f"[v{a.version}] events={m[0]} nodes={m[1]} edges={m[2]} chains={m[3]} -> {out}")


if __name__ == "__main__":
    main()
