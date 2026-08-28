"""v25 CERN 数据: 链死亡机制三分分解
- survival_both(t)    : 全部径迹 + 全部正边都存活 (现状)
- survival_nodes_only(t): 全部径迹存活 (边全部保留) -> 方案E(edge top-k)的天花板
- survival_edges_only(t): 全部正边存活 (节点全部保留) -> 方案B(node)的天花板
"""
import glob
import io
import os
import sys

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import analyze_reco_visual as A
from wmpgnn.reconstruction.reco_helper import lca_truth_matrix, reconstruct_decay

A.VERSION = 25
A.DATA_DIR = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed"
A.SAMPLE = "inclusive_00342442"

MAX_SCAN = int(os.environ.get("MAX_SCAN", "150"))
THRESHOLDS = [0.05, 0.2, 0.5]

def main():
    module, _ = A.load_module(None)
    module.eval()

    files = sorted(glob.glob(f"{A.DATA_DIR}/{A.SAMPLE}/tst_data_*"))
    dctx = zstd.ZstdDecompressor()

    n_chains = 0
    surv = {t: {"both": 0, "nodes": 0, "edges": 0} for t in THRESHOLDS}
    n_events = 0

    for f in files:
        with open(f, "rb") as fh:
            with dctx.stream_reader(fh) as r:
                data = torch.load(io.BytesIO(r.read()), weights_only=False)
        for evt in data:
            n_events += 1
            if n_events > MAX_SCAN:
                break
            et = ("tracks", "to", "tracks")
            store = evt[et]
            store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
            store.edges = store.edges.repeat(2, 1)
            store.y = store.y.repeat(2)

            outputs, batch = A.predict(module, evt)
            node_w = outputs["node_weights"].cpu().numpy()
            edge_w = outputs["edge_weights"].cpu().numpy()

            part_keys = evt["tracks"].part_keys.numpy()
            ei_np = evt[et].edge_index.cpu().numpy()
            y_np = evt[et].y.cpu().numpy()
            if y_np.ndim == 2 and y_np.shape[-1] > 1:
                y_np = y_np.argmax(axis=-1)
            pos = (y_np > 0) & (ei_np[0] < ei_np[1])

            tl = lca_truth_matrix(evt)
            if len(tl) == 0:
                continue
            keys = evt["tracks"].sig_keys.tolist()
            tc, _, _ = reconstruct_decay(tl, keys)
            for c in tc.values():
                ckeys = list(c["node_keys"])
                cidx = [int(np.where(part_keys == k)[0][0]) for k in ckeys]
                if not cidx:
                    continue
                in_chain = np.zeros(len(part_keys), dtype=bool)
                in_chain[cidx] = True
                chain_pos = pos & in_chain[ei_np[0]] & in_chain[ei_np[1]]
                n_chains += 1
                for t in THRESHOLDS:
                    ok_nodes = all(node_w[i] > t for i in cidx)
                    ok_edges = bool((edge_w[chain_pos] > t).all()) if chain_pos.any() else True
                    surv[t]["both"] += int(ok_nodes and ok_edges)
                    surv[t]["nodes"] += int(ok_nodes)
                    surv[t]["edges"] += int(ok_edges)
        if n_events % 50 == 0:
            print(f"  {n_events} events", flush=True)
        if n_events > MAX_SCAN:
            break

    print(f"\n=== v25 CERN 链死亡机制分解 (n_events={n_events}, n_chains={n_chains}) ===")
    for t in THRESHOLDS:
        b = surv[t]["both"] / n_chains * 100
        nd = surv[t]["nodes"] / n_chains * 100
        ed = surv[t]["edges"] / n_chains * 100
        print(f"thr={t}: 链存活 both={b:.1f}% | 仅节点保留(nodes-only)={nd:.1f}% | 仅边保留(edges-only)={ed:.1f}%")

if __name__ == "__main__":
    main()
