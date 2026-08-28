"""
量化剪枝数据流失: version_25 (CERN新数据) 在不同剪枝阈值下的信号保留情况
- 节点剪枝: 真值信号径迹中 node_weight > thr 的比例
- 边剪枝: 真值正边(y>0)中 edge_weight > thr 的比例
- 链完整性: 完整存活(所有径迹+所有正边保留)的链比例
"""
import glob
import io
import sys
import os
import copy

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import analyze_reco_visual as A
from wmpgnn.reconstruction.reco_helper import lca_truth_matrix, reconstruct_decay

# 覆盖为 version_25 + CERN新数据
A.VERSION = 25
A.DATA_DIR = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed"
A.SAMPLE = "inclusive_00342442"


def main():
    max_scan = int(os.environ.get("MAX_SCAN", "400"))
    module, _ = A.load_module(None)
    module.eval()

    files = sorted(glob.glob(f"{A.DATA_DIR}/{A.SAMPLE}/tst_data_*"))
    dctx = zstd.ZstdDecompressor()

    # 统计
    sig_track_tot = 0
    sig_track_kept = {t: 0 for t in [0.2, 0.5, 0.9]}
    tp_edge_tot = 0
    tp_edge_kept = {t: 0 for t in [0.2, 0.5, 0.9]}
    n_chain_tot = 0
    n_chain_full = {t: 0 for t in [0.2, 0.5, 0.9]}
    n_events = 0

    def tt(y):  # 边 y 可能是 [N,4] onehot 或 [N]
        y = torch.as_tensor(y)
        return y.argmax(dim=-1) if y.dim() == 2 and y.shape[-1] > 1 else y

    for f in files:
        with open(f, "rb") as fh:
            with dctx.stream_reader(fh) as r:
                data = torch.load(io.BytesIO(r.read()), weights_only=False)
        for evt in data:
            n_events += 1
            if n_events > max_scan:
                break
            # 双向化 (与 load_dataset 一致)
            et = ("tracks", "to", "tracks")
            store = evt[et]
            store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
            store.edges = store.edges.repeat(2, 1)
            store.y = store.y.repeat(2)

            outputs, batch = A.predict(module, evt)
            node_w = outputs["node_weights"].cpu().numpy()
            edge_w = outputs["edge_weights"].cpu().numpy()

            # 真值信号径迹: part_keys ∈ sig_keys
            part_keys = evt["tracks"].part_keys.numpy()
            sig_set = set(evt["tracks"].sig_keys.numpy().tolist())
            sig_mask = np.isin(part_keys, list(sig_set))
            n_sig = int(sig_mask.sum())
            sig_track_tot += n_sig
            for t in sig_track_kept:
                sig_track_kept[t] += int((node_w[sig_mask] > t).sum())

            # 真值正边 (y>0, 且两端都是信号径迹)
            ei_np = evt[et].edge_index.cpu().numpy()
            y_np = tt(evt[et].y.cpu()).numpy()
            pos = (y_np > 0) & (ei_np[0] < ei_np[1]) & sig_mask[ei_np[0]] & sig_mask[ei_np[1]]
            tp_edge_tot += int(pos.sum())
            for t in tp_edge_kept:
                tp_edge_kept[t] += int((edge_w[pos] > t).sum())

            # 链完整性: 用 truth LCA 聚类
            tl = lca_truth_matrix(evt)
            if len(tl):
                keys = evt["tracks"].sig_keys.tolist()
                tc, _, _ = reconstruct_decay(tl, keys)
                for c in tc.values():
                    n_chain_tot += 1
                    ckeys = set(c["node_keys"])
                    cidx = [int(np.where(part_keys == k)[0][0]) for k in ckeys]
                    in_chain = np.zeros(len(part_keys), dtype=bool)
                    in_chain[cidx] = True
                    chain_pos = pos & in_chain[ei_np[0]] & in_chain[ei_np[1]]
                    for t in n_chain_full:
                        ok_tracks = all(node_w[i] > t for i in cidx)
                        ok_edges = True
                        if chain_pos.any():
                            ok_edges = bool((edge_w[chain_pos] > t).all())
                        if ok_tracks and ok_edges:
                            n_chain_full[t] += 1
            if n_events % 100 == 0:
                print(f"  {n_events} events", flush=True)
        if n_events > max_scan:
            break

    print(f"\n=== version_25 (CERN新数据) 剪枝数据流失分析 (n_events={n_events}) ===")
    print(f"信号径迹总数: {sig_track_tot}")
    for t in [0.2, 0.5, 0.9]:
        print(f"  node_thr={t}: 信号径迹存活 {sig_track_kept[t]/sig_track_tot*100:.1f}%")
    print(f"真值正边总数: {tp_edge_tot}")
    for t in [0.2, 0.5, 0.9]:
        print(f"  edge_thr={t}: 正边存活 {tp_edge_kept[t]/tp_edge_tot*100:.1f}%")
    print(f"真值链总数: {n_chain_tot}")
    for t in [0.2, 0.5, 0.9]:
        print(f"  thr={t}: 完整存活链 {n_chain_full[t]/n_chain_tot*100:.1f}%")


if __name__ == "__main__":
    main()
