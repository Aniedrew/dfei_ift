"""
非贪心重建分析: 用 v23 (论文数据训练) 模型 + 论文测试数据,
量化不同剪枝阈值下信号径迹/正边/完整链的存活率梯度,
回答"贪心剪枝是否损失了大量数据, 非贪心(低阈值/软权重)是否有提升空间"。
"""
import glob
import io
import sys
import os

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn")

import analyze_reco_visual as A
from wmpgnn.reconstruction.reco_helper import lca_truth_matrix, reconstruct_decay

# 覆盖为 v23 + 论文数据 (含完整 truth)
A.VERSION = 23
A.DATA_DIR = "/lzufs/user/guoqingxiang/DFEI_data/converted_LHCb_truth"
A.SAMPLE = "00342442_inclusive"

THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5]


def main():
    max_scan = int(os.environ.get("MAX_SCAN", "600"))
    module, hparams = A.load_module(None)
    module.eval()

    files = sorted(glob.glob(f"{A.DATA_DIR}/{A.SAMPLE}/tst_data_*"))
    dctx = zstd.ZstdDecompressor()

    sig_track_tot = 0
    sig_track_kept = {t: 0 for t in THRESHOLDS}
    tp_edge_tot = 0
    tp_edge_kept = {t: 0 for t in THRESHOLDS}
    n_chain_tot = 0
    n_chain_full = {t: 0 for t in THRESHOLDS}
    n_events = 0

    def tt(y):
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

            outputs, batch = A.predict(module, evt, use_pid=None)
            node_w = outputs["node_weights"].cpu().numpy()
            edge_w = outputs["edge_weights"].cpu().numpy()

            # 真值信号径迹: ft==2 (论文数据无 part_keys/sig_keys, 用 ft 定义信号)
            ft = evt["tracks"].ft.numpy()
            sig_mask = ft == 2
            n_sig = int(sig_mask.sum())
            sig_track_tot += n_sig
            for t in sig_track_kept:
                sig_track_kept[t] += int((node_w[sig_mask] > t).sum())

            # 真值正边 (y>0, 两端都是信号径迹)
            ei_np = evt[et].edge_index.cpu().numpy()
            y_np = tt(evt[et].y.cpu()).numpy()
            pos = (y_np > 0) & (ei_np[0] < ei_np[1]) & sig_mask[ei_np[0]] & sig_mask[ei_np[1]]
            tp_edge_tot += int(pos.sum())
            for t in tp_edge_kept:
                tp_edge_kept[t] += int((edge_w[pos] > t).sum())

            # 链完整性: 用 truth LCA 聚类 (论文数据 truth 走旧格式分支)
            tl = lca_truth_matrix(evt)
            if len(tl):
                from wmpgnn.reconstruction.reco_helper import get_truth_part_keys, get_final_keys
                tkeys = get_truth_part_keys(evt)
                tc, _, _ = reconstruct_decay(tl, tkeys.tolist())
                fk = get_final_keys(evt).numpy()
                for c in tc.values():
                    n_chain_tot += 1
                    ckeys = set(c["node_keys"])
                    cidx = []
                    for k in ckeys:
                        hit = np.where(fk == k)[0]
                        if len(hit):
                            cidx.append(int(hit[0]))
                    if len(cidx) == 0:
                        continue
                    in_chain = np.zeros(len(sig_mask), dtype=bool)
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

    print(f"\n=== v23 (论文数据) 剪枝流失分析 (n_events={n_events}) ===")
    print(f"信号径迹总数: {sig_track_tot}")
    for t in THRESHOLDS:
        print(f"  node_thr={t}: 信号径迹存活 {sig_track_kept[t]/sig_track_tot*100:.1f}%")
    print(f"真值正边总数: {tp_edge_tot}")
    for t in THRESHOLDS:
        print(f"  edge_thr={t}: 正边存活 {tp_edge_kept[t]/tp_edge_tot*100:.1f}%")
    print(f"真值链总数: {n_chain_tot}")
    for t in THRESHOLDS:
        print(f"  thr={t}: 完整存活链 {n_chain_full[t]/n_chain_tot*100:.1f}%")


if __name__ == "__main__":
    main()
