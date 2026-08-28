"""生成 v25 (CERN 新数据) 重建事件可视化: 事件显示 + 真值/重建衰变树
适配新格式 truth (sig_keys/sig_ids), CPU 推理
输出到 meeting_20260811_figs/
"""
import os
import sys
import glob
import io
import numpy as np
import torch
import zstandard as zstd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
sys.path.insert(0, BASE)
OUT = f"{BASE}/meeting_20260811_figs"

import analyze_reco_visual as A
from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, lca_reco_matrix,
                                               reconstruct_decay, get_final_keys,
                                               get_truth_part_keys, get_truth_part_ids)
from wmpgnn.reconstruction.signal_dict import particle_name

A.VERSION = 25
A.DATA_DIR = "/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed"
A.SAMPLE = "inclusive_00342442"
N_SCAN = int(os.environ.get("N_SCAN", "250"))
A.NODE_THR = 0.2
A.EDGE_THR = 0.2


def get_chain_info(evt, graph):
    true_LCA = lca_truth_matrix(evt)
    particle_keys = get_truth_part_keys(evt).tolist()
    particle_ids = get_truth_part_ids(evt).numpy().tolist()
    tc_dict, _, _ = reconstruct_decay(true_LCA, particle_keys,
                                      particle_ids=list(map(particle_name, particle_ids)),
                                      truth_level_simulation=1)
    reco_keys = set()
    try:
        reco_LCA = lca_reco_matrix(graph, mode="reco")
        rc_dict, _, _ = reconstruct_decay(reco_LCA, get_final_keys(graph).tolist())
        for c in rc_dict.values():
            reco_keys.update(c["node_keys"])
    except Exception:
        pass
    all_keys = get_final_keys(evt).numpy().tolist()
    track_of_key = {k: i for i, k in enumerate(all_keys)}
    surv_keys = set(get_final_keys(graph).tolist())
    pruned_set = {i for i, k in enumerate(all_keys) if k not in surv_keys}
    key_pos = {k: i for i, k in enumerate(particle_keys)}
    chains = []
    for ck, c in tc_dict.items():
        keys = list(c["node_keys"])
        idxs = [track_of_key[k] for k in keys if k in track_of_key]
        if not idxs:
            continue
        pos_set = {key_pos[k] for k in keys if k in key_pos}
        rows = true_LCA[(true_LCA['senders'].isin(pos_set)) & (true_LCA['receivers'].isin(pos_set))]
        mother = "B?"
        if len(rows) > 0:
            lbl = rows.loc[rows['LCA_dec'].idxmax(), 'LCA_id_label']
            if lbl:
                mother = lbl
        nreco = sum(1 for k in keys if k in reco_keys)
        chains.append({
            "tracks": idxs,
            "names": [particle_name(particle_ids[key_pos[k]]) for k in keys if k in key_pos],
            "pids": [particle_ids[key_pos[k]] for k in keys if k in key_pos],
            "mother": mother,
            "nreco": nreco,
            "ntot": len(keys),
        })
    return chains, pruned_set


def classify(evt, graph):
    try:
        true_LCA = lca_truth_matrix(graph)
        tk = get_truth_part_keys(graph).tolist()
        tc_dict, _, _ = reconstruct_decay(true_LCA, tk)
        reco_LCA = lca_reco_matrix(graph, mode="reco")
        rc_dict, _, _ = reconstruct_decay(reco_LCA, get_final_keys(graph).tolist())
        if tc_dict == {}:
            return "nosignal"
        sts = []
        for tc in tc_dict.values():
            best = "notfound"
            for rc in rc_dict.values():
                if rc["node_keys"] == tc["node_keys"] and rc["LCA_values"] == tc["LCA_values"]:
                    best = "perfect"; break
                elif rc["node_keys"] == tc["node_keys"]:
                    best = "allfound"; break
                t_frac = np.sum(np.isin(tc["node_keys"], rc["node_keys"])) / len(tc["node_keys"])
                if t_frac == 1 and len(rc["node_keys"]) > len(tc["node_keys"]):
                    best = "noniso"; break
                elif 0.2 <= t_frac < 1:
                    best = "partial"; break
            sts.append(best)
        if all(s == "perfect" for s in sts):
            return "perfect"
        if all(s in ("perfect", "allfound") for s in sts):
            return "allfound"
        if any(s in ("perfect", "allfound", "noniso", "partial") for s in sts):
            return "partial"
        return "notfound"
    except Exception as e:
        return f"error:{e}"


def main():
    module, _ = A.load_module(None)
    module.eval()
    files = sorted(glob.glob(f"{A.DATA_DIR}/{A.SAMPLE}/tst_data_*"))
    dctx = zstd.ZstdDecompressor()
    results = []
    n = 0
    for f in files:
        with open(f, "rb") as fh:
            with dctx.stream_reader(fh) as r:
                data = torch.load(io.BytesIO(r.read()), weights_only=False)
        for evt in data:
            n += 1
            if n > N_SCAN:
                break
            et = ("tracks", "to", "tracks")
            st = evt[et]
            st.edge_index = torch.cat([st.edge_index, st.edge_index.flip(0)], dim=1)
            st.edges = st.edges.repeat(2, 1)
            st.y = st.y.repeat(2)
            outputs, batch = A.predict(module, evt)
            # 注意: classify_event 的默认阈值在 import 时绑定为 0.9, 必须显式传 0.2
            graph = A.classify_event(evt, outputs, node_thr=0.2, edge_thr=0.2)
            status = classify(evt, graph)
            results.append((status, n - 1, graph, evt))
            if (n) % 50 == 0:
                print(f"  {n} scanned", flush=True)
        if n > N_SCAN:
            break
    from collections import Counter
    print("Status counts:", Counter(r[0] for r in results))
    # 每类挑 1-2 个径迹数适中的事件
    np.random.seed(7)
    for status in ["perfect", "allfound", "partial", "notfound"]:
        cands = [r for r in results if r[0] == status]
        cands = [r for r in cands if 6 <= r[3]["tracks"].x.shape[0] <= 45]
        for j, (stt, idx, graph, evt) in enumerate(cands[:1]):
            fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
            evt_num = evt["EVENTNUMBER"].item() if "EVENTNUMBER" in evt else idx
            chains, pruned_set = get_chain_info(evt, graph)
            summary = "; ".join(f"{c['mother']} ({c['nreco']}/{c['ntot']})" for c in chains) or "no chains"
            A.plot_event_display(axes[0], evt, chains, pruned_set,
                                 f"Event {evt_num}: track display (by truth chain)")
            true_LCA = lca_truth_matrix(evt)
            A.plot_decay_tree(axes[1], true_LCA, get_truth_part_keys(evt).tolist(),
                              "True decay tree",
                              particle_ids=get_truth_part_ids(evt).numpy(), truth=True)
            reco_LCA = lca_reco_matrix(graph, mode="reco")
            A.plot_decay_tree(axes[2], reco_LCA, get_final_keys(graph).tolist(),
                              "Reconstructed decay tree")
            fig.suptitle(f"{status.upper()} - Event {evt_num} (tracks={evt['tracks'].x.shape[0]}, "
                         f"kept={graph['tracks'].x.shape[0]}) | {summary}", fontsize=12)
            fig.tight_layout()
            fn = f"{OUT}/fig01_event_{status}.png"
            fig.savefig(fn, dpi=130)
            plt.close(fig)
            print("saved", fn, "|", summary, flush=True)
    print("done events")


if __name__ == "__main__":
    main()
