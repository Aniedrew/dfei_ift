"""
扫描测试数据, 找一个"事件级perfect + 多衰变链 + 含中间态"的事件并可视化
(类似89921802但被完美重建的例子)

逻辑: 对每个事件做模型推理 -> 严格事件级分类 -> 检查链数>=2 且 至少一条链
有中间态 (链内最大LCA generation > 1)。找到第一个就画图停止。
"""
import glob
import io
import os
import sys
import time

import numpy as np
import torch
import zstandard as zstd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import analyze_reco_visual as A


def chain_max_lca(evt, chain):
    """链内最大 LCA generation (1=直接衰变无中间态, >1=有中间态)"""
    true_LCA = A.lca_truth_matrix(evt)
    particle_keys = evt["truth_part_keys"].tolist()
    key_pos = {k: i for i, k in enumerate(particle_keys)}
    pos_set = {key_pos[k] for k in chain["keys"] if k in key_pos}
    rows = true_LCA[(true_LCA['senders'].isin(pos_set)) & (true_LCA['receivers'].isin(pos_set))]
    if len(rows) == 0:
        return 1
    return int(rows['LCA_dec'].max())


def main():
    max_scan = int(os.environ.get("MAX_SCAN", "4000"))
    t0 = time.time()
    module, _ = A.load_module(None)

    files = sorted(glob.glob(f"{A.DATA_DIR}/{A.SAMPLE}/tst_data_*"))
    dctx = zstd.ZstdDecompressor()
    n_evt = 0
    for f in files:
        with open(f, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                data = torch.load(io.BytesIO(reader.read()), weights_only=False)
        for evt in data:
            # 双向化
            et = ("tracks", "to", "tracks")
            store = evt[et]
            store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
            store.edges = store.edges.repeat(2, 1)
            store.y = store.y.repeat(2)

            n_evt += 1
            outputs, batch = A.predict(module, evt)
            graph = A.classify_event(evt, outputs)
            status = A.classify_reco(graph)
            if status != "perfect":
                if n_evt % 200 == 0:
                    print(f"[{n_evt} scanned, {time.time()-t0:.0f}s] last status={status}", flush=True)
                continue

            chains, pruned_set = A.get_chain_info(evt, graph)
            if len(chains) < 2:
                continue
            max_lcas = [chain_max_lca(evt, ch) for ch in chains]
            if max(max_lcas) <= 1:
                continue

            # 找到!
            evt_num = evt["EVENTNUMBER"].item()
            print(f"\n=== FOUND at event #{n_evt}: Event {evt_num} "
                  f"(elapsed {time.time()-t0:.0f}s) ===", flush=True)
            print(f"tracks={evt['tracks'].x.shape[0]}, pruned={graph['tracks'].x.shape[0]}")
            for ch, ml in zip(chains, max_lcas):
                print(f"  chain {ch['mother']}: nreco={ch['nreco']}/{ch['ntot']} "
                      f"maxLCA={ml} names={ch['names']}")

            # 画图
            fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
            A.plot_event_display(axes[0], evt, chains, pruned_set,
                                 f"Event {evt_num}: track display (by truth chain)")
            true_LCA = A.lca_truth_matrix(evt)
            A.plot_decay_tree(axes[1], true_LCA, evt["truth_part_keys"].tolist(),
                              "True decay tree",
                              particle_ids=evt["truth_part_ids"].numpy(), truth=True)
            reco_LCA = A.lca_reco_matrix(graph, mode="reco")
            A.plot_decay_tree(axes[2], reco_LCA, graph["final_keys"].tolist(),
                              "Reconstructed decay tree")
            chain_summary = "; ".join(
                f"{ch['mother']} ({ch['nreco']}/{ch['ntot']})" for ch in chains)
            fig.suptitle(f"PERFECT multi-chain+intermediate - Event {evt_num} "
                         f"(tracks={evt['tracks'].x.shape[0]}, pruned={graph['tracks'].x.shape[0]}) | chains: {chain_summary}",
                         fontsize=12)
            fig.tight_layout()
            out = f"reco_analysis/evt_perfect_multichain_event{evt_num}.png"
            fig.savefig(out, dpi=130)
            plt.close(fig)
            print(f"已保存: {out}", flush=True)
            return

        if n_evt >= max_scan:
            break

    print(f"扫描了 {n_evt} 个事件, 未找到符合条件的例子 (max_scan={max_scan})")


if __name__ == "__main__":
    main()
