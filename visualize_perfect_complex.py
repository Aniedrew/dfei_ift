"""
找出并可视化一个"多衰变链 + 中间态 + 完美重建"的事件 (官方事件级perfect)
验证模型确实完整重建了类似89921802结构的复杂事件
"""
import glob
import io
import os
import sys

import numpy as np
import torch
import zstandard as zstd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import analyze_reco_visual as A

TARGET_EVT = 172  # 官方: 2链, Gen2=1, 深度6, PerfectEventReconstruction=1


def find_event(events, num):
    for e in events:
        if e["EVENTNUMBER"].item() == num:
            return e
    return None


def main():
    # 加载足够多的事件找 TARGET_EVT
    n = 2000
    while True:
        events = A.load_events(n)
        evt = find_event(events, TARGET_EVT)
        if evt is not None:
            break
        print(f"not found in first {n}, extending...")
        n *= 2
        if n > 20000:
            print("NOT FOUND"); return

    print(f"=== Event {TARGET_EVT}: {evt['tracks'].x.shape[0]} tracks, "
          f"{evt['pvs'].x.shape[0]} PVs ===")

    module, _ = A.load_module(None)
    outputs, batch = A.predict(module, evt)
    graph = A.classify_event(evt, outputs)
    status = A.classify_reco(graph)  # 严格事件级
    print(f"严格事件级分类: {status}")
    print(f"剪枝后保留径迹: {graph['tracks'].x.shape[0]} / {evt['tracks'].x.shape[0]}")

    chains, pruned_set = A.get_chain_info(evt, graph)
    print("衰变链:")
    for ch in chains:
        lost = ch["ntot"] - ch["nreco"]
        print(f"  {ch['mother']}: nreco={ch['nreco']}/{ch['ntot']} "
              f"tracks={ch['tracks']} names={ch['names']}")

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4))
    A.plot_event_display(axes[0], evt, chains, pruned_set,
                         f"Event {TARGET_EVT}: track display (by truth chain)")

    true_LCA = A.lca_truth_matrix(evt)
    A.plot_decay_tree(axes[1], true_LCA, evt["truth_part_keys"].tolist(),
                      "True decay tree",
                      particle_ids=evt["truth_part_ids"].numpy(), truth=True)

    reco_LCA = A.lca_reco_matrix(graph, mode="reco")
    A.plot_decay_tree(axes[2], reco_LCA, graph["final_keys"].tolist(),
                      "Reconstructed decay tree")

    chain_summary = "; ".join(f"{ch['mother']} ({ch['nreco']}/{ch['ntot']})" for ch in chains)
    fig.suptitle(f"PERFECT multi-chain event - Event {TARGET_EVT} "
                 f"(tracks={evt['tracks'].x.shape[0]}, pruned={graph['tracks'].x.shape[0]}) | chains: {chain_summary}",
                 fontsize=12)
    fig.tight_layout()
    out = f"reco_analysis/evt_perfect_multichain_event{TARGET_EVT}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
