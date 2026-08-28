"""
精确复现 to_data_list 不一致: 事件0有165条径迹但反解包图只有6条
"""
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import yaml
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.get_data_loader import load_tst_loader
from wmpgnn.data_loader.weights_calculator import transform_pos_weight

CONFIG = "config_files/train_CERN_normed.yaml"

if __name__ == "__main__":
    with open(CONFIG) as f:
        configs = yaml.safe_load(f)
    configs["model"] = "DFEI"
    configs["DFEI"]["cpt"] = 25
    configs["log_dir"] = "LHCb_logs"
    for k in ["plt_nodes", "plt_edges", "plt_pvs"]:
        configs["inference"].setdefault(k, False)

    configs, tst_loader, chunkloader = load_tst_loader(configs)
    pos_weights = transform_pos_weight(None, None, mode="eval")
    module = load_module(configs, pos_weights)
    module.eval()

    loader = chunkloader.test_dataloader()
    batch = next(iter(loader))
    print("batch keys:", batch.keys())
    # 与 shared_step 一致: use_pid=true 时拼接 pid
    batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
    print("tracks.x (with pid):", tuple(batch["tracks"].x.shape), " batch attr present:", hasattr(batch["tracks"], "batch"))
    tb = batch["tracks"].batch
    nb = int(tb.max()) + 1
    nper = [int((tb == i).sum()) for i in range(nb)]
    print("tracks per event (from batch):", nper)
    print("ptr:", batch["tracks"].ptr.tolist() if hasattr(batch["tracks"], "ptr") else None)

    with torch.no_grad():
        outputs = module.model(batch)
    graphs = outputs.to_data_list()
    print("\nlen(graphs) from to_data_list:", len(graphs))
    for i in range(min(nb, len(graphs))):
        n_tracks = graphs[i]["tracks"].x.shape[0]
        if n_tracks != nper[i]:
            print(f"  MISMATCH event {i}: batch says {nper[i]}, to_data_list gives {n_tracks}")
            print(f"    tracks node keys: {list(graphs[i]['tracks'].keys())}")
            for k in graphs[i]["tracks"].keys():
                if hasattr(graphs[i]["tracks"][k], "shape"):
                    print(f"      {k}: {tuple(graphs[i]['tracks'][k].shape)}")
    print("node_weights len:", len(outputs["node_weights"]))
    print("tracks.x in outputs:", tuple(outputs["tracks"].x.shape))
