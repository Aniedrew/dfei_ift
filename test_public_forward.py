#!/usr/bin/env python3
"""端到端验证 (CPU): 公开数据 + v38 算法栈 (B2/source_head/chain_lca/class2加权) 前向正常。"""
import sys, os
import torch
import yaml

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
sys.path.insert(0, BASE)

from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.lightning_module.lightning_helper import init_logs
from wmpgnn.data_loader.helper import load_dataset
from wmpgnn.analysis.load_module import load_module
from torch_geometric.data import Batch

torch.manual_seed(0)
cfg = yaml.safe_load(open(f"{BASE}/config_files/train_public_v38.yaml"))
cfg["settings"]["nfiles"] = [1]
cfg["settings"]["epochs"] = 1
cfg["settings"]["batch_size"] = 1
cfg["settings"]["ncpu"] = 1
cfg["settings"]["resume_ckpt"] = None
cfg = adjust_config_training(cfg)

files = sorted([f for f in os.listdir(cfg["settings"]["data_dir"] + "/00342442_inclusive")
                if f.startswith("trn_")])[:1]
path = cfg["settings"]["data_dir"] + "/00342442_inclusive/" + files[0]
evts, weights = load_dataset(path, cfg, mode="weights")
evts = evts[:2]
b = Batch.from_data_list(evts)
print(f"[data] {len(evts)} events, tracks={b['tracks'].x.shape}, pvs={b['pvs'].x.shape}, "
      f"tt_edges={b[('tracks','to','tracks')].edges.shape}")

pos_weights = transform_pos_weight(weights, cfg["inference"])
print("[weights] LCA pos:", pos_weights["LCA"].tolist())
module = load_module(cfg, pos_weights)
module.train()

log = init_logs(cfg, mode="train")[0]
try:
    loss = module.shared_step(b, 0, log, mode="train")
    print(f"[ok] shared_step 完成, combined_loss={loss.item():.4f} finite={torch.isfinite(loss).item()}")
    print(f"     LCA={log['LCA_loss'][-1]:.4f} tt={log['tt_edges_loss'][-1]:.4f}")
    if "chain_lca_loss" in log and log["chain_lca_loss"]:
        print(f"     chain_lca_loss={log['chain_lca_loss'][-1]:.4f}")
    if "source_loss" in log and log["source_loss"]:
        print(f"     source_loss={log['source_loss'][-1]:.4f}")
    assert torch.isfinite(loss), "combined_loss 非有限!"
    print("[ok] 公开数据 v38 前向通过")
except Exception:
    import traceback
    traceback.print_exc()
    print("[FAIL] 失败")
