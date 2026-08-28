#!/usr/bin/env python3
"""端到端验证 (CPU): v41 完整模型 + shared_step (train/val 都切子图, 课程过渡, cap)。

验证点:
  1. train: pv_cluster_head loss + 课程 alpha=1.0(纯 truth 分簇) -> 切子图 -> 前向正常
  2. val:   同样切子图 (确定性分配, 无 Gumbel), 无 cluster loss
  3. 各 loss 有限, combined_loss 有限
"""
import sys, os
import torch
import yaml

BASE = "/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn"
sys.path.insert(0, BASE)

from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.lightning_helper import init_logs
from wmpgnn.data_loader.helper import load_dataset
from wmpgnn.analysis.load_module import load_module
from torch_geometric.data import Batch

torch.manual_seed(0)
cfg = yaml.safe_load(open(f"{BASE}/config_files/train_CERN_v41.yaml"))
cfg["settings"]["nfiles"] = [1]
cfg["settings"]["epochs"] = 1
cfg["settings"]["batch_size"] = 2
cfg["settings"]["ncpu"] = 1
cfg["settings"]["resume_ckpt"] = None
cfg = adjust_config_training(cfg)

files = sorted([f for f in os.listdir(cfg["settings"]["data_dir"] + "/inclusive_00342442")
                if f.startswith("trn_")])[:1]
path = cfg["settings"]["data_dir"] + "/inclusive_00342442/" + files[0]
evts, weights = load_dataset(path, cfg, mode="weights")
evts = evts[:4]
b = Batch.from_data_list(evts)
print(f"[data] {len(evts)} events, 拆分前 tracks={b['tracks'].x.shape[0]}, pvs={b['pvs'].x.shape[0]}")

pos_weights = transform_pos_weight(weights, cfg["inference"])
module = load_module(cfg, pos_weights)
module.train()

# 课程 alpha 初始应为 1.0 (run 起点, 纯 truth 分簇)
print(f"[curriculum] run_epoch={module._pv_run_epoch} alpha={module.pv_curriculum_alpha:.3f} "
      f"tau={module.pv_tau:.3f}")

for mode in ("train", "val"):
    if mode == "train":
        module.train()
        log = init_logs(cfg, mode="train")[0]
    else:
        module.eval()  # 真实流程里 val 是 eval 模式 (无 B2 掩码)
        from collections import defaultdict
        log = defaultdict(list)  # 真实流程 val_log 是 defaultdict(list)
    try:
        loss = module.shared_step(b, 0, log, mode=mode)
        print(f"[ok] mode={mode}: combined_loss={loss.item():.4f} finite={torch.isfinite(loss).item()}")
        if mode == "train" and "pv_cluster_loss" in log and log["pv_cluster_loss"]:
            print(f"     pv_cluster_loss={log['pv_cluster_loss'][-1]:.4f} "
                  f"LCA={log['LCA_loss'][-1]:.4f} tt={log['tt_edges_loss'][-1]:.4f}")
        assert torch.isfinite(loss), f"mode={mode} combined_loss 非有限!"
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] mode={mode} 训练前向失败")
        raise

# 验证课程混合: alpha=0.5 时应有 truth 与 cluster 混合
module.pv_curriculum_alpha = 0.5
logits = module._pv_cluster_scores(b)
sb = module._split_by_pv(b, logits=logits, sample=False, cap=0)
print(f"[curriculum] alpha=0.5 时子图数: {int(sb['tracks'].batch.max())+1} (含 truth+cluster 混合分配)")
print("[ok] 全部通过")
