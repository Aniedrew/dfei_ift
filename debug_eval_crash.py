"""
复现 CERN新数据(version_25) evaluate 阶段的 IndexError 崩溃
定位是哪个事件/哪类结构触发 (mask [46] vs tensor [0])

方案: monkeypatch module.test_step, 崩溃时打印batch结构, 然后走官方 trainer.test 路径
"""
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import yaml
from pytorch_lightning import Trainer
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.get_data_loader import load_tst_loader
from wmpgnn.data_loader.weights_calculator import transform_pos_weight

CONFIG = "config_files/train_CERN_normed.yaml"

if __name__ == "__main__":
    with open(CONFIG) as f:
        configs = yaml.safe_load(f)

    # 手动补齐评估所需键 (模拟 adjust_config_evaluation 的默认值)
    configs["model"] = "DFEI"
    configs["DFEI"]["cpt"] = 25  # version_25
    configs["log_dir"] = "LHCb_logs"
    for k in ["plt_nodes", "plt_edges", "plt_pvs"]:
        configs["inference"].setdefault(k, False)

    configs, tst_loader, chunkloader = load_tst_loader(configs)
    pos_weights = transform_pos_weight(None, None, mode="eval")
    module = load_module(configs, pos_weights)
    module.eval()

    # monkeypatch test_step 捕获崩溃
    import copy
    orig_test_step = module.test_step

    def safe_test_step(batch, batch_idx):
        snapshot = copy.deepcopy(batch)
        try:
            return orig_test_step(batch, batch_idx)
        except IndexError as e:
            print(f"\n>>> CRASH at batch_idx={batch_idx}: {e}")
            tracks = batch["tracks"]
            nb = int(tracks.batch.max().item()) + 1
            print("tracks.x:", tuple(tracks.x.shape), " n_events:", nb)
            nper = [int((tracks.batch == i).sum()) for i in range(nb)]
            print("tracks per event:", nper)
            print("events with 0 tracks:", [i for i, n in enumerate(nper) if n == 0])
            if "pvs" in batch.node_types:
                print("pvs.x:", tuple(batch["pvs"].x.shape))
            if "EVENTNUMBER" in batch.keys():
                ev = batch["EVENTNUMBER"]
                print("EVENTNUMBER:", ev.tolist() if ev.dim() else ev.item())

            # 深度诊断: 用未修改的snapshot重跑模型, 对比 block.node_weights / node key shapes
            try:
                snap = copy.deepcopy(snapshot)
                if module.use_pid == "true":
                    snap["tracks"].x = torch.cat([snap["tracks"].x, snap["tracks"].pid], dim=1)
                with torch.no_grad():
                    out = module.model(snap)
                graphs = out.to_data_list()
                block = module.model._blocks[-1]
                nw = block.node_weights["tracks"].squeeze()
                tb = snap["tracks"].batch
                nb2 = int(tb.max()) + 1
                print(f"\n[deep] len(node_weights)={len(nw)}  len(track_batch)={len(tb)}  len(graphs)={len(graphs)}")
                for i in range(nb2):
                    mask_len = int((tb == i).sum())
                    g = graphs[i] if i < len(graphs) else None
                    if g is None:
                        print(f"[deep] event {i}: NO GRAPH (mask_len={mask_len})")
                        continue
                    for k in g["tracks"].keys():
                        t = g["tracks"][k]
                        if hasattr(t, "shape") and t.shape[0] != mask_len and k not in ("ptr",):
                            print(f"[deep] event {i} (mask_len={mask_len}): node key '{k}' shape {tuple(t.shape)} MISMATCH")
            except Exception as e2:
                print("[deep] re-run failed:", type(e2).__name__, e2)
            raise

    module.test_step = safe_test_step

    trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=True, num_sanity_val_steps=0)
    trainer.test(module, dataloaders=chunkloader.test_dataloader())
    print("done (no crash through full test set)")
