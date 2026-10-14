"""组合臂冒烟: 真实 batch 跑一次 shared_step(mode="train"), 并确认新键被代码真正读取。"""
import sys

import torch
import yaml

from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.get_data_loader import load_tst_loader
from wmpgnn.data_loader.weights_calculator import transform_pos_weight

CFGS = sys.argv[1:] or [
    "config_files/train_CERN_0904_combo_all.yaml",
    "config_files/train_CERN_0904_combo_noseqrec.yaml",
    "config_files/train_CERN_0904_base_ep120.yaml",
    "config_files/train_CERN_0904_base_seqrec.yaml",
]

for p in CFGS:
    tag = p.split("/")[-1][:-5]
    cfg = yaml.safe_load(open(p))
    cfg = adjust_config_training(cfg)
    cfg, tst, chunk = load_tst_loader(cfg)
    module = load_module(cfg, transform_pos_weight(None, cfg["inference"], mode="eval"))
    module.eval()
    batch = next(iter(chunk.test_dataloader()))
    try:
        out = module.shared_step(batch, 0, module.trn_log, mode="train")
        loss = out["loss"] if isinstance(out, dict) else out
        lv = float(loss)
        ok = bool(torch.isfinite(torch.tensor(lv)))
    except Exception as e:
        print(f"[FAIL] {tag}: {type(e).__name__}: {e}")
        continue
    keys = {k: getattr(module, k, "—") for k in
            ("chain_recall_weight", "chain_recall_tau", "seqrec_use", "seqrec_metric", "seqrec_lam_learn")}
    n_param = sum(1 for n, pp in module.named_parameters() if "seqrec" in n or "lam" in n)
    print(f"[OK] {tag}: combined_loss={lv:.4f} finite={ok}  epochs={cfg['settings'].get('epochs')}")
    print(f"     读取到的键: {keys}   seqrec相关参数数={n_param}")
    sys.stdout.flush()
