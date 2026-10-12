"""方向3 真实数据 CPU 冒烟: 构建真实模型 (v654 底子) + 真实 batch 上跑 shared_step, 核对:
  (1) seqrec_learned: train 一步 -> seqrec 损失入 loss, lam/cut 梯度有限, combined_loss 有限;
  (2) v654 (缺省关闭): train 一步 -> 无 seqrec_params, 无 seqrec 损失 (缺省逐位无影响);
  (3) test 一步 (seqrec_use): 真正跑 _reconstruct_seqrec 解码器, 产出 sig_df。

用法 (CPU):
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python smoke_seqrec_cpu.py
"""
import os
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from wmpgnn.analysis.config_adjusting import adjust_config_training      # noqa: E402
from wmpgnn.analysis.load_module import load_module                      # noqa: E402
from wmpgnn.data_loader.get_data_loader import load_tst_loader           # noqa: E402

PW = {"LCA": torch.ones(4), "nodes": torch.tensor(1.0),
      "edges": torch.tensor(1.0), "pv_asso": torch.tensor(1.0)}


def first_batch(cfg):
    _, _, ckl = load_tst_loader(cfg)
    for b in ckl.test_dataloader():
        return b
    raise RuntimeError("空 test loader")


def build(cfg_path):
    cfg = adjust_config_training(yaml.safe_load(open(cfg_path)))
    cfg["evaluate"]["nfiles"] = [1]
    module = load_module(cfg, PW)
    module.eval()
    return module, cfg


def main():
    print("=" * 70)
    print("[1] seqrec_learned (lam 可学习, metric=mix, seqrec_use=True)")
    module, cfg = build("config_files/train_CERN_0904_seqrec_learned.yaml")
    assert module.seqrec_params is not None, "seqrec_params 未创建"
    print(f"    seqrec_use={module.seqrec_use} metric={module.seqrec_metric} "
          f"lam_learn={module.seqrec_lam_learn} cut_learn={module.seqrec_cut_learn}")
    batch = first_batch(cfg)
    n_ev = int(batch["tracks"].batch.max()) + 1
    print(f"    真实 batch: {n_ev} 事件 / {batch['tracks'].x.shape[0]} 径迹 / "
          f"{batch[('tracks', 'to', 'tracks')].edge_index.shape[1]} tt 边")
    loss = module.shared_step(batch.clone(), 0, module.trn_log, mode="train")
    assert torch.isfinite(loss), "combined_loss 非有限"
    assert module.trn_log.get("seqrec_loss"), "seqrec 损失未入日志 (train 未接线?)"
    print(f"    combined_loss={loss.item():.4f}  seqrec_loss={module.trn_log['seqrec_loss'][-1]:.4f}")
    module.zero_grad(set_to_none=True)
    loss = module.shared_step(batch.clone(), 0, module.trn_log, mode="train")
    loss.backward()
    g_lam = module.seqrec_params.log_lam.grad
    assert g_lam is not None and torch.isfinite(g_lam).all(), "lam 梯度非有限"
    n_bad = [n for n, p in module.named_parameters()
             if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not n_bad, f"非有限梯度: {n_bad[:5]}"
    print(f"    [OK] lam 梯度 finite={bool(torch.isfinite(g_lam).all())}  "
          f"lam.grad={float(g_lam):.4e}  lam={float(module.seqrec_params.lam):.4f}")

    print("=" * 70)
    print("[3] seqrec_learned test 一步 (真正跑 _reconstruct_seqrec 解码器)")
    module.zero_grad(set_to_none=True)
    module.eval()
    module.evt_reco.sig_df, module.evt_reco.evt_df = [], []
    module.shared_step(batch.clone(), 0, module.tst_log, mode="test")
    sig, evt = module.evt_reco.collect_results()
    print(f"    sig_df 行数={len(sig)}  evt_df 行数={len(evt)}")
    assert len(evt) > 0, "test 未产出 evt_df"
    print("    [OK] seqrec 推理路径跑通")

    print("=" * 70)
    print("[2] v654 (缺省关闭) —— 无 seqrec_params / 无 seqrec 损失 (逐位无影响)")
    module2, cfg2 = build("config_files/train_CERN_0904_v654.yaml")
    assert module2.seqrec_params is None, "v654 不应有 seqrec_params"
    assert not module2.seqrec_use and module2.seqrec_loss_weight == 0.0
    batch2 = first_batch(cfg2)
    loss2 = module2.shared_step(batch2.clone(), 0, module2.trn_log, mode="train")
    assert torch.isfinite(loss2)
    assert not module2.trn_log.get("seqrec_loss"), "v654 不应有 seqrec 损失"
    assert isinstance(module2.model._blocks[-1].edge_logits, dict)   # 前向正常
    print(f"    [OK] v654 combined_loss={loss2.item():.4f}, 无 seqrec 损失, 前向正常")

    print("\nALL SEQREC SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
