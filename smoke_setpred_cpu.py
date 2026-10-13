"""方向2 真实数据 CPU 冒烟: 构建真实模型 (v654 底子) + 真实 batch 上跑 shared_step, 核对:
  (1) setpred_k4: train 一步 -> setpred 损失入 loss, 头梯度有限, combined_loss 有限;
  (2) setpred_decode: test 一步 -> 真正跑 _reconstruct_setpred 解码器 (取代连通分量), 产出 sig_df;
  (3) v654 (缺省关闭): train 一步 -> 无 set_pred_head / 无 setpred 损失 (缺省逐位无影响)。

用法 (CPU):
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python smoke_setpred_cpu.py
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
    print("[1] setpred_k4 (K=4, softmax_bg, setpred_use=True, decode=False) —— train 一步")
    module, cfg = build("config_files/train_CERN_0904_setpred_k4.yaml")
    assert module.set_pred_head is not None, "set_pred_head 未创建"
    print(f"    use={module.setpred_use} decode={module.setpred_decode} w={module.setpred_loss_w} "
          f"K={module.setpred_nslots} pairwise={module.setpred_pairwise} assign={module.setpred_assign}")
    batch = first_batch(cfg)
    n_ev = int(batch["tracks"].batch.max()) + 1
    print(f"    真实 batch: {n_ev} 事件 / {batch['tracks'].x.shape[0]} 径迹 / "
          f"{batch[('tracks', 'to', 'tracks')].edge_index.shape[1]} tt 边")
    loss = module.shared_step(batch.clone(), 0, module.trn_log, mode="train")
    assert torch.isfinite(loss), "combined_loss 非有限"
    assert module.trn_log.get("setpred_loss"), "setpred 损失未入日志 (train 未接线?)"
    print(f"    combined_loss={loss.item():.4f}  setpred_loss={module.trn_log['setpred_loss'][-1]:.4f}")
    print(f"    diag={module._setpred_last_diag}")
    module.zero_grad(set_to_none=True)
    loss = module.shared_step(batch.clone(), 0, module.trn_log, mode="train")
    loss.backward()
    n_bad = [n for n, p in module.named_parameters()
             if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not n_bad, f"非有限梯度: {n_bad[:5]}"
    hs = [n for n, p in module.set_pred_head.named_parameters()
          if p.grad is not None and p.grad.abs().sum() > 0]
    assert hs, "set_pred_head 未收到梯度 (未进计算图?)"
    print(f"    [OK] 全参梯度有限; 头内 {len(hs)} 个参数收到非零梯度")

    print("=" * 70)
    print("[2] setpred_decode (K=4, setpred_decode=True) test 一步 (真正跑 _reconstruct_setpred 解码器)")
    module2, cfg2 = build("config_files/train_CERN_0904_setpred_decode.yaml")
    assert module2.setpred_decode, "setpred_decode 未打开"
    module2.zero_grad(set_to_none=True)
    module2.eval()
    module2.evt_reco.sig_df, module2.evt_reco.evt_df = [], []
    batch2 = first_batch(cfg2)
    with torch.no_grad():
        module2.shared_step(batch2.clone(), 0, module2.tst_log, mode="test")
    sig, evt = module2.evt_reco.collect_results()
    print(f"    sig_df 行数={len(sig)}  evt_df 行数={len(evt)}")
    assert len(evt) > 0, "test 未产出 evt_df"
    print("    [OK] setpred 推理路径跑通 (槽位解码取代连通分量)")

    print("=" * 70)
    print("[3] 对照臂 setpred_node_mask (mask 建在节点池化上) —— train 一步")
    module3, cfg3 = build("config_files/train_CERN_0904_setpred_node_mask.yaml")
    assert module3.setpred_pairwise is False, "pairwise=False 未生效"
    batch3 = first_batch(cfg3)
    loss3 = module3.shared_step(batch3.clone(), 0, module3.trn_log, mode="train")
    assert torch.isfinite(loss3) and module3.trn_log.get("setpred_loss")
    print(f"    [OK] node_mask combined_loss={loss3.item():.4f} "
          f"setpred_loss={module3.trn_log['setpred_loss'][-1]:.4f}")

    print("=" * 70)
    print("[4] v654 (缺省关闭) —— 无 set_pred_head / 无 setpred 损失 (逐位无影响)")
    module2, cfg2 = build("config_files/train_CERN_0904_v654.yaml")
    assert module2.set_pred_head is None, "v654 不应有 set_pred_head"
    assert not module2.setpred_use and module2.setpred_loss_w == 0.0
    batch2 = first_batch(cfg2)
    loss2 = module2.shared_step(batch2.clone(), 0, module2.trn_log, mode="train")
    assert torch.isfinite(loss2)
    assert not module2.trn_log.get("setpred_loss"), "v654 不应有 setpred 损失"
    assert isinstance(module2.model._blocks[-1].edge_logits, dict)   # 前向正常
    # [逐位一致] 本方向新增的"边表征暂存"钩子必须**惰性**: 第一次调用时 _stash_emb 尚未置位
    # (暂存关闭), 第二次已被 chain_contrast 置 True (暂存打开) -> 两次 combined_loss 必须逐位相同。
    _l1 = float(loss2)
    _l2 = float(module2.shared_step(batch2.clone(), 0, module2.trn_log, mode="train"))
    assert abs(_l1 - _l2) < 1e-6, f"边表征暂存钩子非惰性 (改变了损失): {_l1} vs {_l2}"
    print(f"    [OK] v654 combined_loss={_l1:.4f}, 无 setpred 损失, 前向正常; "
          f"暂存钩子惰性 (两调用 {_l1:.6f} / {_l2:.6f})")

    print("\nALL SETPRED SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
