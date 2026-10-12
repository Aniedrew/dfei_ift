"""方向4 真实数据 CPU 冒烟: 构建真实模型 + 在真实 batch 上跑一次 train shared_step 和一次
推理(test) shared_step, 核对:
  (b) 非候选边被正确排除在损失之外 —— 直接检查 edge_criterion 收到的目标只含候选边;
  (e) 梯度有限 —— combined_loss.backward() 后所有梯度 finite;
  推理门控生效 —— 被 gate 掉的 (非候选) 边边权 = 0 (< edge_prune_thr), 确实不可能被选中;
  打印"被窗口漏掉的真值边比例"。

用法 (CPU):
    export PATH=$HOME/miniconda3/envs/dfei/bin:$PATH
    python smoke_geomcand_cpu.py --config config_files/train_CERN_0904_geomcand_both.yaml
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_files/train_CERN_0904_geomcand_both.yaml")
    a = ap.parse_args()

    cfg = adjust_config_training(yaml.safe_load(open(a.config)))
    cfg["evaluate"]["nfiles"] = [1]          # 只读 1 个 tst 文件, 冒烟够用
    module = load_module(cfg, PW)
    module.eval()
    print(f"[smoke] 模型已构建: train_mask={module.geomcand_train_mask} gate={module.geomcand_gate} "
          f"window dR<={module.geomcand_max_r} doca<={module.geomcand_max_doca} "
          f"dz<={module.geomcand_max_dz}")

    batch = first_batch(cfg)
    n_ev = int(batch["tracks"].batch.max()) + 1
    e = batch[("tracks", "to", "tracks")].edge_index.shape[1]
    print(f"[smoke] 真实 batch: {n_ev} 事件 / {batch['tracks'].x.shape[0]} 径迹 / {e} 条 tt 边")

    # ================= (b) 非候选边不进损失 + (e) 梯度有限 (train) =================
    real_crit = module.edge_criterion
    rec = {}

    def spy(logits, targets):
        rec["n_edges_in_loss"] = int(targets.reshape(-1).numel())
        rec["targets"] = targets.detach().reshape(-1).clone()
        return real_crit(logits, targets)

    # nn.Module.__setattr__ 禁止把非 Module 赋给已有子模块名 -> 直接写入 __dict__ (绕过)
    object.__setattr__(module, "edge_criterion", spy)
    try:
        loss = module.shared_step(batch.clone(), 0, module.trn_log, mode="train")
    finally:
        module.__dict__.pop("edge_criterion", None)   # 回落到 _modules 里的原 criterion
    from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
    y_edges = (batch[("tracks", "to", "tracks")].y.reshape(-1) > 0)
    n_true = int(y_edges.sum())

    if module.geomcand_train_mask:
        keep = DFEILightningModule._geomcand_keep_mask(module, batch) > 0.5
        n_keep = int(keep.sum())
        n_true_keep = int((y_edges & keep).sum())
        print(f"[smoke] 候选边 {n_keep}/{e} (压缩 {e / max(1, n_keep):.2f}x), "
              f"真值边 {n_true}, 候选内 {n_true_keep}, "
              f"被窗口漏掉的真值边比例 = {1 - n_true_keep / max(1, n_true):.4f}")
        # (b) edge_criterion 收到的目标 = y_edges[keep] (非候选边一律不在)
        assert rec["n_edges_in_loss"] == n_keep, \
            f"(b) 损失里边数 {rec['n_edges_in_loss']} != 候选边数 {n_keep}"
        ref = y_edges[keep].to(rec["targets"].dtype)
        assert torch.equal(rec["targets"].reshape(-1), ref.reshape(-1)), "(b) 损失目标 != 候选边标签"
        assert int((rec["targets"] > 0.5).sum()) == n_true_keep, "(b) 损失里出现了窗口外真值边"
        print(f"[b] 非候选边被正确排除在损失外 OK (损失内 {rec['n_edges_in_loss']} 条边 = 候选边)")
        print(f"[smoke] trn_log 中累计的漏边比例样本 = "
              f"{module.trn_log.get('geomcand_true_missed_frac')}")
    else:
        # 默认关闭 -> 损失统计必须用**全部**边, 逐位无影响
        assert rec["n_edges_in_loss"] == e, \
            f"[off] train_mask 关闭时损失只用 {rec['n_edges_in_loss']}/{e} 条边 (应全用)"
        assert torch.equal(rec["targets"].reshape(-1), y_edges.to(rec["targets"].dtype).reshape(-1)), \
            "[off] train_mask 关闭时损失目标 != 全部边标签"
        print(f"[off] train_mask 关闭: 损失用全部 {e} 条边 (缺省逐位无影响) OK")

    # (e) 梯度有限
    loss.backward()
    n_grad = 0
    bad = []
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        n_grad += 1
        if not torch.isfinite(p.grad).all():
            bad.append(name)
    assert n_grad > 0, "(e) 没有任何参数拿到梯度"
    assert not bad, f"(e) 梯度非有限: {bad[:5]}"
    print(f"[e] 梯度有限 OK ({n_grad} 个参数有梯度, 全部 finite); combined_loss={loss.item():.4f}")

    # ================= 推理 (test) 门控生效 =================
    if not module.geomcand_gate:
        print("[gate] gate 关闭 -> 跳过 (缺省逐位无影响)")
        return
    module.zero_grad(set_to_none=True)
    module.eval()
    captured = {}
    real_reco = module.evt_reco.reconstruct_heavyhadrons
    module.evt_reco.reconstruct_heavyhadrons = lambda outputs, pv_des=None: captured.update(out=outputs)
    try:
        module.shared_step(batch.clone(), 0, module.tst_log, mode="test")
    finally:
        module.evt_reco.reconstruct_heavyhadrons = real_reco

    ew = captured["out"]["edge_weights"].reshape(-1)
    gk = module._geomcand_keep.to(ew.device).reshape(-1)
    thr = float(module.edge_prune)
    n_out = int((gk < 0.5).sum())
    max_out = float(ew[gk < 0.5].abs().max()) if n_out else 0.0
    n_out_selected = int(((gk < 0.5) & (ew >= thr)).sum())
    n_in_selected = int(((gk > 0.5) & (ew >= thr)).sum())
    print(f"[gate] 非候选边 {n_out} 条: max|边权|={max_out:.6g}, 过阈(>={thr}) 的条数={n_out_selected} "
          f"(应为 0); 候选边内过阈={n_in_selected}")
    assert n_out_selected == 0, "门控失效: 非候选边仍可过剪枝阈值被选中"
    assert n_out == 0 or max_out == 0.0, "门控失效: 非候选边边权非 0"
    print("[gate] 推理门控生效 OK: 被门控的边边权=0, 不可能被选中")


if __name__ == "__main__":
    main()
