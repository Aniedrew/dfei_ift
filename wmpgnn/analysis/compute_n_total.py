"""CPU-only: 统计测试集在**完整图**上的真值 B 链总数 N_total (固定分母)。

与 evaluate 的区别: 只跑真值解码, 不加载模型、不做剪枝、不需要 GPU。
原理: 重建指标的分母原本是"剪枝后仍能解出真值链的 B 数", 随阈值/模型变化;
     这里在未剪枝的原始事件图上跑同样的 lca_truth_matrix + reconstruct_decay,
     得到与模型无关的常数 N_total, 作为所有版本统一的分母。

用法:
    python3 wmpgnn/analysis/compute_n_total.py [config.yaml]
"""
import sys

import yaml

from wmpgnn.analysis.config_adjusting import adjust_config_evaluation
from wmpgnn.data_loader.get_data_loader import load_tst_loader
from wmpgnn.reconstruction.reco_helper import (
    lca_truth_matrix, get_truth_part_keys, get_truth_part_ids, reconstruct_decay,
    particle_name,
)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_files/eval_CERN_noprune.yaml"
    with open(cfg_path) as f:
        configs = yaml.safe_load(f)
    configs = adjust_config_evaluation(configs)
    configs, tst_loader, chunkloader = load_tst_loader(configs)
    chunkloader.num_workers = 0          # 前台跑, 避免多进程开销

    dl = chunkloader.test_dataloader()
    total = 0
    nev = 0
    for bi, batch in enumerate(dl):
        graphs = batch.to_data_list()
        for g in graphs:
            nev += 1
            try:
                true_LCA = lca_truth_matrix(g)
                keys = get_truth_part_keys(g).tolist()
                ids = list(map(particle_name, get_truth_part_ids(g).numpy()))
                tc, _, _ = reconstruct_decay(true_LCA, keys, particle_ids=ids,
                                             truth_level_simulation=1)
                total += len(tc)
            except Exception as e:
                print(f"[warn] event {nev} 真值解码异常: {type(e).__name__}: {e}")
        print(f"  chunk/batch {bi + 1}: events={nev} 累积 N_total={total}", flush=True)

    print(f"\nN_total (完整图真值 B 链总数) = {total}  (events={nev})")


if __name__ == "__main__":
    main()
