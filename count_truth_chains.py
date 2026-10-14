"""统计"完整图上的真值链总数" N_total —— 固定分母。

为什么需要它
------------
上游旧 benchmark 的分母是"剪枝后仍存在的真值链数"（`signal_reco_df` 的行数），
**随模型/阈值变化**（0904 线实测 6089~9516），所以百分比不可跨版本比较。
无条件口径要把分母固定为**完整图（不剪枝）上的真值链总数** N_total。

为什么不用跑完整评测
--------------------
N_total 只依赖数据与真值解码，**与模型无关**：
    particle_keys = get_truth_part_keys(graph)      # 真值末态径迹
    true_LCA      = lca_truth_matrix(graph)
    tc_dict       = reconstruct_decay(true_LCA, particle_keys, ..., truth_level_simulation=1)
    N_total       = Σ_events len(tc_dict)
因此本脚本只做数据加载 + 真值解码，**不加载模型、不做重建**，CPU 上前台即可跑完。

输出
----
report_figs/truth_chain_total_<sample>.csv   # 逐事件: RUN/EVENT, 真值链数, 真值径迹数
并打印: N_total、按 1B/2B/>2B 分层的分母、总真值径迹数。

用法
----
    python3 count_truth_chains.py --config config_files/eval_CERN_0904_noprune.yaml
    python3 count_truth_chains.py --config config_files/eval_CERN_0904_noprune.yaml --chunks 1  # 快速自测
"""
import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from wmpgnn.analysis.config_adjusting import adjust_config_evaluation
from wmpgnn.data_loader.get_data_loader import load_tst_loader
from wmpgnn.reconstruction.reco_helper import (get_truth_part_keys, get_truth_part_ids,
                                               lca_truth_matrix, reconstruct_decay)
from wmpgnn.reconstruction.signal_dict import particle_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--chunks", type=int, default=0, help="只跑前 N 个 chunk (自测用)")
    args = ap.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)
    configs = adjust_config_evaluation(configs)
    sample = configs["evaluate"]["sample"][0] if isinstance(configs["evaluate"]["sample"], list) \
        else configs["evaluate"]["sample"]

    configs, _, chunkloader = load_tst_loader(configs)
    if chunkloader is None:
        raise SystemExit("本脚本只支持 chunk loader (CERN/公开数据)")

    rows = []
    n_chunk = 0
    for chunk in chunkloader.test_dataloader():
        n_chunk += 1
        evts = chunk.to_data_list()
        n_keys = 0
        for evt in evts:
            keys = get_truth_part_keys(evt).tolist()
            if len(keys) == 0:
                continue
            n_keys += 1
            ids = list(map(particle_name, get_truth_part_ids(evt).numpy()))
            try:
                tc_dict, _, _ = reconstruct_decay(lca_truth_matrix(evt), keys,
                                                  particle_ids=ids, truth_level_simulation=1)
            except Exception as e:                      # 逐事件容错, 不中断全样本
                print(f"  [warn] 真值解码失败 ({type(e).__name__}): {e}")
                continue
            n_chains = len(tc_dict)
            n_tracks = sum(len(tc["node_keys"]) for tc in tc_dict.values())
            rows.append({
                "RUNNUMBER": int(evt["RUNNUMBER"].item()) if "RUNNUMBER" in evt.keys() else -1,
                "EVENTNUMBER": int(evt["EVENTNUMBER"].item()) if "EVENTNUMBER" in evt.keys() else -1,
                "n_truth_chains": n_chains, "n_truth_tracks": n_tracks,
            })
        print(f"  chunk {n_chunk}: chunk内事件 {len(evts)}  有真值键 {n_keys}  "
              f"累计事件 {len(rows)}  累计真值链 {sum(r['n_truth_chains'] for r in rows)}")
        sys.stdout.flush()
        if args.chunks and n_chunk >= args.chunks:
            break

    d = pd.DataFrame(rows)
    n_total = int(d["n_truth_chains"].sum())
    d["stratum"] = d["n_truth_chains"].map(lambda n: "1B" if n == 1 else ("2B" if n == 2 else ">2B"))
    d.loc[d["n_truth_chains"] == 0, "stratum"] = "0链(解码空)"
    strat = d.groupby("stratum")["n_truth_chains"].sum().to_dict()
    print("\n================ 结果 ================")
    print(f"样本            : {sample}")
    print(f"事件数(有真值链) : {len(d)}")
    print(f"N_total         : {n_total}   <- 固定分母")
    print(f"分层分母(真值链) : 1B={int(strat.get('1B',0))}  2B={int(strat.get('2B',0))}  >2B={int(strat.get('>2B',0))}")
    print(f"真值径迹总数     : {int(d['n_truth_tracks'].sum())}")
    print("分层事件数       : " + "  ".join(f"{k}={int((d['stratum']==k).sum())}" for k in ("1B", "2B", ">2B")))
    out = f"report_figs/truth_chain_total_{sample}.csv"
    os.makedirs("report_figs", exist_ok=True)
    d.to_csv(out, index=False)
    print(f"已写出           : {out}")


if __name__ == "__main__":
    main()
