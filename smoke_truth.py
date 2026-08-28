"""
冒烟测试: lca_truth_matrix + reconstruct_decay 对旧格式(truth_*)和新格式(sig_keys)都工作
"""
import glob
import io
import sys
import os

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from wmpgnn.reconstruction.reco_helper import (lca_truth_matrix, reconstruct_decay,
                                               get_final_keys, get_truth_part_keys, get_truth_part_ids)
from wmpgnn.reconstruction.signal_dict import particle_name


def load_first(files):
    dctx = zstd.ZstdDecompressor()
    with open(files[0], "rb") as fh:
        with dctx.stream_reader(fh) as r:
            data = torch.load(io.BytesIO(r.read()), weights_only=False)
    return data


# ===== 旧格式 =====
old_files = sorted(glob.glob('/lzufs/user/guoqingxiang/DFEI_data/CERN_data_LHCb/00342442_inclusive/tst_data_*'))
old_evts = load_first(old_files)
print(f"OLD format: {len(old_evts)} events")
for evt in old_evts[:3]:
    tl = lca_truth_matrix(evt)
    fk = get_final_keys(evt)
    tk = get_truth_part_keys(evt)
    print(f"  evt truth_lca rows={len(tl)} final_keys={len(fk)} truth_part_keys={len(tk)}")
    keys = tk.tolist()
    pids = list(map(particle_name, get_truth_part_ids(evt).numpy()))
    tc, _, _ = reconstruct_decay(tl, keys, particle_ids=pids, truth_level_simulation=1)
    print(f"    -> truth chains: {len(tc)}")

# ===== 新格式 =====
new_files = sorted(glob.glob('/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_*'))
new_evts = load_first(new_files)
print(f"\nNEW format: {len(new_evts)} events")
for evt in new_evts[:3]:
    tl = lca_truth_matrix(evt)
    fk = get_final_keys(evt)
    tk = get_truth_part_keys(evt)
    print(f"  evt truth_lca rows={len(tl)} final_keys(part_keys)={len(fk)} truth_part_keys(sig_keys)={len(tk)}")
    print(f"    LCA_dec values: {tl['LCA_dec'].tolist() if len(tl) else '[]'}")
    keys = tk.tolist()
    pids = list(map(particle_name, get_truth_part_ids(evt).numpy()))
    tc, nco, depth = reconstruct_decay(tl, keys, particle_ids=pids, truth_level_simulation=1)
    print(f"    -> truth chains: {len(tc)}, max_depth={depth}")
    for ckey, c in tc.items():
        print(f"       chain {ckey}: node_keys={c['node_keys']} LCA={c['LCA_values']}")
print("\nSMOKE TEST PASSED")
