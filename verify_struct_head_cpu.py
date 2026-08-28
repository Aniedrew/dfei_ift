"""CPU 验证: 节点结构监督 (depth + RC) —— truth_chain_structure 标签 + shared_step 全路径。"""
import sys, io
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')

import yaml
import torch
import zstandard as zstd
from torch_geometric.data import Batch

from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.reconstruction.topk_selection import truth_chain_structure

torch.manual_seed(0)

# ---- 1) truth_chain_structure 标签合理性 (真实事件) ----
DATA = '/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00220000_00220999.pt.zst'
dctx = zstd.ZstdDecompressor()
with open(DATA, 'rb') as f:
    with dctx.stream_reader(f) as reader:
        data = torch.load(io.BytesIO(reader.read()), weights_only=False)

# 双向化 (与 load_dataset 一致)
for evt in data[:10]:
    store = evt[('tracks', 'to', 'tracks')]
    store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
    store.edges = store.edges.repeat(2, 1)
    if getattr(store, 'y', None) is not None:
        store.y = store.y.repeat(2)

evt = data[0]
st = truth_chain_structure(evt[('tracks','to','tracks')].y, evt[('tracks','to','tracks')].edge_index,
                           torch.zeros(evt['tracks'].x.shape[0], dtype=torch.long), torch.device('cpu'))
n_in = st['in_chain'].sum().item()
print('[1] 事件0 链内节点数:', n_in, '/', evt['tracks'].x.shape[0])
print('[1] 链根节点数:', st['root'].sum().item())
if n_in > 0:
    m = st['in_chain']
    print('[1] depth 分布:', st['depth'][m].tolist())
    print('[1] rc 分布  :', [round(v, 3) for v in st['rc'][m].tolist()])

# ---- 2) shared_step 全路径 (struct_head + mass_head 组合配置, 从 masshead2 best 续训) ----
with open('config_files/train_CERN_v38_structhead.yaml') as f:
    configs = yaml.safe_load(f)
configs = adjust_config_training(configs)
CKPT2 = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/LHCb_logs/DFEI/version_47/checkpoints/best-epoch=113-val_combined_loss=35.677.ckpt'
configs['DFEI']['cpt'] = CKPT2
pos_weights = transform_pos_weight(None, None, mode="eval")
module = load_module(configs, pos_weights)
print('[2] struct_head_on:', module.struct_head_on)
print('[2] mass_head_on:', module.mass_head_on)
print('[2] mom_head_on:', module.mom_head_on)
print('[2] node_struct_head 挂载:', hasattr(module.model, 'node_struct_head'))
print('[2] node_mom_head 挂载:', hasattr(module.model, 'node_mom_head'))

batch = Batch.from_data_list(data[:2])
module.train()
loss = module.shared_step(batch, 0, module.trn_log, mode='train')
print('[3] combined_loss:', float(loss))
print('[3] struct_loss:', module.trn_log.get('struct_loss'))
print('[3] mass_loss:', module.trn_log.get('mass_loss'))
print('[3] mom_loss:', module.trn_log.get('mom_loss'))
assert torch.isfinite(loss), 'combined_loss 非有限!'
print('[4] PASS: 结构监督头 (depth+RC) 正常参与组合')
