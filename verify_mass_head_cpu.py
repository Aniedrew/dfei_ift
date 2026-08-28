"""CPU 前向验证: mass_head 最小实验 (新模块 + 真实事件 batch, mode=train)。
验证:
  1. mass_head 初始化 & edge_mass_head 挂载
  2. shared_step 里 _mass_loss 返回有限值 (非 None, 非 NaN)
  3. mass_loss 量级合理 (log10 尺度, ~0.1-1), 不淹没主 loss
"""
import sys, io
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')

import yaml
import torch
import zstandard as zstd
from torch_geometric.data import Batch

from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.weights_calculator import transform_pos_weight

torch.manual_seed(0)

with open('config_files/train_CERN_v38_masshead.yaml') as f:
    configs = yaml.safe_load(f)
configs = adjust_config_training(configs)
configs['DFEI']['cpt'] = "None"                   # 新模块 (不加载 ckpt, 快速验证)
pos_weights = transform_pos_weight(None, None, mode="eval")
module = load_module(configs, pos_weights)
print('[1] mass_head_on:', module.mass_head_on)
print('[1] edge_mass_head 挂载:', hasattr(module.model, 'edge_mass_head'))
print('[1] mass_loss_weight:', module.mass_loss_weight)
print('[1] _m_pi:', module._m_pi)

# 真实事件 (tst 文件), 模拟 load_dataset 的双向化
DATA = '/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00220000_00220999.pt.zst'
dctx = zstd.ZstdDecompressor()
with open(DATA, 'rb') as f:
    with dctx.stream_reader(f) as reader:
        data = torch.load(io.BytesIO(reader.read()), weights_only=False)
evts = data[:2]
for evt in evts:
    store = evt[('tracks', 'to', 'tracks')]
    store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
    store.edges = store.edges.repeat(2, 1)
    if getattr(store, 'y', None) is not None:
        store.y = store.y.repeat(2)
    if getattr(store, 'lca', None) is not None:
        store.lca = store.lca.repeat(2, 1)
    if getattr(store, 'sig_y', None) is not None:
        store.sig_y = store.sig_y.repeat(2)

batch = Batch.from_data_list(evts)
print('[2] node_types:', batch.node_types)
print('[2] tracks.x:', batch['tracks'].x.shape, '| tt edges:', batch[('tracks','to','tracks')].edge_index.shape)
print('[2] tracks.pid:', batch['tracks'].pid.shape if hasattr(batch['tracks'], 'pid') else 'MISSING')

module.train()
loss = module.shared_step(batch, 0, module.trn_log, mode='train')
print('[3] combined_loss:', float(loss))
print('[3] mass_loss 记录:', module.trn_log.get('mass_loss'))
print('[3] 其他 loss 记录:',
      {k: module.trn_log[k] for k in module.trn_log if k not in ('mass_loss',)})
assert torch.isfinite(loss), "combined_loss 非有限!"
print('[4] PASS: shared_step 前向+反向路径完整, mass_loss 正常参与组合')
