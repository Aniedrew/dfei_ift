"""CPU 验证: 不对称 latent 扩展 (tracks 32 / tt边 24) 的 forward 维度正确性。"""
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

with open('config_files/train_CERN_v38_masshead2_asym.yaml') as f:
    configs = yaml.safe_load(f)
configs = adjust_config_training(configs)
configs['DFEI']['cpt'] = "None"
pos_weights = transform_pos_weight(None, None, mode="eval")
module = load_module(configs, pos_weights)
print('[1] 模型构建 OK')

DATA = '/lzufs/user/guoqingxiang/DFEI_IFT_20260702/data/MC_normed/inclusive_00342442/tst_data_00220000_00220999.pt.zst'
dctx = zstd.ZstdDecompressor()
with open(DATA, 'rb') as f:
    with dctx.stream_reader(f) as reader:
        data = torch.load(io.BytesIO(reader.read()), weights_only=False)
for evt in data[:2]:
    store = evt[('tracks', 'to', 'tracks')]
    store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
    store.edges = store.edges.repeat(2, 1)
    if getattr(store, 'y', None) is not None:
        store.y = store.y.repeat(2)

batch = Batch.from_data_list(data[:2])
module.train()
with torch.no_grad():
    loss = module.shared_step(batch, 0, module.trn_log, mode='train')
    # 检查维度 (重新 forward 获取 outputs)
    batch2 = Batch.from_data_list(data[:2])
    p_norm = batch2['tracks'].x[:, :3].clone()
    outputs = module.forward(batch2)
    nx = outputs['tracks'].x
    ex = outputs[('tracks', 'to', 'tracks')].latent_edges
    print('[2] 节点表征 tracks.x 维度:', tuple(nx.shape), '(期望 [N, 32])')
    print('[2] 边表征 latent_edges 维度:', tuple(ex.shape), '(期望 [E, 24])')
print('[3] combined_loss:', float(loss))
print('[3] mass_loss:', module.trn_log.get('mass_loss'))
assert nx.shape[-1] == 32, f"节点维度错误: {nx.shape[-1]}"
assert ex.shape[-1] == 24, f"边维度错误: {ex.shape[-1]}"
assert torch.isfinite(loss), 'combined_loss 非有限!'
print('[4] PASS: 不对称扩展维度正确, 全路径正常')
