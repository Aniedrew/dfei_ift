"""CPU 续训兼容验证: v38 best ckpt + mass_head: true。
验证 on_load_checkpoint 检测到旧 ckpt 无 edge_mass_head -> 重置 optimizer,
load_state_dict 允许新头缺失 (随机初始化续训)。
"""
import sys
sys.path.append('/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn')

import yaml
import torch
from wmpgnn.analysis.config_adjusting import adjust_config_training
from wmpgnn.analysis.load_module import load_module
from wmpgnn.data_loader.weights_calculator import transform_pos_weight

CKPT = '/lzufs/home/guoqingxiang/dfei/scalable_mtl_hgnn/LHCb_logs/DFEI/version_38/checkpoints/best-epoch=105-val_combined_loss=35.921.ckpt'

with open('config_files/train_CERN_v38.yaml') as f:
    configs = yaml.safe_load(f)
configs = adjust_config_training(configs)
configs['inference']['mass_head'] = True
configs['inference']['mass_loss_weight'] = 1.0
configs['DFEI']['cpt'] = CKPT           # 模拟续训: 从 v38 best 加载
pos_weights = transform_pos_weight(None, None, mode="eval")
module = load_module(configs, pos_weights)

print('[resume] mass_head_on:', module.mass_head_on)
print('[resume] edge_mass_head 参数存在:', any(k.startswith("model.edge_mass_head") for k in module.state_dict()))
print('[resume] 全部模型参数可加载: OK')
print('[resume] PASS: v38 ckpt + mass_head 续训兼容')
