import glob
import re
import yaml

import torch
import torch.nn.modules.lazy as _lazy_mods

from typing import Dict

from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.model.model import DFEI_HGNN, FT_HGNN
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.ift_lightning_module import IFTLightningModule
from wmpgnn.lightning_module.ift_data_lightning_module import IFTLightningModuleData


def load_dfei_for_ift(configs):
    version = configs['IFT']['dfei_model']
    log_dir = configs['log_dir']
    if version != "None":
        with open(f"{log_dir}/DFEI/version_{version}/hparams.yaml", "r") as file:
            dfei_hparams = yaml.safe_load(file)
        if isinstance(version, int):
            dfei_bis_model = get_bis_model(version, dfei_hparams)
        elif isinstance(version, str):
            dfei_bis_model = version
        else:
            raise RuntimeError(f"Unsupported load_dfei: {type(version)}")
        pos_weights = transform_pos_weight(None, None, mode="eval")
        print("Using DFEI model:", dfei_bis_model)

        dfei_hparams["DFEI"]["cpt"] = dfei_bis_model
        configs["DFEI"] = dfei_hparams["DFEI"]
        module = load_module(dfei_hparams, pos_weights)
        model = module.model
    else:
        print("No DFEI model specified. Information from pv asso/truth is used as a replacement")
        model = None
    return configs, model


def get_bis_model(version: int, configs: Dict) -> str:
    # find the model with the best performance in the checkpoints
    model = configs["model"]
    log_dir = configs["log_dir"]
    files = glob.glob(f"{log_dir}/{model}/version_{version}/checkpoints/*best-epoch*.ckpt")
    if model == "DFEI":
        pattern = re.compile(r"val_combined_loss=([\d.]+)")
    elif model == "IFT":
        pattern = re.compile(r"val_ft_loss=([\d.]+)")
    else:
        raise ValueError(f"undefined model: {model}")
    bis = min(files, key=lambda s: float(pattern.search(s).group(1)[:-1]))
    return bis


def _upscale_conflicts(model, state_dict) -> list:
    """结构升级 (latent 升维 16→32/24 / GN 宽度 128→256) 继承冲突检测:
    返回需从 ckpt 删除(保持随机/重新初始化)的 key 列表。

    先判断"结构是否真的变了" (encoder/blocks/decoder/op_trafo 内存在 shape 可读的
    weight/bias mismatch)。结构未变 → 返回空 (全走原 PL 路径, lazy 用 ckpt 模板继承);
    结构变了 → shape 冲突层重学 + lazy 输入层按段重学 (防 ckpt 模板错 materialize):
       - encoder lazy: 输入 raw 恒不变 → 保留继承
       - blocks 段更新网络: 输入/输出依赖 latent 与自身宽度 → 一律重学
       - decoder/op_trafo: 单输入只随 latent 变 (tracks/tt 重学, 其余恒 16 保留)
    返回空列表 = 无冲突, 走原 PL load_from_checkpoint 路径 (零影响)。
    """
    STRUCT = ("_encoder.", "_blocks.", "_decoder.", "_op_trafo.")
    model_sd = model.state_dict()
    drop = []
    struct_changed = False
    # 阶段 1: 结构层 shape mismatch → 架构确实升级 (latent/宽度), 冲突层重学
    for k, v in state_dict.items():
        if k not in model_sd or not k.startswith(STRUCT):
            continue
        try:
            if tuple(v.shape) != tuple(model_sd[k].shape):
                drop.append(k)
                struct_changed = True
        except RuntimeError:                     # lazy 层 (shape 不可读): 阶段 2 处理
            pass
    if not struct_changed:
        return []                                # 结构未变 → 不 drop 任何, 走原 PL 路径
    # 阶段 2: 结构变了 → lazy 输入层按段重学
    for k, v in state_dict.items():
        if k not in model_sd or not k.startswith(STRUCT):
            continue
        modname = k[: k.rfind(".")]
        if modname.startswith("_encoder."):
            continue                             # encoder lazy 输入 raw 恒不变 → 继承
        try:
            _ = model_sd[k].shape                # 非 lazy → 阶段 1 已处理
            continue
        except RuntimeError:
            pass
        if modname.startswith("_blocks."):
            drop.append(k)                       # blocks 更新网络 → 重学
        else:
            lowered = (modname.replace("'", "").replace("(", "").replace(")", "")
                       .replace(", ", "_"))
            if "tracks_to_tracks" in lowered or "_node_models_model_dict.tracks." in lowered:
                drop.append(k)                   # decoder/op_trafo tracks/tt → 重学
    return drop


def load_module(configs: Dict, pos_weights:Dict, dfei_model=None, mode="simulation"):
    model = configs["model"]
    # Checking if need to load from cpt
    load_from_cpt = configs[model]["cpt"]
    if isinstance(configs[model]["cpt"], int):  # adjusting if passed an int
        bis_model = get_bis_model(load_from_cpt, configs)
    else:
        bis_model = configs[model]["cpt"]
    lr = float(configs["settings"]["lr"])
    weight_decay = float(configs["settings"]["weight_decay"])
    if mode == "simulation":
        if model == "DFEI":
            model = DFEI_HGNN(configs[model])
            if load_from_cpt == "None":
                module = DFEILightningModule(
                    model=model,
                    optimizer_class=torch.optim.Adam,
                    optimizer_params={"lr": lr, "weight_decay": weight_decay},
                    configs=configs,
                    pos_weights=pos_weights,
                )
            else:
                print("Loading from checkpoint")
                print(bis_model)
                print("=" * 30)
                # ==== 升维部分继承 (latent 16→32/24): shape/Lazy 冲突检测 ====
                # 冲突 → 手工加载 (冲突 key 随机化重学, 其余继承); 无冲突 → 原 PL 路径
                ckpt = torch.load(bis_model, map_location="cpu")
                sd_m = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
                drop = _upscale_conflicts(model, sd_m)
                if drop:
                    print(f"[upscale] 升维继承: 检测到 {len(drop)} 个 shape/Lazy 冲突 key, "
                          f"随机化重学 (其余 {len(ckpt['state_dict']) - len(drop)} 个权重继承)")
                    for k in sorted(drop):
                        print(f"    drop {k}")
                    module = DFEILightningModule(
                        model=model,
                        optimizer_class=torch.optim.Adam,
                        optimizer_params={"lr": lr, "weight_decay": weight_decay},
                        configs=configs,
                        pos_weights=pos_weights,
                    )
                    module.on_load_checkpoint(ckpt)
                    # drop 列表无 'model.' 前缀 (与 DFEI_HGNN key 空间一致), sd_f 用带前缀 key 过滤
                    sd_f = {k: v for k, v in ckpt["state_dict"].items()
                            if k.replace("model.", "", 1) not in set(drop)}
                    # 再剥离 wrapper 级头 shape 冲突 (source_head 等, 不在 DFEI_HGNN 检测范围),
                    # 然后 strict=False 加载: drop/剥离层保持随机, uninit Lazy 由 ckpt 模板或首 forward 初始化
                    module_sd = module.state_dict()
                    sd_f2 = dict(sd_f)
                    for k, v in sd_f.items():
                        if k not in module_sd:
                            continue
                        try:
                            if tuple(v.shape) != tuple(module_sd[k].shape):
                                del sd_f2[k]
                        except RuntimeError:      # cur 侧 uninit (Lazy): 保留, ckpt 模板 materialize
                            pass
                    module.load_state_dict(sd_f2, strict=False)
                else:
                    module = DFEILightningModule.load_from_checkpoint(
                        checkpoint_path=bis_model,
                        model=model,
                        pos_weights=pos_weights,
                        optimizer_class=torch.optim.Adam,
                        optimizer_params={"lr": lr, "weight_decay": weight_decay},
                        configs=configs,
                    )
        elif model == "IFT":
            model = FT_HGNN(configs["IFT"])
            if load_from_cpt == "None":
                module = IFTLightningModule(
                    model=model,
                    dfei_model=dfei_model,
                    optimizer_class=torch.optim.Adam,
                    optimizer_params={"lr": lr, "weight_decay": weight_decay},
                    configs=configs,
                    pos_weights=pos_weights,
                )
            else:
                #bis_model = "LHCb_logs/IFT/version_2/checkpoints/best-epoch=20-val_ft_loss=0.573.ckpt"
                print("Loading from checkpoint")
                print(bis_model)
                print("=" * 30)
                module = IFTLightningModule.load_from_checkpoint(
                    checkpoint_path=bis_model,
                    model=model,
                    dfei_model=dfei_model,
                    pos_weights=pos_weights,
                    optimizer_class=torch.optim.Adam,
                    optimizer_params={"lr": lr, "weight_decay": weight_decay},
                    configs=configs,
                )
        else:
            raise ValueError("Invalid model")
    elif mode == "data":
        print("Loading from checkpoint")
        print(bis_model)
        print("=" * 30)
        if model == "DFEI":
            raise NotImplementedError
            model = DFEI_HGNN(configs[model])
            module = DFEILightningModule.load_from_checkpoint(
                checkpoint_path=bis_model,
                model=model,
                pos_weights=pos_weights,
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": lr, "weight_decay": weight_decay},
                configs=configs,
            )
        elif model == "IFT":
            model = FT_HGNN(configs["IFT"])
            module = IFTLightningModuleData.load_from_checkpoint(
                checkpoint_path=bis_model,
                model=model,
                dfei_model=dfei_model,
                configs=configs,
            )
        else:
            raise ValueError("Invalid model")
    else:
        raise ValueError("Invalid mode")
    return module
