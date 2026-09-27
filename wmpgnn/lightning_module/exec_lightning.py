from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import pytorch_lightning as pl

import torch

torch.set_float32_matmul_precision("high")
seed_everything(42, workers=True)


class ResetEarlyStoppingOnResume(pl.Callback):
    """续训时重置 EarlyStopping 状态。

    resume_ckpt 续训时, checkpoint 里保存了上一轮 EarlyStopping 状态
    (如 v38 结束时 wait_count=15 已满, stopped_epoch=105), 恢复后第一个
    epoch 的 val 未低于旧 best 即立即触发早停 -> 续训只跑 1 个 epoch 就结束
    (mass_head 作业 9997828 事故)。在 fit 开始 (ckpt 已加载) 后把 best 重置为
    无穷、wait_count 清零, 让早停从本 run 重新计。
    """

    def on_fit_start(self, trainer, pl_module):
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb.wait_count = 0
                cb.stopped_epoch = 0
                cb.best_score = torch.tensor(torch.inf)
                print("[early_stop] 续训: 重置 EarlyStopping (wait_count=0, best=inf, 从本 run 重新计)")


def training(module, configs, trn_loader=None, val_loader=None, chunkloader=None):
    # module = torch.compile(module)
    model = configs["model"]

    monitoring_loss = "val_combined_loss" if model == "DFEI" else "val_ft_loss"

    # 早停 patience 可配置: settings.early_stop_patience (默认 15 保持历史行为)。
    # 世代长训需跑满预算, 配置里设 early_stop_patience: 0 即禁用早停
    #  (或用 >= max_epochs 的值, 等效永不触发)。
    _es_patience = int(configs["settings"].get("early_stop_patience", 15) or 0)
    _es_callbacks = []
    if _es_patience > 0:
        _es_callbacks.append(EarlyStopping(
            monitor=monitoring_loss,
            verbose=True,
            mode="min",
            patience=_es_patience,
        ))

    best_model_callback = ModelCheckpoint(
        filename=f"best-{{epoch:02d}}-{{{monitoring_loss}:.3f}}",
        monitor=monitoring_loss,
        mode="min",
        save_top_k=15
    )
    last_epoch_callback = ModelCheckpoint(
        filename="epoch_{epoch:02d}",
        monitor=None,
        save_top_k=-1,
        every_n_epochs=1,
        save_last=False,
    )

    log_dir = configs["log_dir"]
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=model,
                                  version=configs["settings"].get("log_version", None))
    csv_logger = CSVLogger(save_dir=log_dir, name=model, version=tb_logger.version)

    configs = configs["settings"]
    precision = configs.get("precision", "16-mixed")  # 32 / 16-mixed / bf16-mixed
    resume_ckpt = configs.get("resume_ckpt", None)  # 从已有 checkpoint 续训 (绝对路径)
    _callbacks = _es_callbacks + [best_model_callback, last_epoch_callback]
    if resume_ckpt:  # 续训时重置早停状态, 否则 ckpt 里的 wait_count 会立即触发早停
        _callbacks.append(ResetEarlyStoppingOnResume())
    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        max_epochs=configs["epochs"],
        accelerator="auto",
        devices=configs["ngpu"],
        strategy="auto",
        callbacks=_callbacks,
        precision=precision,
        accumulate_grad_batches=configs["gacc"],
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        # benchmark=False: 2026-08-14 修复. benchmark=True 在异构 batch (每事件图大小不同) +
        # 部分 GPU 驱动下会在首个前向 hang (卡死且无报错, 与 9661910 卡死现象一致)
        benchmark=False,
    )

    """Start training"""
    if trn_loader is not None and val_loader is not None:
        trainer.fit(module, trn_loader, val_loader, ckpt_path=resume_ckpt)
    elif chunkloader is not None:
        trainer.fit(module, chunkloader, ckpt_path=resume_ckpt)
    else:
        raise NotImplemented
    _ = trainer.logger.version
    return trainer


def evaluate(trainer, module, tst_loader=None, chunkloader=None):
    if trainer is None:
        # 评估只输出结果, 不写日志/checkpoint; 禁用默认 logger 避免目录冲突
        # (PL 2.6 默认会在 default_root_dir 下再建 TensorBoardLogger)
        trainer = Trainer(
            default_root_dir=f'lightning_logs/version_0',  # save the eval stuff in the dir of the model
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
    # [2026-09-26 FIX] 原来直接 trainer.test(module, ...) -> PL 不加载任何 ckpt, 用的是**末轮
    # in-memory 权重**; 而下游独立 eval(load_module.get_bis_model) 用的是**val 最小的 best ckpt**。
    # 两个口径在 v614 上实测差 2.5pp (16.32 vs 13.81), 会让"同一次 run 自报指标"与"下游评估"不可比。
    # 这里改成优先 test best ckpt; 若 trainer 上没有可用的 checkpoint 回调则显式回退并打印。
    _ds = tst_loader if tst_loader is not None else chunkloader
    try:
        trainer.test(module, dataloaders=_ds, ckpt_path="best")
        print("[evaluate] 已使用 best ckpt (val_combined_loss 最小) 做测试", flush=True)
    except Exception as _e:
        print(f"[evaluate] ckpt_path='best' 不可用 ({type(_e).__name__}: {_e}); "
              f"回退到末轮 in-memory 权重", flush=True)
        trainer.test(module, dataloaders=_ds)
