"""masshead2 (version_47, 训练中) 提前评估: 覆盖 ckpt 为 version_47 当前 best。
用法: python eval_masshead2.py --config config_files/eval_CERN_v38_masshead2.yaml
"""
from optparse import OptionParser
import sys, os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from wmpgnn.analysis.config_adjusting import *
from wmpgnn.analysis.load_module import *
from wmpgnn.data_loader.get_data_loader import load_tst_loader
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.lightning_module.exec_lightning import evaluate
from wmpgnn.performance.plot_results import metrics_eval
import glob

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("", "--config", type=str, default=None, dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    print("=" * 45)
    print("Starting masshead2 evaluation (version_47, 训练中提前评估)")

    with open(option.CONFIG, "r") as file:
        configs = yaml.safe_load(file)
    configs = adjust_config_evaluation(configs)

    # 覆盖 ckpt: version_47 当前 best (训练中, best 会随训练更新; 此处用当前最低 val_comb)
    ckpt_dir = f"{configs['log_dir']}/DFEI/version_47/checkpoints"
    bests = glob.glob(f"{ckpt_dir}/best-epoch=*.ckpt")
    def vc(p):
        s = os.path.basename(p).split("val_combined_loss=")[1].rstrip(".ckpt")
        return float(s)
    best_ckpt = min(bests, key=vc)
    configs["DFEI"]["cpt"] = best_ckpt
    print("使用 ckpt:", best_ckpt)
    print("=" * 45)

    configs, tst_loader, chunkloader = load_tst_loader(configs)
    pos_weights = transform_pos_weight(None, None, mode="eval")
    module = load_module(configs, pos_weights)

    evaluate(None, module, tst_loader=tst_loader, chunkloader=chunkloader)
    metric_path = f"{configs['log_dir']}/DFEI/version_38/metrics.csv"  # v38 基底, 仅画图用
    metrics_eval(metric_path, configs, 38)
    print("Done")
    print("=" * 45)
