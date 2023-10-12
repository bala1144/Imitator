import os, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
import json
from pytorch_lightning import seed_everything
import collections, functools, operator
import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True
from timeit import default_timer as timer
from imitator.utils.init_from_config import instantiate_from_config
from argparse import ArgumentParser

def get_latest_checkpoint(ckpt_dir, pre_fix="epoch") -> str:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None
    :param ckpt_dir: directory of checkpoint
    :param pre_fixe: type of checkpoint, either "_every" or "_best"
    :return: latest checkpoint file
    """
    # Find all the every validation checkpoints
    print(ckpt_dir, pre_fix)
    print("{}/{}*.ckpt".format(ckpt_dir,pre_fix))
    list_of_files = glob.glob("{}/{}*.ckpt".format(ckpt_dir,pre_fix))
    # print(list_of_files)
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        latest_checkpoint = latest_checkpoint.replace('\\','/')
    print('Best checkpoint', latest_checkpoint)
    return latest_checkpoint

def process_result_dict(results_dict):
    # combine the results dicts
    loss_dicts = [batch['results'] for batch in results_dict]
    # add
    combined = dict(functools.reduce(operator.add, map(collections.Counter, loss_dicts)))
    # average the loss
    average_loss = {key: combined[key] / len(loss_dicts) for key in combined.keys()}
    return average_loss

class test_dataset_wise():
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        ### create the one-hot labels
        one_hot_labels = np.eye(8)
        self.one_hot_labels = torch.from_numpy(one_hot_labels).view(1, 8, 8).float()

        ### create the losses
        from imitator.utils.losses import Custom_errors
        from FLAMEModel.flame_masks import get_flame_mask
        loss_cfg = {}
        loss = nn.MSELoss()
        mask = get_flame_mask()
        self.lips_idxs = mask.lips
        lip_mask = torch.zeros((1, 5023, 3))
        lip_mask[0, self.lips_idxs] = 1.0
        self.lip_mask = lip_mask.view(1, -1)
        self.custom_loss = Custom_errors(15069, loss_creterion=loss, loss_dict=loss_cfg)

        ### create a render
        from imitator.utils.render_helper import render_helper
        self.rh = render_helper()

    def run_loop_with_condition_test(self, dataloader, model, condition_id=2):

        if len(model.nn_model.train_subjects) > 1:
            condition_subject = model.nn_model.train_subjects[condition_id]
        else:
            ### personalized style model
            condition_subject = model.nn_model.train_subjects[0]
        print("Current condition subject", condition_subject)

        results_dict_list = []

        full_reconstruction_mm = []
        lip_reconstruction_mm = []
        lip_max_l2 = []

        # set the modelt to eval for computation
        print("Set the model for the evaluation")
        model = model.eval()

        print("Total sequence to run in the dataloader", len(dataloader))
        for batch in dataloader:
            audio, vertice, template, one_hot_all, file_name = batch
            print("file_name", file_name)
            one_hot = self.one_hot_labels[:, condition_id, :]
            prediction = model.nn_model.predict(audio, template, one_hot)

            pred_len = prediction.shape[1]
            vertice = vertice[:, :pred_len]

            # reconstruction in mm
            full_reconstruction_mm.append(self.custom_loss.error_in_mm(prediction, vertice).cpu().numpy())
            lip_reconstruction_mm.append(
                self.custom_loss.compute_masked_error_in_mm(prediction, vertice, self.lip_mask).cpu().numpy())
            # lip metrics
            lip_max_l2.append(self.custom_loss.lip_max_l2(prediction, vertice, self.lip_mask).item())

            # simple metric rec loss
            out_dict = {
                "full_reconstruction_mm": np.mean(full_reconstruction_mm),
                "lip_reconstruction_mm": np.mean(lip_reconstruction_mm),
                "lip_max_l2": np.mean(lip_max_l2),
                        }

            results_dict_list.append({'results': out_dict,
                                      'predict': prediction.detach().cpu(),
                                      'seq':file_name}
                                     )

        return results_dict_list

    def run_test(self, model, data, logdir, dataset_to_eval, condition=2):
        out_dir = os.path.join(logdir, "voca_eval_with_fixed_test_cond")
        os.makedirs(out_dir, exist_ok=True)

        best_ckpt = get_latest_checkpoint(os.path.join(logdir, "checkpoints"))
        if best_ckpt is None:
            raise("Pre-Trained model is not available")

        if dataset_to_eval == "test":
            test_data = data._test_dataloader()
        elif dataset_to_eval == "val":
            test_data = data._val_dataloader()

        print("Current best checkpoint", best_ckpt)
        model.init_from_ckpt(path=best_ckpt)
        results_dict = self.run_loop_with_condition_test(test_data, model, condition)

        ### process and dump the metrics
        metrics = process_result_dict(results_dict)
        with open(os.path.join(out_dir, 'results.json'), 'w') as file:
            file.write(json.dumps(metrics, indent=4))

        for seq in results_dict:
            pred = seq["predict"]
            file_name = seq["seq"]
            seq_name = file_name[0].replace(".wav", "")
            if self.args.render_results:
                vid_dir = os.path.join(out_dir, "vid")
                os.makedirs(vid_dir, exist_ok=True)
                if os.getenv("VOCASET_PATH"):
                    audio_file = os.path.join(os.getenv("VOCASET_PATH"),
                                          data.data_cfg["wav_path"],
                                          file_name[0])
                else:
                    audio_file = os.path.join(os.getenv("HOME"),
                                            data.data_cfg["dataset_root"],
                                            data.data_cfg["wav_path"],
                                            file_name[0])
                self.rh.visualize_meshes(vid_dir, seq_name, pred.reshape(-1, 5023,3), audio_file)

            if self.args.dump_results:
                dump_dir = os.path.join(out_dir, "dump")
                os.makedirs(dump_dir, exist_ok=True)
                out_file = os.path.join(dump_dir, seq_name+".npy")
                np.save(out_file, pred.reshape(-1, 5023, 3).numpy())
                print("Dumping file", out_file)

def get_parser(parser):
    parser.add_argument('-g', "--gpus", type=str, default=0)
    parser.add_argument('-m', "--model", type=str, default="pretrained_model/generalized_model_mbp_vel")
    parser.add_argument('-o', "--out_dir", type=str, default="pretrained_model/generalized_model_mbp_vel")
    parser.add_argument('-c', "--cfg", type=str, default="imitator/test/data_cfg.yaml")
    parser.add_argument('-e', '--data_to_eval', type=str, default="test",  required=False)
    parser.add_argument('-d', '--dump_results', action='store_true')
    parser.add_argument('-r', '--render_results', action='store_true')

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.set_defaults(unseen=False)
    return parser


if __name__ == "__main__":

    start = timer()

    parser = ArgumentParser()
    parser = get_parser(parser)
    opt = parser.parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(opt.seed)

    if opt.model is None:
        raise ("Input a valid model to test")

    # setup dataset the eval
    print("Datasets to eval", opt.data_to_eval)
    print("Data configfile", opt.cfg)

    if "0024" in opt.model or "0138" in opt.model:
        data_cfg = OmegaConf.load(opt.cfg).data_style_cfg
        if "0138" in opt.model:
            data_cfg.params.train_subjects = "FaceTalk_170809_00138_TA"
            data_cfg.params.val_subjects = "FaceTalk_170809_00138_TA"
            data_cfg.params.test_subjects = "FaceTalk_170809_00138_TA"
    else:
        data_cfg = OmegaConf.load(opt.cfg).data

    print("testing config before", data_cfg)
    data = instantiate_from_config(data_cfg)
    data.setup()

    if not os.path.exists(opt.model):
        raise ValueError("Cannot find {}".format(opt.model))

    # load config
    logdir = opt.model
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)
    model = instantiate_from_config(config.model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print("Printed number of trainable params", pytorch_total_params)
    print()
    model.summarize()
    print()

    tester = test_dataset_wise(args=opt)
    print()

    tester.run_test(model, data, logdir,
                              opt.data_to_eval, condition=int(data_cfg.conditiion_id))

    end = timer()
    print("\n\nTime to take to run the tesing suite in sec", end - start)