import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from imitator.utils.init_from_config import instantiate_from_config
from FLAMEModel.flame_masks import get_flame_mask
from imitator.utils.losses import Custom_errors
from imitator.models.nn_model import imitator
class Imitator(pl.LightningModule):
    def __init__(self,
                 optim_params,
                 monitor,
                 nn_model_cfg,
                 loss_cfg
                 ):
        super(Imitator, self).__init__()

        # imitator_no_motion_enc_style_input_to_dec
        self.optim_params = optim_params
        self.nn_model:imitator = instantiate_from_config(nn_model_cfg)

        self.loss = nn.MSELoss()
        self.monitor = monitor

        # setup teacher forcing train_teacher_forcing
        self.teacher_forcing = nn_model_cfg["params"].get("train_teacher_forcing", False)

        # create the custom losses
        self.vertice_dim = self.nn_model.args.vertice_dim
        # masks for the lips
        mask = get_flame_mask()
        self.lips_idxs = mask.lips
        lip_mask = torch.zeros((1, self.nn_model.args.vertice_dim//3, 3))
        lip_mask[0, self.lips_idxs] = 1.0
        self.lip_mask = lip_mask.view(1, -1)
        self.custom_loss = Custom_errors(self.vertice_dim, loss_creterion=self.loss, loss_dict=loss_cfg)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        ignore_keys.extend(["nn_model.PPE.pe"])
        print(keys)

        ignore_keys = set(ignore_keys)
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # reset the ignore keys
        del ignore_keys
        unmatched_keys = self.load_state_dict(sd, strict=False)
        print(f"\nRestored from {path}\n")
        print("unmatched keys", unmatched_keys)

    def configure_optimizers(self):
        # sum all the params
        params = list(self.nn_model.parameters())

         # optim params
        weight_decay = self.optim_params.get("weight_decay", 0.0)
        learning_rate = self.optim_params.get("lr")

        print("running with weight decay", self.optim_params)
        print()

        optimizer = torch.optim.Adam(params,
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        lr_sch_factor = self.optim_params.get("lr_sch_factor", 0.85)
        lr_sch_patience = self.optim_params.get("lr_sch_patience", 500)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=lr_sch_factor, patience=lr_sch_patience,
                                                                    min_lr=1e-9),
            'monitor': "train/rec_loss"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_input(self, batch, batch_idx):

        if self.lip_mask.device != self.device:
            self.lip_mask = self.lip_mask.to(self.device)

        audio, vertice, template, one_hot_all, file_name = batch
        return audio, vertice, template, one_hot_all, file_name

    def training_step(self, batch, batch_idx):

        audio, vertice, template, one_hot, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]

        rec_loss, pred_verts = self.nn_model(audio, template, vertice, one_hot, self.loss, teacher_forcing=self.teacher_forcing)
        mbp_loss = self.custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
        net_loss = rec_loss + mbp_loss

        self.log("train/net_loss", net_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("train/mbp_loss", mbp_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("train/rec_loss", rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        ### velocity loss
        velocity_loss, weighted_velocity_loss = self.custom_loss.velocity_loss(pred_verts, vertice)
        net_loss = net_loss + weighted_velocity_loss
        self.log("train/velocity_loss", velocity_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        out_dict = {
            "net_loss": net_loss.item(),
            "rec_loss": rec_loss.item(),
            "mbp_loss": mbp_loss.item(),
        }

        predicted_keypoints = pred_verts.clone().detach().cpu().numpy()
        gt_kp = vertice.clone().detach().cpu().numpy()

        return {'loss': net_loss,
                'results': out_dict,
                'predicted_kp': predicted_keypoints,
                'gt_kp': gt_kp
                }

    def val_shared_step(self, batch, batch_idx):

        audio, vertice, template, one_hot_all, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]
        train_subject = "_".join(file_name[0].split("_")[:-1])

        if train_subject in self.nn_model.train_subjects:
            condition_subject = train_subject
            iter = self.nn_model.train_subjects.index(condition_subject)
            one_hot = one_hot_all[:, iter, :] if len(one_hot_all.size()) == 3 else one_hot_all
            loss, pred_verts = self.nn_model(audio, template, vertice, one_hot, self.loss)
            mbp_loss = self.custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
            velocity_loss, weighted_velocity_loss = self.custom_loss.velocity_loss(pred_verts, vertice)

        else:
            condition_loss = []
            condition_mbp_loss = []
            condition_velocity_loss = []

            for iter in range(one_hot_all.shape[-1]):
                one_hot = one_hot_all[:, iter, :]
                loss, pred_verts = self.nn_model(audio, template, vertice, one_hot, self.loss)
                mbp_loss = self.custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
                condition_loss.append(loss.item())
                condition_mbp_loss.append(mbp_loss.item())
                velocity_loss, _ = self.custom_loss.velocity_loss(pred_verts, vertice)
                condition_velocity_loss.append(velocity_loss.item())

            loss = np.mean(condition_loss)
            mbp_loss = np.mean(condition_mbp_loss)
            velocity_loss = np.mean(condition_velocity_loss)

        # return loss, mbp_loss, pred_verts, mbp_weight_reg_loss, pred_based_mbp_loss, velocity_loss, viseme_velocity_loss
        return {
            "rec_loss" : loss,
            "mbp_loss" : mbp_loss,
            "pred_verts" : pred_verts,
            "velocity_loss" : velocity_loss,
        }

    def validation_step(self, batch, batch_idx):
        result_dict = self.val_shared_step(batch, batch_idx)
        net_loss = result_dict["rec_loss"] + result_dict["mbp_loss"]
        net_loss += self.custom_loss.loss_dict.get("velocity_weight", 0.0) * result_dict["velocity_loss"]

        self.log("val/rec_loss", result_dict["rec_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/net_loss", net_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mbp_loss", result_dict["mbp_loss"] / self.custom_loss.mbp_reconstruction_loss.closed_frame_weight, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        audio, vertice, template, one_hot_all, file_name = self.get_input(batch, batch_idx)
        return {'loss': net_loss,
                'predicted_kp': result_dict["pred_verts"].cpu().numpy(),
                'gt_kp': vertice.cpu().numpy(),
                }

    def test_step(self, batch, batch_idx):

        audio, vertice, template, one_hot_all, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]
        train_subject = "_".join(file_name[0].split("_")[:-1])

        result_npy_dict = {}
        loss = []
        full_reconstruction_mm = []
        lip_reconstruction_mm = []
        lip_sync_metric = []

        if train_subject in self.nn_model.train_subjects:
            condition_subject = train_subject
            iter = self.nn_model.train_subjects.index(condition_subject)
            one_hot = one_hot_all[:, iter, :] if len(one_hot_all.size()) ==3 else one_hot_all
            prediction = self.nn_model.predict(audio, template, one_hot)
            pred_len = prediction.shape[1]
            vertice = vertice[:, :pred_len]
            loss.append(self.loss(prediction, vertice).item())

            # reconstruction in mm
            full_reconstruction_mm.append(self.custom_loss.error_in_mm(prediction, vertice).cpu().numpy())
            lip_reconstruction_mm.append(
                self.custom_loss.compute_masked_error_in_mm(prediction, vertice, self.lip_mask).cpu().numpy())
            lip_sync_metric.append(self.custom_loss.lip_sync_metric(prediction, vertice, self.lip_mask).item())

            prediction = prediction.squeeze()  # (seq_len, V*3)
            out_file = file_name[0].split(".")[0] + "_condition_" + condition_subject
            result_npy_dict[out_file] = prediction.detach().cpu().numpy()
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = self.nn_model.train_subjects[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = self.nn_model.predict(audio, template, one_hot)

                pred_len = prediction.shape[1]
                vertice = vertice[:, :pred_len]

                loss.append(self.loss(prediction, vertice).item())
                full_reconstruction_mm.append(self.custom_loss.error_in_mm(prediction, vertice).cpu().numpy())
                lip_reconstruction_mm.append(self.custom_loss.compute_masked_error_in_mm(prediction, vertice, self.lip_mask).cpu().numpy())
                lip_sync_metric.append(self.custom_loss.lip_sync_metric(prediction, vertice, self.lip_mask).item())

                prediction = prediction.squeeze()  # (seq_len, V*3)
                out_file = file_name[0].split(".")[0] + "_condition_" + condition_subject
                result_npy_dict[out_file] = prediction.detach().cpu().numpy()

        out_dict = {
                    "metric_rec_loss": np.mean(loss),
                    "full_reconstruction_mm": np.mean(full_reconstruction_mm),
                    "lip_reconstruction_mm": np.mean(lip_reconstruction_mm),
                    "lip_sync_metric": np.mean(lip_sync_metric),
                    }

        gt_kp = vertice.cpu().numpy().squeeze()
        return {'results': out_dict,
                'prediction_dict': result_npy_dict,
                "gt_kp": gt_kp,
                'seq_name': file_name,
                'seq_len': prediction.shape[0]}

