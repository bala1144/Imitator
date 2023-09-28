import numpy as np
import torch
from imitator.pl_trainer import Imitator
from FLAMEModel.flame_masks import get_flame_mask
from imitator.utils.losses import Custom_errors
class Full_training(Imitator):
    def __init__(self,
                 optim_params,
                 monitor,
                 nn_model_cfg,
                 init_from_ckpt=None,
                 loss_cfg={}
                 ):
        super(Full_training, self).__init__(optim_params, monitor, nn_model_cfg, loss_cfg)

        if init_from_ckpt is not None:
            self.init_from_ckpt(init_from_ckpt)

        self.vertice_dim = self.nn_model.args.vertice_dim
        # masks for the lips
        mask = get_flame_mask()
        self.lips_idxs = mask.lips
        lip_mask = torch.zeros((1, self.nn_model.args.vertice_dim // 3, 3))
        lip_mask[0, self.lips_idxs] = 1.0
        self.lip_mask = lip_mask.view(1, -1)
        self.loss_cfg = loss_cfg
        self.custom_loss = Custom_errors(self.vertice_dim, loss_creterion=self.loss, loss_dict=loss_cfg)

    def get_input(self, batch, batch_idx):
        if self.lip_mask.device != self.device:
            self.lip_mask = self.lip_mask.to(self.device)

        audio, vertice, template, one_hot_all, file_name = batch
        return audio, vertice, template, one_hot_all, file_name

    def training_step(self, batch, batch_idx):

        audio, vertice, template, one_hot, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]
        rec_loss, pred_verts = self.nn_model.style_forward(audio, file_name, template, vertice, one_hot, self.loss,
                                                              teacher_forcing=self.teacher_forcing)
        rec_loss = self.loss_cfg.get('full_rec_loss', 1.0) * rec_loss

        # compute the mbp loss
        mbp_loss = self.custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
        aux_loss = self.compute_auxillary_losses(pred_verts, vertice)
        net_loss = rec_loss + aux_loss["aux_losses"] + mbp_loss

        # here the recon loss is plotted train loss to compare with the previous results
        self.log("train/rec_loss", rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("train/mbp_loss", mbp_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("train/net_loss", net_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # plot auxillary loss
        self.plot_auxillary_loss(aux_loss, mode="train")

        out_dict = {
            "net_loss": net_loss.item(),
            "mbp_loss": mbp_loss.item(),
            "rec_loss": rec_loss.item(),
        }

        predicted_keypoints = pred_verts.clone().detach().cpu().numpy()
        gt_kp = vertice.clone().detach().cpu().numpy()
        return {'loss': net_loss,
                'results': out_dict,
                'predicted_kp': predicted_keypoints,
                'seq_name': file_name,
                'seq_len': predicted_keypoints.shape[1],
                'gt_kp': gt_kp
                }

    def validation_step(self, batch, batch_idx):
        audio, vertice, template, one_hot, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]
        train_subject = "_".join(file_name[0].split("_")[:-1])

        if train_subject in self.nn_model.train_subjects:
            rec_loss, pred_verts = self.nn_model.style_forward(audio, file_name, template, vertice, one_hot,
                                                                  self.loss, teacher_forcing=self.teacher_forcing)
            rec_loss = self.loss_cfg.get('full_rec_loss', 1.0) * rec_loss
            mbp_loss = self.custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
            aux_loss = self.compute_auxillary_losses(pred_verts, vertice)
            net_loss = rec_loss + aux_loss["aux_losses"] + mbp_loss
        else:
            raise ("error subject not seen during training")

        self.log("val/rec_loss", rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mbp_loss", mbp_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/net_loss", net_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.plot_auxillary_loss(aux_loss, mode="val")

        audio, vertice, template, one_hot_all, file_name = self.get_input(batch, batch_idx)
        out_dict = {
            "net_loss": net_loss.item(),
            "rec_loss": rec_loss.item(),
        }

        return {'loss': net_loss,
                'results': out_dict,
                'predicted_kp': pred_verts.cpu().numpy(),
                'seq_name': file_name,
                'seq_len': pred_verts.shape[1],
                'gt_kp': vertice.cpu().numpy(),
                }

    def test_step(self, batch, batch_idx):

        # compute the loss
        audio, vertice, template, one_hot, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]
        train_subject = "_".join(file_name[0].split("_")[:-1])

        result_npy_dict = {}
        loss = []
        rec_error_lip = []

        if train_subject in self.nn_model.train_subjects:

            condition_subject = train_subject
            prediction = self.nn_model.predict(audio, template, one_hot)
            pred_len = prediction.shape[1]
            vertice = vertice[:, :pred_len]

            # compute the custom losses
            loss.append(self.custom_loss.error_in_mm(prediction, vertice).item())
            rec_error_lip.append(self.custom_loss.compute_masked_error_in_mm(prediction, vertice, self.lip_mask).item())
            prediction = prediction.squeeze()  # (seq_len, V*3)

            # outfile
            out_file = file_name[0].split(".")[0] + "_condition_" + condition_subject
            result_npy_dict[out_file] = prediction.detach().cpu().numpy()

        else:
            raise ("error subject not seen during training")

        out_dict = {
            "metric_rec_loss": np.mean(loss),
            "metric_rec_lip_loss": np.mean(rec_error_lip),
        }
        gt_kp = vertice.cpu().numpy().squeeze()
        return {'results': out_dict,
                'prediction_dict': result_npy_dict,
                "gt_kp": gt_kp,
                'seq_name': file_name,
                'seq_len': prediction.shape[0],
                'embeddings': self.get_emdbedding()}

    def compute_auxillary_losses(self, predicted_mesh, gt_mesh):

        velocity_loss, velocity_loss_weighted = self.custom_loss.velocity_loss(predicted_mesh, gt_mesh)  # predict, real
        aux_loss_sum = velocity_loss_weighted
        loss_dict = {
            "aux_losses": aux_loss_sum,
            "velocity_loss": velocity_loss,
        }
        return loss_dict

    def plot_auxillary_loss(self, loss_dict, mode="train"):

        # here the recon loss is plotted train loss to compare with the previous results
        self.log("aux_loss_%s/velocity" % mode, loss_dict["velocity_loss"], prog_bar=False, logger=True, on_step=False,
                 on_epoch=True)

        return None

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

        # lr schedular
        lr_sch_factor = self.optim_params.get("lr_sch_factor", 0.85)
        lr_sch_patience = self.optim_params.get("lr_sch_patience", 500)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=lr_sch_factor, patience=lr_sch_patience,
                                                                    min_lr=1e-9),
            'monitor': "train/net_loss"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

#################################################################################################################################################################################
# ********************************** Inherited models are below **********************************
#################################################################################################################################################################################
class Style_only_optim(Full_training):
    def __init__(self,
                 optim_params,
                 monitor,
                 nn_model_cfg,
                 init_from_ckpt=None,
                 transformer_feature_file=None,
                 loss_cfg={}
                 ):
        super(Style_only_optim, self).__init__(optim_params, monitor, nn_model_cfg, None, loss_cfg)

        if init_from_ckpt is not None:
            self.init_from_ckpt(init_from_ckpt)
        else:
            print("No model for init; in general you need model for learning style")

        if transformer_feature_file is not None:
            self.load_transformer_features(transformer_feature_file)

        self.freeze()

    def freeze(self) -> None:
        print("\nFreezing from the imitator_org_vert_reg_style_optim\n")
        # freeze the model expect the style emebeddding
        for param in self.nn_model.parameters():
            param.requires_grad = False

        # unfrezzing the style encoder
        for param in self.nn_model.obj_vector.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(trainable_params)

class Style_and_disp_final_layer_optim_voca(Full_training):
    def __init__(self,
                 optim_params,
                 monitor,
                 nn_model_cfg,
                 init_from_ckpt=None,
                 transformer_feature_file=None,
                 train_motion_dec_from_scratch=False,
                 loss_cfg={}
                 ):
        super(Style_and_disp_final_layer_optim_voca, self).__init__(optim_params, monitor, nn_model_cfg,
                                                                    None,
                                                                             loss_cfg)

        if init_from_ckpt is not None:
            if train_motion_dec_from_scratch:
                self.init_from_ckpt(init_from_ckpt, ignore_keys=["vertice_map_r"])
            else:
                self.init_from_ckpt(init_from_ckpt)
        else:
            print("No model for init; in general you need model for learning style")
        self.freeze()

        if transformer_feature_file is not None:
            self.load_transformer_features(transformer_feature_file)

    def freeze(self) -> None:
        # freeze the model expect the style emebeddding
        for param in self.nn_model.parameters():
            param.requires_grad = False

        # unfreezing the style encoder
        for param in self.nn_model.obj_vector.parameters():
            param.requires_grad = True

        # unfreeze the motion decoder
        # v
        for name, param in self.nn_model.vertice_map_r.named_parameters():
            if "final_out_layer" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("number of trainiable params after freezing", trainable_params)
