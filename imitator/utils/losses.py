import torch
from imitator.utils.ringnet_lip_static_embedding import RingNet_lip_embedding
from imitator.utils.mbp_loss import MBP_reconstruction_loss

class Custom_errors():
    """
    Input to the loss must be in mm
    """
    def __init__(self, vertice_dim, 
                        loss_creterion,
                        loss_dict={}):
        self.vertice_dim = vertice_dim
        self.loss = loss_creterion
        self.loss_dict = loss_dict

        # self lip metric
        self.lip_static_embedding = RingNet_lip_embedding()
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss_with_reduction = torch.nn.MSELoss()

        mbp_weight_reg_loss_type = self.loss_dict.get('mbp_weight_reg_loss_type', "l1")
        if mbp_weight_reg_loss_type == "l1":
            self.mbp_reg_loss_type = self.l1_loss
        else:
            self.mbp_reg_loss_type = self.mse_loss_with_reduction

        # mbp loss
        if self.loss_dict.get('mbp_reconstruction_loss', None) is not None:
            self.mbp_reconstruction_loss = MBP_reconstruction_loss(vertice_dim, loss_dict)
        else:
            self.mbp_reconstruction_loss = None

    @torch.no_grad()
    def error_in_mm(self, pred_verts, gt_verts):
        pred_verts_mm = pred_verts.view(-1, self.vertice_dim//3, 3) * 1000.0
        gt_verts_mm = gt_verts.view(-1, self.vertice_dim//3, 3) * 1000.0
        diff_in_mm = pred_verts_mm - gt_verts_mm
        dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        return torch.mean(dist_in_mm)

    @torch.no_grad()
    def compute_masked_error_in_mm(self, pred_verts, gt_verts, mask):
        return self.error_in_mm(pred_verts * mask, gt_verts * mask)

    @torch.no_grad()
    def compute_max_diff_in_mm(self, pred_verts, gt_verts):
        """
        pred_verts : B x Nf x verice_dim
        """
        pred_verts_mm = pred_verts.view(-1, self.vertice_dim//3, 3) * 1000.0
        gt_verts_mm = gt_verts.view(-1, self.vertice_dim//3, 3) * 1000.0
        diff_in_mm = pred_verts_mm - gt_verts_mm
        dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        return torch.max(dist_in_mm)

    @torch.no_grad()
    def compute_masked_max_diff_in_mm(self, pred_verts, gt_verts, mask):
        return self.compute_max_diff_in_mm(pred_verts * mask, gt_verts * mask)

    def velocity_loss(self, predict, real):
        """
        predict: B x Nf x vertice_dim
        """
        velocity_weight = self.loss_dict.get('velocity_weight', 0.0)
        forward_velocity_weight = self.loss_dict.get('forward_velocity_weight', 0.0)
        if velocity_weight > 0:
            velocity_pred = predict[:, 1:, :] - predict[:, :-1, :]
            velocity_real = real[:, 1:, :] - real[:, :-1, :]
            velocity_loss = self.loss(velocity_pred, velocity_real)
            return velocity_loss, velocity_weight * velocity_loss
        elif forward_velocity_weight > 0:
            velocity_pred = predict[:, 1:, :] - predict[:, :-1, :].detach()
            velocity_real = real[:, 1:, :] - real[:, :-1, :]
            velocity_loss = self.loss(velocity_pred, velocity_real)
            return velocity_loss, forward_velocity_weight * velocity_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    def VertsRegLoss(self, expression_offset):
        verts_regularizer_weight = self.loss_dict.get('verts_regularizer_weight', 0.0)
        if verts_regularizer_weight > 0.0:
            verts_reg_loss = verts_regularizer_weight * torch.mean(torch.sum(torch.abs(expression_offset), dim=2))
            return verts_reg_loss, verts_regularizer_weight * verts_reg_loss
        else:
            return torch.tensor(0, dtype=expression_offset.dtype, device=expression_offset.device), torch.tensor(0, dtype=expression_offset.dtype, device=expression_offset.device)

    def lip_reconstruction_loss(self, predict, real, mask):
        lip_reconstruction_loss_weight = self.loss_dict.get('lip_reconstruction_loss_weight', 0.0)
        if lip_reconstruction_loss_weight > 0.0:
            lip_reconstruction_loss = self.loss(predict * mask,
                                                real * mask)
            return lip_reconstruction_loss, lip_reconstruction_loss_weight * lip_reconstruction_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    # compute the mbp reconstruction metric
    def compute_mbp_reconstruction_loss(self, predict, real, file_name):
        if self.mbp_reconstruction_loss is not None:
            loss = self.mbp_reconstruction_loss.compute_loss(predict, real, file_name)
        else:
             loss = torch.tensor(0, dtype=predict.dtype, device=predict.device)
        return loss

    @torch.no_grad()
    def lip_max_l2(self, predict, real, mask):
        """
        This is the lip sync metric used in the faceformer paper
        """
        mask = mask.to(real.device)
        lip_pred = predict * mask
        lip_real = real * mask

        pred_verts_mm = lip_pred.view(-1, self.vertice_dim//3, 3) * 1000.0
        gt_verts_mm = lip_real.view(-1, self.vertice_dim//3, 3) * 1000.0

        diff_in_mm = pred_verts_mm - gt_verts_mm
        l2_dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        max_l2_error_lip_vert, idx = torch.max(l2_dist_in_mm, dim=-1)
        mean_max_l2_error_lip_vert = torch.mean(max_l2_error_lip_vert)
        return mean_max_l2_error_lip_vert

    def mbp_weight_reg_loss(self, pred_mbp_weight, filename):
        mbp_weight_reg_loss_weight = self.loss_dict.get('mbp_weight_reg_loss', 0.0)
        if mbp_weight_reg_loss_weight > 0.0:
            gt_per_frame_weights = self.mbp_reconstruction_loss.normalized_weight_dict[filename][:pred_mbp_weight.shape[1]].to(pred_mbp_weight.device)
            gt_per_frame_weights = gt_per_frame_weights.view(1, -1, 1)
            mbp_weight_reg_loss = self.mbp_reg_loss_type(gt_per_frame_weights, pred_mbp_weight)
            return mbp_weight_reg_loss, mbp_weight_reg_loss_weight * mbp_weight_reg_loss
        else:
            return torch.tensor(0, dtype=pred_mbp_weight.dtype, device=pred_mbp_weight.device), torch.tensor(0, dtype=pred_mbp_weight.dtype, device=pred_mbp_weight.device)




