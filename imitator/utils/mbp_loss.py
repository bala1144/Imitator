import torch
import numpy as np
import os
from glob import glob

class MBP_reconstruction_loss():
    """
    Input to the loss must be in mm
    """
    def __init__(self, vertice_dim,
                        loss_dict={}):
        self.vertice_dim = vertice_dim
        self.loss_dict = loss_dict

        mbp_config = self.loss_dict.get('mbp_reconstruction_loss')
        weight_path = mbp_config.get("frame_weight_path")
        # weight_path = os.path.join(os.getenv("HOME"), weight_path)
        print("Loading the  mbp weight from ", weight_path)

        ## load the weights
        combined_weight_file = os.path.join(weight_path, "combine_weights.npy")
        if os.path.exists(combined_weight_file):
            weight_dict = np.load(combined_weight_file, allow_pickle=True)
        else:
            all_files = glob(os.path.join(weight_path, "*.npy"))
            weight_dict = {}
            for file in all_files:
                if "all_seq" in file:
                    continue
                weight = np.load(file, allow_pickle=True)
                subj_sen = file.split("/")[-1].split(".npy")[0]
                weight_dict[subj_sen] = weight

            if len(weight_dict) == 0:
                raise("unable to load the weights, check whether the file exists or not")

        # process the weights and move it the device, we want to
        self.closed_frame_weight = mbp_config.get("closed_frame_weight", 1)
        normalized_weight_dict = {}
        if self.closed_frame_weight > 0:
            for k, d in weight_dict.items():
                weight_dict[k] = torch.from_numpy(d) * self.closed_frame_weight
                normalized_weight_dict[k] = torch.from_numpy(d)
        else:
            for k, d in weight_dict.items():
                weight_dict[k] = torch.from_numpy(d)
                normalized_weight_dict[k] = torch.from_numpy(d)

        # weight dict
        self.weight_dict = weight_dict
        self.normalized_weight_dict = normalized_weight_dict
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def compute_loss(self, predict, real, subjsen):
        
        if self.closed_frame_weight > 0:
            per_frame_weights = self.weight_dict[subjsen][:predict.shape[1]].to(real.device)
            biased_full_reconstruction_loss = self.mse_loss(predict, real)
            per_frame_weights = per_frame_weights.view(1, -1, 1)
            biased_full_reconstruction_loss = torch.mean(per_frame_weights * biased_full_reconstruction_loss)
            return biased_full_reconstruction_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device)