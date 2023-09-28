import numpy as np
import torch
import torch.nn as nn
import pickle
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler
import os

class RingNet_lip_embedding(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self):
        super(RingNet_lip_embedding, self).__init__()
        flame_model_path = os.path.join("FLAMEModel",
                                              "model/generic_model.pkl")
        with open(flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))

        self.dtype = torch.float32
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        static_embedding_path = os.path.join("FLAMEModel",
                                             "model/flame_static_embedding.pkl")
        with open(static_embedding_path, 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))
        lmk_faces_idx = (static_embeddings.lmk_face_idx).astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))

        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=self.dtype))

        self.inner_upper_lip = [44, 45, 46]
        self.inner_lower_lip = [50, 49, 48]

        self.outer_upper_lip = [32, 33, 34, 35, 36]
        self.outer_lower_lip = [42, 41, 40, 39, 38]

    def extract_lip_keypoints(self, vertices):
        """
            Input:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = vertices.shape[0]
        self.device = vertices.device

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(
            batch_size, 1).to(self.device)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1).to(self.device)

        landmarks = vertices2landmarks(vertices, self.faces_tensor.to(self.device),
                                             lmk_faces_idx,
                                             lmk_bary_coords)

        return landmarks

    def extract_upper_and_lower_lip(self, vertices):
        """
            Input:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        landmarks = self.extract_lip_keypoints(vertices)
        return landmarks[:, self.inner_upper_lip+self.outer_upper_lip], landmarks[:, self.inner_lower_lip+self.outer_lower_lip]


    def get_lip_distance(self, vertices, vertice_dim=15069):
        face_keypoints = self.extract_lip_keypoints(vertices.view(-1, vertice_dim // 3, 3) * 1000.0)
        upper_lip_midpoint = face_keypoints[:, 45]  # this is fixed for the ring embedding
        lower_lip_midpoint = face_keypoints[:, 49]  # this is fixed for the ring embedding
        diff = upper_lip_midpoint - lower_lip_midpoint
        gt_distance_in_mm = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        return gt_distance_in_mm