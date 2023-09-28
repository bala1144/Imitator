"""
FLAMEModel Layer: Implementation of the 3D Statistical Face model in PyTorch
It is designed in a way to directly plug in as a decoder layer in a 
Deep learning framework for training and testing
It can also be used for 2D or 3D optimisation applications
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about FLAMEModel is available at http://flame.is.tue.mpg.de.
For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler

class dict_to_module(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAMEModel function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self, config):
        super(FLAME, self).__init__()
        self.num_vertices = 5023
        if type(config) is dict:
            config = dict_to_module(**config)

        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        # self.use_face_contour = config.use_face_contour
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAMEModel; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project
        default_shape = torch.zeros([self.batch_size, 300-config.shape_params],
                                            dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape_betas', nn.Parameter(default_shape,
                                                      requires_grad=False))

        default_pose = torch.zeros([self.batch_size, 6-config.pose_params],
                                            dtype=self.dtype, requires_grad=False)
        self.register_parameter('default_pose', nn.Parameter(default_pose,
                                                      requires_grad=False))


        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAMEModel; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project
        default_exp = torch.zeros([self.batch_size, 100 - config.expression_params],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('expression_betas', nn.Parameter(default_exp,
                                                            requires_grad=False))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros([self.batch_size, 6],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                            requires_grad=False))

        default_neck_pose = torch.zeros([self.batch_size, 3],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                            requires_grad=False))

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = config.use_3D_translation

        default_transl = torch.zeros([self.batch_size, 3],
                                     dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            'transl',
            nn.Parameter(default_transl, requires_grad=False))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))
        

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))


    def morph(self, expression_params,  neck_pose=None, jaw_pose=None, eye_pose=None, transl=None, shape_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        current_bs = expression_params.shape[0]
        shape_params = (shape_params if shape_params is not None else self.shape_betas[:current_bs])

        if expression_params.shape[-1] < 100:
            expression_params = torch.cat([expression_params, self.expression_betas[:current_bs]], dim=1)

        if shape_params.shape[-1] < 300:
            shape_params = torch.cat([shape_params, self.shape_betas[:current_bs]], dim=1)

        betas = torch.cat([shape_params, expression_params], dim=1)

        neck_pose = (neck_pose if neck_pose is not None else self.neck_pose[:current_bs])
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose[:current_bs])
        jaw_pose = (jaw_pose if jaw_pose is not None else self.default_pose[:current_bs,:3])

        full_pose = torch.cat([self.default_pose[:current_bs, :3], neck_pose, jaw_pose, eye_pose], dim=1)  # N x 15
        template_vertices = self.v_template.unsqueeze(0).repeat(current_bs, 1, 1)

        transl = (transl if transl is not None else self.transl[:current_bs])
        vertices, _ = lbs(betas, full_pose, template_vertices,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights)

        if self.use_3D_translation:
            vertices += transl.unsqueeze(dim=1)

        return vertices

    def morph_vis(self, current_bs=1, expression_params=None,  neck_pose=None, jaw_pose=None, eye_pose=None, transl=None, shape_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """

        shape_params = (shape_params if shape_params is not None else self.shape_betas[:current_bs])
        expression_params = (expression_params if expression_params is not None else self.expression_betas[:current_bs])


        if expression_params.shape[-1] < 100:
            expression_params = torch.cat([expression_params, self.expression_betas[:current_bs]], dim=1)

        if shape_params.shape[-1] < 300:
            shape_params = torch.cat([shape_params, self.shape_betas[:current_bs]], dim=1)

        betas = torch.cat([shape_params, expression_params], dim=1)

        neck_pose = (neck_pose if neck_pose is not None else self.neck_pose[:current_bs])
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose[:current_bs])
        jaw_pose = (jaw_pose if jaw_pose is not None else self.default_pose[:current_bs,:3])

        full_pose = torch.cat([self.default_pose[:current_bs, :3], neck_pose, jaw_pose, eye_pose], dim=1)  # N x 15
        template_vertices = self.v_template.unsqueeze(0).repeat(current_bs, 1, 1)

        transl = (transl if transl is not None else self.transl[:current_bs])
        vertices, _ = lbs(betas, full_pose, template_vertices,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights)

        if self.use_3D_translation:
            vertices += transl.unsqueeze(dim=1)

        return vertices

