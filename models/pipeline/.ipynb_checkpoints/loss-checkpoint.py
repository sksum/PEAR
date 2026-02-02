import torch
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import matrix_to_rotation_6d, matrix_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, x, y):
        residual = x - y
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return (self.rho ** 2 * dist).mean()

class Landmark2DLoss(nn.Module):
    def __init__(self, left_indices=None, right_indices=None, front_indices=None, 
                 selected_mp_indices=None, metric='l1', **kwargs) -> None:
        super().__init__()
        if metric == 'robust':
            self.metric = GMoF(rho=kwargs.get('rho', 1.0))
        elif metric == 'l2':
            self.metric = nn.MSELoss()
        else:
            self.metric = nn.L1Loss()

        self.left_indices  = torch.tensor(left_indices) if left_indices is not None else None
        self.right_indices = torch.tensor(right_indices) if right_indices is not None else None
        self.front_indices = torch.tensor(front_indices) if front_indices is not None else None
        self.selected_mp_indices = torch.tensor(selected_mp_indices) if selected_mp_indices is not None else None
    
    def forward(self, x:torch.Tensor, y:torch.Tensor, cam=None, weight=None):
        """calc face landmark loss

        Args:
            x (torch.Tensor): [B, N, x]
            y (torch.Tensor): [B, N, x]
            cam (torch.Tensor, optional): [B, 3]. Defaults to None.

        Returns:
            torch.Tensor: loss value
        """

        _x, _y = x, y

        if _x.shape[1] == 203:
            # return 0
            assert cam is not None
            assert self.left_indices is not None and self.right_indices is not None and self.front_indices is not None
            t_loss = 0
            t_x, t_y = _x[cam[:, 1] < -0.05], _y[cam[:, 1] < -0.05]
            if t_x.shape[0] > 0:
                t_loss += self.metric(t_x[:, self.left_indices], t_y[:, self.left_indices])
            t_x, t_y = x[cam[:, 1] > 0.05], y[cam[:, 1] > 0.05]
            if t_x.shape[0] > 0:
                t_loss += self.metric(t_x[:, self.right_indices], t_y[:, self.right_indices])
            mask = (cam[:, 1] >= -0.05) & (cam[:, 1] <= 0.05)
            t_x, t_y = x[mask], y[mask]
            if t_x.shape[0] > 0:
                t_loss += self.metric(t_x[:, self.front_indices], t_y[:, self.front_indices])
            return (t_loss / _x.shape[0]) * 15  # trust mouth region
        elif _x.shape[1] == 478:
            return self.metric(_x[:, self.selected_mp_indices], _y) / 5
        elif _x.shape[1] == 105 and _y.shape[1] == 478:
            return self.metric(_x, _y[:, self.selected_mp_indices]) / 5
        elif _x.shape[1] != _y.shape[1]:
            min_len = min(_x.shape[1], _y.shape[1])
            return self.metric(_x[:, :min_len], _y[:, :min_len]) / 5
        else:
            if weight is None:
                return self.metric(_x, _y).float()
            else:
                return self.metric(_x * weight, _y * weight).float()


class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)

        loss = ( self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()

class CameraLoss(nn.Module):

    
    def __init__(self):
        """
        camera parameter loss module.
        """
        super(CameraLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='sum')  # 要不试试 sum 

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):
        """
        pred_param: R T
        """
        # matrix_to_rotation_6d  matrix_to_axis_angle
        # pd_R_vec = matrix_to_axis_angle(pred_param[:,:3,:3]) # [B,3] 转换为轴角  
        pd_T = pred_param[:,:3,3]
        # pd_RT = torch.cat([pd_R_vec,pd_T] , dim = 1) # [B,9]   


        # gt_R_vec = matrix_to_axis_angle(gt_param[:,:3,:3])
        gt_T = gt_param[:,:3,3]
        # gt_RT = torch.cat([gt_R_vec, gt_T] , dim = 1)


        camera_loss = self.loss_fn(pd_T,gt_T)
        return camera_loss






class HeadParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(HeadParameterLoss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction='sum')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        # batch_size = pred_param.shape[0]
        # num_dims = len(pred_param.shape)
        # mask_dimension = [batch_size] + [1] * (num_dims-1)
        # has_param = has_param.type(pred_param.type()).view(*mask_dimension)


        B = pred_param['eye_pose_params'].shape[0]
        param1 = torch.cat([pred_param['eye_pose_params'], pred_param['pose_params'],  # pose_params 似乎没必要 loss
        pred_param['jaw_params'],pred_param['eyelid_params']] ,dim = 1) # [B,14]
    
        pd_exp = pred_param['expression_params']  # [B,50]
        pd_shape = pred_param['shape_params']  # [B,300]



        param2 = torch.cat([gt_param['eye_pose_params'], gt_param['pose_params'],
        gt_param['jaw_params'],gt_param['eyelid_params']] ,dim = 1)
    
        gt_exp = gt_param['expression_params']
        gt_shape = gt_param['shape_params']

        loss_all = self.loss_fn(param1, param2) * 0.01 + self.loss_fn(pd_exp, gt_exp)  * 0.01 + self.loss_fn(pd_shape, gt_shape) *  0.01
        return loss_all


class BodyParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(BodyParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='sum') # None 换成 sum   None -> Batch

    def smooth_bounded_exp_loss(self, param, threshold=0.5, scale_in=1.0, scale_out=5.0):
        abs_param = torch.abs(param)
        inside = scale_in * abs_param ** 2              # 在[-0.5, 0.5]内：小平方损失
        excess = torch.relu(abs_param - threshold)
        outside = torch.exp(scale_out * excess) - 1     # 超出后指数增长
        return torch.where(abs_param <= threshold, inside, outside)


    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        B = pred_param['body_pose'].shape[0]
        pd_pose = torch.cat([pred_param['body_pose'], pred_param['left_hand_pose'],pred_param['right_hand_pose']] ,dim = 1)
        
        pd_ori_pose = pred_param['global_pose']
        pd_exp = pred_param['exp']
        # pd_offset =  pred_param['joints_offset']
        pd_shape = pred_param['shape']
        pd_scale = torch.cat([pred_param['head_scale'][:,None], pred_param['hand_scale'][:,None]], dim =1 )

        gt_pose = torch.cat([gt_param['body_pose'],gt_param['left_hand_pose'],gt_param['right_hand_pose']] ,dim = 1)
        
        gt_ori_pose = gt_param['global_pose'][:,None]
        gt_exp = gt_param['exp']
        # gt_offset =  gt_param['joints_offset']
        gt_shape = gt_param['shape']
        gt_scale = torch.cat([ gt_param['head_scale'][:,None],gt_param['hand_scale'][:,None]] , dim =1)

        loss_ori_pose = self.loss_fn(pd_ori_pose, gt_ori_pose)  
        loss_scale =  self.loss_fn(pd_scale, gt_scale)  
        loss_pose =  self.loss_fn(pd_pose, gt_pose)  
        loss_exp =  self.loss_fn(pd_exp, gt_exp)  
        loss_shape = self.loss_fn(pd_shape, gt_shape)  
        # HSMR loss weight  ; {'kp3d': 0.05, 'kp2d': 0.01, 'prior': 0.0, 'poses_orient': 0.002, 'poses_body': 0.001, 'betas': 0.0005}
        # 
        bone_params = pd_pose[:,[2,5,8],1]
        loss_bone = self.smooth_bounded_exp_loss(bone_params).sum()

        loss_all = loss_ori_pose * 0.02 + loss_pose * 0.01 + loss_bone * 0.01 \
          + loss_exp * 0.005 +  loss_shape * 0.005 + loss_scale * 0.01  
        # 0.2889
        return loss_all




