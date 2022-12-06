import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import torch.nn.functional as F
import os

class CORNIA(nn.Module):
    def __init__(self, method_path):
        super(CORNIA, self).__init__()
        #read params
        codebook = scio.loadmat(os.path.join(method_path,'CSIQ_codebook_BS7.mat'))
        whiten_param = scio.loadmat(os.path.join(method_path, 'CSIQ_whitening_param.mat'))
        soft_scale_param = scio.loadmat(os.path.join(method_path, 'soft_scale_param.mat'))
        soft_model = scio.loadmat(os.path.join(method_path, 'soft_model.mat'))
        self.D_tmp = torch.from_numpy(codebook['codebook0']).float()
        self.register_buffer('D', self.D_tmp)
        self.M_tmp = torch.from_numpy(whiten_param['M']).float()
        self.register_buffer('M', self.M_tmp)
        self.P_tmp = torch.from_numpy(whiten_param['P']).float()
        self.register_buffer('P', self.P_tmp)
        self.sv_tmp = torch.from_numpy(soft_model['SVs']).float()
        self.register_buffer('sv', self.sv_tmp)
        self.sv_coef_tmp = torch.from_numpy(soft_model['sv_coef']).float()
        self.register_buffer('sv_coef', self.sv_coef_tmp)
        self.rho_tmp = torch.from_numpy(soft_model['rho']).float()
        self.register_buffer('rho', self.rho_tmp)
        self.scale_tmp = torch.from_numpy(soft_scale_param['soft_scale_param']).float()
        self.register_buffer('scale', self.scale_tmp)
        self.kernel_size = 7
        self.num_patch = 4000
        self.im2col = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size))
        self.num_random = 10
    def forward(self, x, seed):
        #x = x * 255.

        torch.manual_seed(seed)

        if len(x.size()) != 4:
            x = x.unsqueeze(0)
        x = x * 255
        x = torch.transpose(x, 2, 3)
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        x = 0.2989 * r + 0.587 * g + 0.114 * b

        x = x.unsqueeze(1)

        patches = self.im2col(x)

        final_score = 0

        for i in range(self.num_random):
            J = torch.randperm(patches.size(2))
            if self.num_patch > J.size(0):
                self.num_patch = J.size()

            patches_ = patches[..., J[0:self.num_patch]]
            # patches = patches[..., 0:self.num_patch]
            patch_mean = torch.mean(patches_, dim=1)
            patch_var = torch.sqrt(torch.var(patches_, dim=1) + 10)

            patches_ = (patches_ - patch_mean.unsqueeze(1)) / patch_var.unsqueeze(1)
            patches_ = torch.bmm((torch.transpose(patches_, 1, 2) - self.M.unsqueeze(1)), self.P.unsqueeze(0))

            fv = self.soft_encoding_func(patches_)
            fv = fv * self.scale[:, 0].unsqueeze(0) + self.scale[:, 1].unsqueeze(0)

            kernel_features = self.linear_kernel(features=fv, sv=self.sv)
            score = kernel_features @ self.sv_coef - self.rho

            final_score = final_score + score[0]
        final_score = final_score / self.num_random
        return final_score


    def soft_encoding_func(self, fv):
        D = self.D / (torch.sqrt(torch.sum(torch.pow(self.D, 2), dim=0) + 1e-20)).unsqueeze(0)
        z = torch.bmm(fv, D.unsqueeze(0))
        z_ = -z
        z1 = F.relu(z, inplace=True)
        z2 = F.relu(z_, inplace=True)
        z1, _ = torch.max(z1, dim=1)
        z2, _ = torch.max(z2, dim=1)
        soft_fv = torch.cat((z1, z2), dim=1)
        return soft_fv[0]

    def linear_kernel(self, features, sv):
        dist = torch.mm(features, sv.t())
        return dist


