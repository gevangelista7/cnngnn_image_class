import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
# from ..train_util import *
from .model_utils_gnncnn import *
import torch.nn.functional as F

# define the function includes in import *
# __all__ = [
#     'SpixelNet1l', 'SpixelNet1l_bn'
# ]

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNGNNModel(nn.Module):
    expansion = 1

    def __init__(self, input_shape, batchNorm=True, device=default_device, downsize=16):
        super(CNNGNNModel, self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9
        self.device = device
        self.batch_size, img_h, img_w, _ = input_shape

        # reshaping transformations
        # prob neighbourhood to all-superpixels
        self.spix_idx_tensor = create_spix_idx(img_H=img_h, img_W=img_w, downsize=downsize,
                                               batch_size=self.batch_size).to(self.device)
        n_spixl, _, _ = calc_n_spix(img_h, img_w, downsize)

        # cache of probabilities (spix membership)
        self.prob_all_shape = (self.batch_size, n_spixl, img_h, img_w)

        # transform matrix to up-sample the rich feature map
        identity_matrix = torch.tensor([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]], dtype=torch.float32)

        transform_matrix = identity_matrix.unsqueeze(0).expand(self.batch_size, 2, 3).to(device)

        # Generate the sampling grid
        # todo make this better
        rich_feature_dimension = 256

        self.sampling_grid = F.affine_grid(transform_matrix,
                                           torch.Size((self.batch_size, rich_feature_dimension, img_h, img_w)))

        self.sampling_grid.to(self.device)

        # original layer
        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.pred_mask1 = predict_mask(32, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32, 16)
        self.pred_mask0 = predict_mask(16, self.assign_ch)

        self.softmax = nn.Softmax(1)

        self.to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        ## original forward
        out1 = self.conv0b(self.conv0a(x))
        out2 = self.conv1b(self.conv1a(out1))
        out3 = self.conv2b(self.conv2a(out2))
        out4 = self.conv3b(self.conv3a(out3))
        out5 = self.conv4b(self.conv4a(out4))

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob_neighborhood = self.softmax(mask0)

        ##
        membership_maps = torch.zeros(self.prob_all_shape).to(self.device)
        membership_maps.scatter_(1, self.spix_idx_tensor, prob_neighborhood)

        features_map = F.grid_sample(out5, self.sampling_grid)

        return membership_maps, features_map

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def load_spix_net(self, state_dict):
        # todo
        pass

    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name not in own_state:
    #             continue
    #         if isinstance(param, Parameter):
    #             # backwards compatibility for serialized parameters
    #             param = param.data
    #         own_state[name].copy_(param)


# def SpixelNet1l( data=None):
#     # Model without  batch normalization
#     model = SpixelNet(batchNorm=False)
#     if data is not None:
#         model.load_state_dict(data['state_dict'])
#     return model
#
#
# def SpixelNet1l_bn(data=None):
#     # model with batch normalization
#     model = SpixelNet(batchNorm=True)
#     if data is not None:
#         model.load_state_dict(data['state_dict'])
#     return model
#
