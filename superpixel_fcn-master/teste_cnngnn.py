from models.Spixel_single_layer_gnn import CNNGNNModel
import torch
import numpy as np

import argparse
import os
import torch.backends.cudnn as cudnn
import models
import torchvision.transforms as transforms
import flow_transforms
from skimage.io import imread, imsave
from loss import *
import time
import random
from glob import glob
from models.model_utils_gnncnn import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = 'demo/inputs/Lena.jpg'
    imgId = os.path.basename(img_file)[:-4]

    # may get 4 channel (alpha channel) for some format
    img_ = imread(img_file)[:, :, :3]
    H, W, _ = img_.shape
    H_, W_ = int(np.ceil(H / 16.) * 16), int(np.ceil(W / 16.) * 16)

    # get spixel id
    downsize = 16
    n_spixl_h = int(np.floor(H_ / downsize))
    n_spixl_w = int(np.floor(W_ / downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, downsize, axis=1), downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

    n_spixel = int(n_spixl_h * n_spixl_w)

    img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    model = CNNGNNModel(input_shape=(1,)+img.shape, device=device)

    # compute output
    tic = time.time()
    membership_maps, features_map = model(img1.unsqueeze(0).to(device))
    toc = time.time() - tic
    print(f'forward time: {toc}')


    # spix_idx_tensor = torch.arange(18).reshape((2, 3, 3))
    # prob_neighbourhood = (torch.randn(size=spix_idx_tensor.shape) * 0.3 + 0.5).clamp(0, 1).unsqueeze(0)
    # prob_all = torch.zeros((18, 3, 3)).unsqueeze(0)
    #
    # for batchidx, batch in enumerate(prob_neighbourhood):
    #     for channel_neigh, prob_ch in enumerate(batch):
    #         for hidx, row in enumerate(prob_ch):
    #             for widx, pixel in enumerate(row):
    #                 channel_all = spix_idx_tensor[channel_neigh, hidx, widx]
    #                 prob_all[batchidx, channel_all, hidx, widx] = prob_neighbourhood[batchidx, channel_neigh, hidx, widx]
    #
    # expanded_spix_idx = spix_idx_tensor.unsqueeze(0)
    # prob_all2 = torch.zeros_like(prob_all)
    # prob_all2.scatter_(1, expanded_spix_idx, prob_neighbourhood)

