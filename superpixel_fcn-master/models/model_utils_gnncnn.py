import numpy as np
import torch
from .model_util import *
from train_util import *


def calc_n_spix(img_H, img_W, downsize):
    H_norm, W_norm = int(np.ceil(img_H / downsize) * downsize), int(np.ceil(img_W / downsize) * downsize)

    # get spixel id
    n_spixl_h = int(np.floor(H_norm / downsize))
    n_spixl_w = int(np.floor(W_norm / downsize))
    n_spixl = n_spixl_w * n_spixl_h

    return n_spixl, n_spixl_h, n_spixl_w


def create_spix_idx(img_H, img_W, downsize, batch_size):
    n_spixl, n_spixl_h, n_spixl_w = calc_n_spix(img_H, img_W, downsize)

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_region = shift9pos(spix_values)

    spix_idx_np = np.repeat(
        np.repeat(spix_idx_tensor_region, downsize, axis=1), downsize, axis=2)

    spix_idx_tensor = torch.tensor(spix_idx_np, dtype=torch.int64).repeat(batch_size, 1, 1, 1)

    return spix_idx_tensor


def one_sp_center(membership_map):
    n_sp, h, w = membership_map.size()
    x_coords, y_coords = torch.meshgrid(torch.arange(h), torch.arange(w))

    # Calculate the weighted sum of coordinates
    weighted_x = torch.sum(x_coords * membership_map)
    weighted_y = torch.sum(y_coords * membership_map)

    # Calculate the sum of tensor values
    sum_of_values = torch.sum(membership_map)

    # Calculate the center of mass
    center_x = weighted_x / sum_of_values
    center_y = weighted_y / sum_of_values

    return center_x, center_y


def all_sp_centers(membership_map):
    # todo make this with tensors
    centers = []
    for sp_membership_map in membership_map:
        centers.append(one_sp_center(sp_membership_map))

    return torch.tensor(centers)


def one_sp_features(probs_map, x_coords, y_coords):
    # Calculate the weighted sum of coordinates
    weighted_x = torch.sum(x_coords * probs_map)
    weighted_y = torch.sum(y_coords * probs_map)

    # Calculate the sum of tensor values
    sum_of_values = torch.sum(probs_map)

    # Calculate the center of mass
    center_x = weighted_x / sum_of_values
    center_y = weighted_y / sum_of_values

    return center_x, center_y

