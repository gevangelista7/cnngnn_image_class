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


def calc_sp_centers(membership_map):
    batch_size, sp, h, w = membership_map.shape

    x_grid, y_grid = torch.meshgrid(torch.arange(h), torch.arange(w))

    x_grid = x_grid.to(membership_map.device)
    x_grid = x_grid / x_grid.max()

    y_grid = y_grid.to(membership_map.device)
    y_grid = y_grid / y_grid.max()

    # Expand the grids to match the batch size and 'sp' dimensions
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)

    # Multiply the grids by the membership tensor
    x_center = torch.sum(x_grid * membership_map, dim=(2, 3)) / (torch.sum(membership_map, dim=(2, 3)) + 1e-10)
    y_center = torch.sum(y_grid * membership_map, dim=(2, 3)) / (torch.sum(membership_map, dim=(2, 3)) + 1e-10)

    centers = torch.stack((x_center, y_center), dim=-1)

    return centers


def calc_sp_features(membership_maps, features_map):
    bs, n_sp, _, _ = membership_maps.shape
    n_feat = features_map.shape[1]
    sp_features = membership_maps.view(bs, n_sp, -1) @ features_map.view(bs, -1, n_feat)
    sp_features = sp_features / sp_features.sum(dim=1)
    return sp_features


def calc_corr_matrix(membership_maps, features_map, pos_enc=True):
    sp_features = calc_sp_features(membership_maps, features_map)
    if pos_enc:
        centers = calc_sp_centers(membership_maps)
        sp_features = torch.concat((sp_features, centers), dim=-1)

    corr_matrix = sp_features @ sp_features.transpose(1, 2)

    return corr_matrix


def knn_adjacency_matrix(correlation_matrix, k_neighbors=8):

    batch_size, n_sp, _ = correlation_matrix.shape

    adjacency_matrix = np.zeros_like(correlation_matrix)

    # Loop through each correlation matrix in the batch
    for i in range(batch_size):
        for j in range(n_sp):
            # Find the indices of the k largest values in the correlation matrix
            k_indices = np.argpartition(correlation_matrix[i, j], -k_neighbors)[-k_neighbors:]

            # Set the corresponding entries in the adjacency matrix to 1
            adjacency_matrix[i, j, k_indices] = 1

    return adjacency_matrix


def knn_edges_from_corr(correlation_matrix, k_neighbors=8):

    batch_size, n_sp, _ = correlation_matrix.shape

    edge_lists = []
    for i in range(batch_size):
        edges = []
        for j in range(n_sp):
            # Find the indices of the k largest values in the correlation matrix
            k_indices = np.argpartition(correlation_matrix[i, j], -k_neighbors)[-k_neighbors:]
            edges.extend([(j, idx) for idx in k_indices])

        edge_lists.append(edges)

    return edge_lists

