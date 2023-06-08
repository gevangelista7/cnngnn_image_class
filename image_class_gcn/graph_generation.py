import numpy as np
import scipy
from networkx import adjacency_matrix
from skimage.segmentation import slic
import scipy.ndimage
import scipy.spatial
import argparse
from scipy.spatial.distance import cdist
import networkx as nx


def process_image(img, segmentation_function, n_sp=75):
    assert img.dtype == np.uint8, img.dtype
    img = (img / 255.).astype(np.float32)

    superpixels = segmentation_function(img, n_segments=n_sp, slic_zero=True)
    sp_indices = np.unique(superpixels)

    ind = np.arange(sp_indices.max())

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]

    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord = [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
        cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        sp_intensity.append(avg_value)
        sp_coord.append(cntr)
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)

    return sp_intensity, sp_coord, sp_order, superpixels


def compute_adjacency_matrix(coord, knn=True, weighted=True, k=8):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord, 'euclidean')

    # get the indices of the k+1 (from KNN) closer nodes. There will always be a self-loop that must be thrown out
    indices = np.argpartition(dist, k + 1, axis=1)[:, :k + 1]

    # allocate memory for the adjacency matrix
    adj = np.zeros_like(dist)

    if knn:
        adj[np.arange(dist.shape[0])[:, None], indices] = dist[np.arange(dist.shape[0])[:, None], indices]

    # Cutting the self loops
    adj[np.diag_indices_from(dist)] = 0

    # Normalize neighborhoods
    non_zeros = np.where(adj != 0, adj, np.nan)
    non_zero_avg = np.nanmean(non_zeros, axis=1, keepdims=True)
    adj = adj / non_zero_avg

    # Exponential function on valid values
    adj = np.where(adj != 0, np.exp(adj), 0)

    if not weighted:
        adj = np.where(adj != 0, 1, 0)

    adj = np.max(np.stack((adj, adj.T)), axis=0)

    return adj


def compute_adj_matrix_pixel(coord, pix, weighted=True, k=8, gamma=.5):
    coord_dist = cdist(coord, coord, 'seuclidean')
    pix_dist = cdist(pix, pix, 'seuclidean')

    dist = gamma * np.exp(-coord_dist ** 2) + (1 - gamma) / np.exp(-pix_dist ** 2)

    # get the indices of the k+1 (from KNN) closer nodes. There will always be a self-loop that must be thrown out
    indices = np.argpartition(dist, k + 1, axis=1)[:, :k + 1]

    # allocate memory for the adjacency matrix
    adj = np.zeros_like(dist, dtype=np.float32)

    adj[np.arange(dist.shape[0])[:, None], indices] = dist[np.arange(dist.shape[0])[:, None], indices]

    # Cutting the self loops
    adj[np.diag_indices_from(dist)] = 0

    if not weighted:
        adj = np.where(adj != 0, 1, 0)

    adj = np.max(np.stack((adj, adj.T)), axis=0)

    return adj


def compute_graph_from_data(nodes, adjacency_matrix):
    G = nx.from_numpy_matrix(adjacency_matrix)
    for i, node in enumerate(nodes):
        G.nodes[i]['features'] = node

    return G


def img2graph(img, knn=True, weighted=True, n_sp=75, k=8, pixel_dist=False):
    sp_intensity, sp_coord, _, _ = process_image(img, n_sp=n_sp)
    if pixel_dist:
        adj = compute_adj_matrix_pixel(sp_coord, sp_intensity, weighted=weighted, k=k)
    else:
        adj = compute_adjacency_matrix(sp_coord, knn=knn, weighted=weighted, k=k)
    G = compute_graph_from_data(sp_intensity, adj)

    return G


if __name__ == "__main__":
    # test site
    # ds = pd.read_pickle('./UATD_graphs/slic/equalize_knn_w.pkl')
    # g = ds.graphs[0]

    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open('../datasets/UATD_classification/samples_equalize/Test_1/ball/1.bmp')
    # img = Image.open('UATD/UATD_Training/images/00001.bmp')
    g = img2graph(np.array(img), n_sp=1000, pixel_dist=True)
    print('fim')
