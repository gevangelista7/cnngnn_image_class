import numpy as np
from torch_geometric.data import Data
from torchvision.models import vgg16
from torchvision.transforms import Resize, ToTensor, Pad
import torch.nn as nn
from torch_geometric.utils import from_networkx
from skimage.segmentation import slic
import networkx as nx
import scipy
import torch
import scipy.ndimage
import scipy.spatial
from scipy.spatial.distance import cdist
from utils import label_map, partition_map


half_vgg16_conv2 = nn.Sequential(*list(vgg16(pretrained=True).features)[:8])


class ITGProcessor:
    """ ImageToGraph"""
    def __init__(self, backbone_net=half_vgg16_conv2,
                 segmentation_function=slic, device=None,
                 n_sp=75, weighted=True, knn_k=8):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone_net = backbone_net
        self.backbone_net.to(self.device)
        self.backbone_net.eval()

        self.n_sp = n_sp
        self.knn_k = knn_k
        self.weighted = weighted

        self.segmentation_function = segmentation_function

    def segment_image(self, img):
        """ Input: PIL image """

        img = np.array(img, dtype=np.float32)
        img = (img / 255.)

        # get the segmentation mask
        superpixels = self.segmentation_function(img, n_segments=self.n_sp, slic_zero=True)
        sp_indices = np.unique(superpixels)

        # find the final number of superpixels
        ind = np.arange(sp_indices.max())
        sp_order = sp_indices[ind].astype(np.int32)

        # standardize the shape for HxWxC and get the number of channels
        if len(img.shape) == 2:
            img = img[:, :, None]
        n_ch = 1 if img.shape[2] == 1 else 3

        # collect the intensity by channel and center position of each superpixel
        sp_intensity, sp_coord = [], []
        for seg in sp_order:
            # channel mean intensity calc
            mask = (superpixels == seg).squeeze()
            avg_value = np.zeros(n_ch)
            for c in range(n_ch):
                avg_value[c] = np.mean(img[:, :, c][mask])
            sp_intensity.append(avg_value)

            # superpixel center position
            cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
            sp_coord.append(cntr)
        sp_intensity = np.array(sp_intensity, np.float32)
        sp_coord = np.array(sp_coord, np.float32)

        return sp_intensity, sp_coord, sp_order, superpixels

    def full_adj_matrix(self, coord):
        n_nodes = coord.size
        if not self.weighted:
            return np.ones((n_nodes, n_nodes))
        else:
            raise NotImplementedError()

    def calc_coord_dist(self, coord):
        dist = cdist(coord, coord, 'seuclidean')
        return dist

    def calc_pix_coord_dist(self, coord, pix, gamma=.5):
        coord_dist = cdist(coord, coord, 'seuclidean')
        pix_dist = cdist(pix, pix, 'seuclidean')

        dist = gamma * np.exp(-coord_dist ** 2) + (1 - gamma) / (1 + np.exp(-pix_dist ** 2))
        return dist

    def knn_adj_matrix(self, coord):
        dist = self.calc_coord_dist(coord)
        adj_matrix = self.build_dist_knn_adj_matrix(dist)
        return adj_matrix

    def knn_pix_adj_matrix(self, coord, pix, gamma=.5):
        dist = self.calc_pix_coord_dist(coord, pix, gamma)
        adj_matrix = self.build_dist_knn_adj_matrix(dist)
        return adj_matrix

    def build_dist_knn_adj_matrix(self, dist):
        # get the indices of the k+1 (from KNN) closer nodes.
        # There will always be a self-loop that must be thrown out later
        indices = np.argpartition(dist, self.knn_k + 1, axis=1)[:, :self.knn_k + 1]

        # construct the adjacency matrix
        adj = np.zeros_like(dist)
        adj[np.arange(dist.shape[0])[:, None], indices] = dist[np.arange(dist.shape[0])[:, None], indices]

        # cutting the self loops
        adj[np.diag_indices_from(dist)] = 0

        # normalize neighborhoods
        non_zeros = np.where(adj != 0, adj, np.nan)
        non_zero_avg = np.nanmean(non_zeros, axis=1, keepdims=True)
        adj = adj / non_zero_avg

        # Exponential function on valid values
        adj = np.where(adj != 0, np.exp(adj), 0)
        if not self.weighted:
            adj = np.where(adj != 0, 1, 0)
        adj = np.max(np.stack((adj, adj.T)), axis=0)

        return adj

    def up_sample(self, img):
        max_w = 224
        max_h = 224

        max_side, min_side = max(img.size), min(img.size)
        img = Resize(int(224*min_side/max_side))(img)

        imsize = img.size
        h_padding = (max_w - imsize[0]) / 2
        v_padding = (max_h - imsize[1]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

        img = Pad(padding)(img)

        return img

    def preprocess_img_vgg(self, img):
        img = self.up_sample(img)
        img = ToTensor()(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        return img

    def preprocess_sp_mask(self, sp_mask):
        sp_mask = self.up_sample(Image.fromarray(sp_mask.astype(np.int8)))
        sp_mask = Resize((112, 112))(sp_mask)
        sp_mask = ToTensor()(sp_mask)
        sp_mask = sp_mask.type(torch.int8).squeeze(0)
        sp_mask = sp_mask.to(self.device)
        return sp_mask

    def generate_node_features(self, img, sp_mask):
        """ this function must use the backbone net to generate the node features"""
        """ remember the possible not mapped node, the ones with the masked vanished in the process"""
        n_nodes = sp_mask.max()
        torch.no_grad()

        input_tensor = self.preprocess_img_vgg(img)
        image_features = self.backbone_net.forward(input_tensor)
        image_features = image_features.squeeze(0)

        sp_mask = self.preprocess_sp_mask(sp_mask)

        nodes_features = torch.zeros((n_nodes, 128))

        for node_id in range(n_nodes):
            if not (sp_mask == node_id).any():
                continue
            node_mask = sp_mask == node_id
            filtered_features = image_features * node_mask
            node_features = filtered_features.mean(dim=(1, 2)).unsqueeze(0)
            nodes_features[node_id] = node_features

        all_zero_mask = torch.all(torch.eq(nodes_features, 0), dim=1)
        for i in range(nodes_features.shape[0]):
            if all_zero_mask[i]:
                if i == 0:
                    nodes_features[i] = nodes_features[i+1]
                elif i == nodes_features.shape[0] - 1:
                    nodes_features[i] = nodes_features[i - 1]
                else:
                    nodes_features[i] = (nodes_features[i-1] + nodes_features[i+1])/2.0
        nodes_features.detach_()
        return nodes_features

    def compute_graph_from_data(self, nodes, adjacency_matrix, label, partition):
        """ Convert preprocessed data to pytorch geometric Data"""

        # G = nx.from_numpy_matrix(adjacency_matrix)
        # for i, node in enumerate(nodes):
        #     G.nodes[i]['features'] = node
        #
        # g = from_networkx(G)
        #
        # data = Data(
        #     edge_index=g.edge_index[:3],
        #     x=g.features,
        #     edge_attr=g.weight.unsqueeze(-1),
        #     y=label
        # )

        g = from_networkx(nx.from_numpy_matrix(adjacency_matrix))

        g = Data(
            x=nodes,
            y=label,
            edge_index=g.edge_index,
            edge_attr=g.weight.unsqueeze(-1),
            partition=partition,
        )

        return g

    def img2graph_intensity(self, img, label, partition, pixel_dist=False):
        sp_intensity, sp_coord, _, _ = self.segment_image(img)
        if pixel_dist:
            adj = self.knn_pix_adj_matrix(sp_coord, sp_intensity)
        else:
            adj = self.knn_adj_matrix(sp_coord)

        node_features = sp_intensity
        label = label_map[label]
        g = self.compute_graph_from_data(node_features, adj)
        g.y = label_map[label]
        g.partition = partition_map[partition]
        return g

    def img2graph_vgg(self, img, label, partition, pixel_dist=False):
        sp_intensity, sp_coord, _, superpixels = self.segment_image(img)
        if pixel_dist:
            adj = self.knn_pix_adj_matrix(sp_coord, sp_intensity)
        else:
            adj = self.knn_adj_matrix(sp_coord)

        nodes_features = self.generate_node_features(img, superpixels)
        g = self.compute_graph_from_data(nodes_features, adj,
                                         label=label_map[label],
                                         partition=partition_map[partition])
        return g


if __name__ == '__main__':
    from PIL import Image
    img = Image.open('278.bmp')
    img_np = np.array(img)

    sp_intensity, sp_coord, sp_order, superpixels = ITGProcessor().segment_image(img_np)
    vgg_graph = ITGProcessor().img2graph_vgg(img, 'plane', 'Test_1')
