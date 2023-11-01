import numpy as np
from scipy.ndimage import label
from torch_geometric.data import Data
from torchvision.models import vgg16, resnet18
from torchvision.transforms import Resize, ToTensor, Pad, InterpolationMode
import torch.nn as nn
from torch_geometric.utils import from_networkx
from skimage.segmentation import slic
import networkx as nx
import scipy
import torch
import scipy.ndimage
import scipy.spatial
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import PIL
from PIL import Image

class ImageToGraph:
    """ Image to Graph"""

    def __init__(self,
                 feature_generator: nn.Module,
                 n_sp: int = 75,
                 device: torch.device = None,
                 ):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_generator = feature_generator
        self.feature_generator.to(self.device)
        self.feature_generator.eval()
        self.n_sp = n_sp

    def image_to_graph(self):
        pass

    def segment_image(self):
        pass


class FeatureExpansionGraphGenerator(ImageToGraph):
    def __init__(self,
                 feature_generator: nn.Module,
                 feature_dims,
                 n_sp: int = 75,
                 device: torch.device = None
                 ):
        super(FeatureExpansionGraphGenerator, self).__init__(feature_generator, n_sp, device)

    def __call__(self, img):
        if type(img) == PIL.BmpImagePlugin.BmpImageFile:
            img = np.array(img)

        img_h, img_w, img_c = img.shape

        superpixels = slic(img, n_segments=self.n_sp, slic_zero=True)
        sp_indices = np.unique(superpixels)

        shrinked_features = self.feature_generator(img)

        identity_matrix = torch.tensor([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]], dtype=torch.float32)

        transform_matrix = identity_matrix.to(self.device)

        sampling_grid = F.affine_grid(transform_matrix,
                                           torch.Size((rich_feature_dimension, img_h, img_w)))
                                           # torch.Size((rich_feature_dimension, img_h, img_w)))








