import torch
import joblib
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pandas import read_pickle
from torchvision.transforms import Resize, Pad, InterpolationMode
import os


root_dir = '../' if os.name == 'nt' else '/home/gabriel/thesis/workspace'
datasets_dir = root_dir + '/datasets'
code_dir = root_dir+'/image_class_gcn'
graphs_dir = datasets_dir + '/UATD_graphs'
classification_dir = datasets_dir+'/UATD_classification'
models_dir = code_dir+'/models'
results_dir = code_dir+'/results'

label2tensor_map = {
    'ball':         torch.tensor([0]),
    'circle cage':  torch.tensor([1]),
    'cube':         torch.tensor([2]),
    'cylinder':     torch.tensor([3]),
    'human body':   torch.tensor([4]),
    'metal bucket': torch.tensor([5]),
    'plane':        torch.tensor([6]),
    'rov':          torch.tensor([7]),
    'square cage':  torch.tensor([8]),
    'tyre':         torch.tensor([9]),
}
tensor2label_map = {value: key for key, value in label2tensor_map.items()}
partition2idx_map = {
    'Training': 0,
    'Test_1': 1,
    'Test_2': 2,
}


class FitAndPad:
    def __call__(self, img, max_w=224):
        max_h = max_w

        max_side, min_side = max(img.size), min(img.size)
        img = Resize(int(max_w * min_side / max_side))(img)

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

class SPFitAndPad:
    def __call__(self, img, max_w=224):
        max_h = max_w

        max_side, min_side = max(img.size), min(img.size)
        factor = max(int(max_w * min_side / max_side), 1)
        img = Resize(factor, interpolation=InterpolationMode.NEAREST)(img)

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



class PILResize:
    def __call__(self, img):
        img = img.resize((224, 224))

        return img


def convert_to_pyg(g, label):
    g = from_networkx(g)

    data = Data(
        edge_index=g.edge_index,
        x=g.features,
        edge_attr=g.weight.unsqueeze(-1),
        y=label2tensor_map[label]
    )

    return data


def loader_from_df(path, partition, batch_size=32):
    """possible partitions: ['Test_1', 'Test_2', 'Training']"""
    assert partition in ['Test_1', 'Test_2', 'Training']

    df = read_pickle(path)
    data_list = []

    for idx, row in df.iterrows():
        if row['partition'] != partition:
            continue

        data_list.append(convert_to_pyg(row['graph'], row['label']))

    return DataLoader(data_list, batch_size=batch_size, shuffle=True)


def loader_from_pyg_list(path, partition, batch_size=32, shuffle=True):
    """possible partitions: ['Test_1', 'Test_2', 'Training']"""
    data_list = joblib.load(path)
    data_list = [data for data in data_list if data.partition == partition2idx_map[partition]]
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def df2pyg_list(path):
    df = joblib.load(path)
    data_list = []
    root, file = os.path.split(path)
    dest_file = 'pyg_list_'+file

    for idx, row in df.iterrows():
        data = convert_to_pyg(row['graph'], row['label'])
        data['partition'] = partition2idx_map[row['partition']]
        data_list.append(data)

    joblib.dump(data_list, os.path.join(root, dest_file))

#
# if __name__ == '__main__':
#     import time
#     st = time.time()
#     # a = loader_from_df('./datasets/UATD_graphs/slic/equalize_knn_w.pkl', partition='Training')
#     # df2pyg_list('./datasets/UATD_graphs/pixel_inf/original_knn_w_pix_300.pkl')
#     # loader = loader_from_pyg_list('./datasets/UATD_graphs/slic/pyg_list_equalize_knn_w.pkl',
#     #                               'Training', True)
#
#     print(time.time()-st)
