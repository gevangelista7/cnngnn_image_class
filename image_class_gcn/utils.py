import torch
import joblib
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pandas import read_pickle
import os


label_map = {
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
partition_map = {
    'Training': 0,
    'Test_1': 1,
    'Test_2': 2,
}


def convert_to_pyg(g, label):
    g = from_networkx(g)

    data = Data(
        edge_index=g.edge_index,
        x=g.features,
        edge_attr=g.weight.unsqueeze(-1),
        y=label_map[label]
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
    data_list = [data for data in data_list if data.partition == partition_map[partition]]
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def df2pyg_list(path):
    df = joblib.load(path)
    data_list = []
    root, file = os.path.split(path)
    dest_file = 'pyg_list_'+file

    for idx, row in df.iterrows():
        data = convert_to_pyg(row['graph'], row['label'])
        data['partition'] = partition_map[row['partition']]
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


