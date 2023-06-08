from models import GCN, GIN, GAT
from train import instrumented_train
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
import torch
import itertools
import argparse
import os
import sys
sys.path.append('.')

torch.manual_seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_start', default=0, type=int)
    args = parser.parse_args()

    iter_start = args.iter_start

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_dir = '../datasets/UATD_graphs' if os.name == 'nt' else '/home/gabriel/thesis/dataset/UATD_graphs'

    grid = {
        'hidden_dim': [16, 32],
        'num_layers': [2, 4, 8],
        'pooling': [global_mean_pool, global_add_pool],
        'preprocessing': [root_dir+'/pixel_inf/original_knn_w_pix_300.pkl',
                          root_dir+'/slic/autocontrast1_knn_w.pkl',
                          root_dir+'/slic/equalize_knn_w.pkl',
                          root_dir+'/slic/original_knn_w.pkl'
        ]
    }
    n_epochs = 300
    iter_count = iter_start

    for parameters in itertools.islice(itertools.product(*grid.values()), iter_start, None):
        hidden_dim = parameters[0]
        num_layers = parameters[1]
        pooling = parameters[2]
        preprocessing = parameters[3]

        GCN_model = GCN(input_dim=3, output_dim=10,
                        hidden_dim=hidden_dim, num_layers=num_layers, pooling=pooling).to(device)

        GAT_model = GAT(input_dim=3, output_dim=10,
                        hidden_dim=hidden_dim, num_layers=num_layers, pooling=pooling, heads=4).to(device)

        GIN_model = GIN(input_dim=3, output_dim=10,
                        hidden_dim=hidden_dim, num_layers=num_layers, pooling=pooling).to(device)

        print('Beginning of test with the parameters: ' + str(parameters))

        print('=== GCN model ===')
        print(str(parameters))
        instrumented_train(GCN_model, preprocessing, device, n_epochs)

        print('=== GAT model ===')
        print(str(parameters))
        instrumented_train(GAT_model, preprocessing, device, n_epochs)

        print('=== GIN model ===')
        print(str(parameters))
        instrumented_train(GIN_model, preprocessing, device, n_epochs)

        iter_count += 1
        with open(root_dir+'/log_last_iter.txt', 'w') as f:
            f.write('last iter: ' + iter_count)
