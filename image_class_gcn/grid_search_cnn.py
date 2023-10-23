from models import GCN, GIN, GAT
from model_evaluation import instrumented_train
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from utils import datasets_dir, code_dir
import torch
import itertools
import argparse
import os
import sys
import joblib
sys.path.append('.')

torch.manual_seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_start', default=0, type=int)
    args = parser.parse_args()

    iter_start = args.iter_start

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graphs_dir = datasets_dir + '/UATD_graphs'

    grid = {
        'hidden_dim': [8, 32],
        'num_layers': [4, 8],
        # 'pooling': [global_mean_pool, global_add_pool],
        'pooling': [global_mean_pool],
        'preprocessing': [
            # graphs_dir+'/pixel_inf/original_knn_w_pix_300.pkl',
            # graphs_dir+'/slic/autocontrast1_knn_w.pkl',
            # graphs_dir+'/slic/equalize_knn_w.pkl',
            # graphs_dir+'/slic/original_knn_w.pkl',

            graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut1',
            graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut2',
            graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut5',
            # graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut6',
            graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut8',

            graphs_dir + '/cnn/autocontrast1_knn_w_75_ResNet18_cut1_ft',
            graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut2_ft',
            graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut5_ft',
            # graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut6_ft',
            graphs_dir+'/cnn/autocontrast1_knn_w_75_ResNet18_cut8_ft',

            # graphs_dir+'/intensity/original_knn_w_75',

            graphs_dir+'/cnn/original_knn_w_75_ResNet18_cut1',
            graphs_dir+'/cnn/original_knn_w_75_ResNet18_cut2',
            graphs_dir+'/cnn/original_knn_w_75_ResNet18_cut5',
            # graphs_dir+'/cnn/original_knn_w_75_ResNet18_cut6',
            graphs_dir+'/cnn/original_knn_w_75_ResNet18_cut8',
            #
            graphs_dir + '/cnn/original_knn_w_75_ResNet18_cut1_ft',
            graphs_dir + '/cnn/original_knn_w_75_ResNet18_cut2_ft',
            graphs_dir + '/cnn/original_knn_w_75_ResNet18_cut5_ft',
            # graphs_dir + '/cnn/original_knn_w_75_ResNet18_cut6_ft',
            graphs_dir + '/cnn/original_knn_w_75_ResNet18_cut8_ft',

            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut1',
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut2',
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut5',
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut6',
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut8',
            #
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut1_ft',
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut2_ft',
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut5_ft',
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut6_ft',
            # graphs_dir + '/cnn/equalize_knn_w_75_ResNet18_cut8_ft',
        ]
    }
    n_epochs = 100
    iter_count = iter_start

    for parameters in itertools.islice(itertools.product(*grid.values()), iter_start, None):
        hidden_dim = parameters[0]
        num_layers = parameters[1]
        pooling = parameters[2]
        preprocessing = parameters[3]

        ds = joblib.load(preprocessing)
        num_features = ds[0].x.shape[1]

        GCN_model = GCN(input_dim=num_features, output_dim=10,
                        hidden_dim=hidden_dim, num_layers=num_layers, pooling=pooling).to(device)

        GAT_model = GAT(input_dim=num_features, output_dim=10,
                        hidden_dim=hidden_dim, num_layers=num_layers, pooling=pooling, heads=4).to(device)

        # GIN_model = GIN(input_dim=num_features, output_dim=10,
        #                 hidden_dim=hidden_dim, num_layers=num_layers, pooling=pooling).to(device)

        print('Beginning of test with the parameters: ' + str(parameters))

        print('=== GCN model ===')
        print(str(parameters))
        instrumented_train(GCN_model, preprocessing, device, n_epochs)

        print('=== GAT model ===')
        print(str(parameters))
        instrumented_train(GAT_model, preprocessing, device, n_epochs)

        # print('=== GIN model ===')
        # print(str(parameters))
        # instrumented_train(GIN_model, preprocessing, device, n_epochs)

        iter_count += 1
        with open(code_dir+'/log_last_iter.txt', 'w') as f:
            f.write('last iter: ' + str(iter_count))
