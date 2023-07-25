from joblib import load
from torch_geometric.nn import global_mean_pool

from image_class_gcn.models import GCN, GAT
from model_evaluation import test, instrumented_train
from utils import models_dir, graphs_dir, loader_from_pyg_list

exp_names = {'exp1', 'exp4', 'exp6'}

models_files = {'exp1': 'GATConv_2023-06-09T23_47_50.409051',
                'exp4': 'GCNConv_2023-06-11T19_37_26.375840',
                'exp6': 'GCNConv_2023-06-11T21_10_10.856862'}
device = 'cuda'
models = {
    'exp1': GAT(input_dim=3, output_dim=10, hidden_dim=8, num_layers=4, pooling=global_mean_pool, heads=4).to(device),
    'exp4': GCN(input_dim=512, output_dim=10, hidden_dim=32, num_layers=4, pooling=global_mean_pool).to(device),
    'exp6': GCN(input_dim=512, output_dim=10, hidden_dim=32, num_layers=4, pooling=global_mean_pool).to(device)
}

#
# ds_files = [
#     {'exp1': '/home/gabriel/thesis/workspace/datasets/UATD_graphs/slic/autocontrast1_knn_w.pkl'},
#     {'exp4': '/home/gabriel/thesis/workspace/datasets/UATD_graphs/cnn/original_knn_w_75_ResNet18_cut8.pkl'},
#     {'exp6': '/home/gabriel/thesis/workspace/datasets/UATD_graphs/cnn/original_knn_w_75_ResNet18_cut8_ft.pkl'}
# ]

ds_files = {'exp1': graphs_dir+'/slic/autocontrast1_knn_w.pkl',
            'exp4': graphs_dir+'/cnn/original_knn_w_75_ResNet18_cut8',
            'exp6': graphs_dir+'/cnn/original_knn_w_75_ResNet18_cut8_ft'}


if __name__ == '__main__':
    for exp in exp_names:
        ds_file = ds_files[exp]
        model = models[exp]
        instrumented_train(model=model, device=device, run_test=True, file_name=ds_file, n_epochs=300)

