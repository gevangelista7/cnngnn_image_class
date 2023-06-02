from datetime import datetime
import time

import pandas as pd
import torch

from models import GIN
from utils import loader_from_pyg_list
from torch_geometric.nn.pool import global_mean_pool

torch.manual_seed(42)


def train(model, train_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for data in train_loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, test_loader, device):
    model.eval()
    correct = 0
    for data in test_loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(test_loader.dataset)


def instrumented_train(model, file_name, device, n_epochs=100):
    train_loader = loader_from_pyg_list(file_name, partition='Training', shuffle=True)
    test_loader = loader_from_pyg_list(file_name, partition='Test_1', shuffle=True)

    start = time.time()
    results = []
    elapsed_time = 0
    max_train_acc = 0
    max_test_acc = 0
    best_model = None
    output_filename = model.model_name+"_"+datetime.now().isoformat().replace(':', '_')

    for epoch in range(1, n_epochs+1):
        train(model, train_loader, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        elapsed_time = time.time() - start

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}, ElapsedTime: {elapsed_time:.1f}s')

        if train_acc > max_train_acc:
            max_train_acc = train_acc

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            print('Best model! Saving Checkpoint...')
            torch.save(model.state_dict(), './models/' + output_filename)

        results.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'test_acc': test_acc
        })

    results = pd.DataFrame.from_records(results)
    results.attrs['model'] = model.model_name
    results.attrs['n_layers'] = model.num_layers
    results.attrs['hidden_channels'] = model.hidden_dim
    results.attrs['pooling'] = model.pooling_name

    results.attrs['datetime'] = time.asctime()
    results.attrs['preprocess_file'] = file_name
    results.attrs['runtime'] = elapsed_time
    results.attrs['best_train_acc'] = max_train_acc
    results.attrs['best_test_acc'] = max_test_acc
    results.attrs['best_model'] = best_model

    results.to_pickle('./results/'+output_filename)


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     file_name = '../datasets/UATD_graphs/pixel_inf/pyg_list_original_knn_w_pix_300.pkl'
#
#     # model = GCN4l(hidden_channels=64, num_classes=10, num_node_features=3).to(device)
#     # model = GCN(input_dim=3, hidden_dim=64, num_layers=2, output_dim=10, pooling=global_mean_pool).to(device)
#     # model = GenericGNN(input_dim=3, hidden_dim=64, num_layers=2, output_dim=10,
#     #                    pooling=global_mean_pool, layer=GCNConv).to(device)
#     model = GIN(input_dim=3, hidden_dim=64, num_layers=2, output_dim=10,
#                 pooling=global_mean_pool).to(device)
#
#     instrumented_train(model, file_name, 3)

