from datetime import datetime
import time

import pandas as pd
import torch
from utils import loader_from_pyg_list

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


def validate(model, test_loader, device):
    model.eval()
    correct = 0
    for data in test_loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(test_loader.dataset)


def test(model, test_loader, device):
    model.eval()
    # num_classes = model.num_classes                                     # Obter o número de classes do modelo
    num_classes = 10
    correct = torch.zeros(num_classes, dtype=torch.float32).to(device)  # Contador de predições corretas por classe
    total = torch.zeros(num_classes, dtype=torch.float32).to(device)    # Contador de amostras por classe

    for data in test_loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)

        # Calcular as precisões por classe
        for i in range(num_classes):
            correct[i] += int(((pred == i) & (data.y == i)).sum())
            total[i] += int((data.y == i).sum())

    class_accuracies = correct / total              # recall médias por classe

    overall_accuracy = correct.sum() / total.sum()  # recall geral

    return overall_accuracy, class_accuracies


def instrumented_train(model, file_name, device, n_epochs=100, run_test=False):
    train_loader = loader_from_pyg_list(file_name, partition='Training', shuffle=True)
    val_loader = loader_from_pyg_list(file_name, partition='Test_1', shuffle=True)

    start = time.time()
    results = []
    elapsed_time = 0
    max_train_acc = 0
    max_val_acc = 0
    finnish_counter = 0
    best_model = None
    output_filename = model.model_name+"_"+datetime.now().isoformat().replace(':', '_')

    for epoch in range(1, n_epochs+1):
        train(model, train_loader, device)
        train_acc = validate(model, train_loader, device)
        val_acc = validate(model, val_loader, device)
        elapsed_time = time.time() - start

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, '
              f'Test Acc: {val_acc:.4f}, ElapsedTime: {elapsed_time:.1f}s')

        if train_acc > max_train_acc:
            max_train_acc = train_acc

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            finnish_counter = 0
            print('Best model! Saving Checkpoint...')
            torch.save(model.state_dict(), './models/' + output_filename)

        results.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'test_acc': val_acc
        })

        if val_acc < max_val_acc - .2:
            finnish_counter += 1
            if finnish_counter > 50:
                break

    results = pd.DataFrame.from_records(results)
    results.attrs['model'] = model.model_name
    results.attrs['n_layers'] = model.num_layers
    results.attrs['hidden_channels'] = model.hidden_dim
    results.attrs['pooling'] = model.pooling_name
    results.attrs['final_epoch'] = epoch

    results.attrs['datetime'] = time.asctime()
    results.attrs['preprocess_file'] = file_name
    results.attrs['runtime'] = elapsed_time
    results.attrs['best_train_acc'] = max_train_acc
    results.attrs['best_val_acc'] = max_val_acc
    results.attrs['best_model'] = best_model

    if run_test:
        test_loader = loader_from_pyg_list(file_name, partition='Test_2', shuffle=True)
        overall_accuracy, class_accuracies = test(model, test_loader, device)
        results.attrs['test_acc'] = overall_accuracy
        results.attrs['class_acc'] = class_accuracies
        output_filename = 'TESTED_'+output_filename

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
