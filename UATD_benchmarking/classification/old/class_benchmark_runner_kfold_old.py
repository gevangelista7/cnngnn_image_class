import numpy as np
import pandas as pd
from datasets import load_dataset
from timm import create_model
from utils import fit_and_pad_map, datasets_dir, models_dir, results_dir
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report

from torchvision import transforms, datasets
from utils import FitAndPad

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

model_names = ["vgg16.tv_in1k", "resnet18.tv_in1k", "pvt_v2_b1.in1k"]
# seeds = [3, 5, 11, 19, 22, 28, 1947, 1989, 1990, 2017]
seeds = [3, 5, 11, 19, 22, 28, 1947, 1989, 1990, 2017]
num_epochs = 20

if __name__ == '__main__':
    #
    dataset = load_dataset("imagefolder",
                           task="image-classification",
                           data_dir=datasets_dir+"/Training")

    dataset = dataset.map(fit_and_pad_map)
    dataset.set_format("torch")

    num_classes = 10
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        # k-fold
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Begin of folding")
        ds_split_train_test = dataset['train'].train_test_split(test_size=0.9, seed=seed)
        train_ds, test_ds = ds_split_train_test["train"], ds_split_train_test["test"]

        ds_split_train_val = train_ds.train_test_split(test_size=0.1 / 0.9, seed=seed)
        train_ds, val_ds = ds_split_train_val["train"], ds_split_train_val["test"]

        # test_ds, val_ds, train_ds = torch.utils.data.random_split(ds, [int(n_samples * 0.1),
        #                                                                int(n_samples * 0.1),
        #                                                                n_samples - 2 * int(n_samples * 0.1)])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        for model_name in model_names:
            print('Begin model' + model_name + "training")

            model = create_model(model_name, pretrained=True, num_classes=num_classes)
            model = model.cuda()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            results = {'epoch': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'reports': []}

            best_val_acc = - np.inf
            best_f1 = 0.0

            # Training loop
            print("training")
            for epoch in range(num_epochs):
                model.train()
                for batch in train_loader:
                    inputs = batch['image'].cuda()
                    labels = batch['labels'].cuda()

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_preds = []
                val_labels = []
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch['image'].cuda()
                        labels = batch['labels'].cuda()
                        val_labels.extend(labels.cpu().numpy())

                        outputs = model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        val_preds.extend(preds.cpu().numpy())

                val_f1 = f1_score(val_labels, val_preds, average='macro')
                val_precision = precision_score(val_labels, val_preds, average='macro')
                val_recall = recall_score(val_labels, val_preds, average='macro')
                val_accuracy = accuracy_score(val_labels, val_preds)
                val_report = classification_report(val_labels, val_preds)

                print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                      f"Acc: {val_accuracy:.4}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    print("Best model, saving checkpoint...")
                    torch.save(model.state_dict(), models_dir+f"/{model_name}_{seed}_best_model_checkpoint.pt")

                results['epoch'].append(epoch + 1)
                results['accuracy'].append(val_accuracy)
                results['precision'].append(val_precision)
                results['recall'].append(val_recall)
                results['f1'].append(val_f1)
                results['reports'].append(val_report)

                df = pd.DataFrame(results)
                df.attrs['model_name'] = model_name
                df.to_csv(results_dir+f"/{model_name}_{seed}_val_results.csv", index=False)

            print("loading best model for test")
            model.load_state_dict(torch.load(models_dir+f"/{model_name}_{seed}_best_model_checkpoint.pt"))
            model.to(device)

            # Testing
            print('begin of testing')
            model.eval()
            test_preds = []
            test_labels = []
            results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'reports': []}

            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch['image'].cuda()
                    labels = batch['labels'].cuda()
                    test_labels.extend(labels.cpu().numpy())

                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    test_preds.extend(preds.cpu().numpy())

            test_f1 = f1_score(test_labels, test_preds, average='macro')
            test_precision = precision_score(test_labels, test_preds, average='macro')
            test_recall = recall_score(test_labels, test_preds, average='macro')
            test_accuracy = accuracy_score(test_labels, test_preds)
            test_reports = classification_report(test_labels, test_preds)

            results['accuracy'].append(test_accuracy)
            results['precision'].append(test_precision)
            results['recall'].append(test_recall)
            results['f1'].append(test_f1)
            results['reports'].append(test_reports)

            print(f"Test F2: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

            # Save results to a file
            df = pd.DataFrame(results)
            df.to_csv(f"./results/{model_name}_{seed}_test_results.csv", index=False)
