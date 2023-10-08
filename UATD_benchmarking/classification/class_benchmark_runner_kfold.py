import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from utils import FitAndPad, classification_dir, datasets_dir, code_dir, results_dir, models_dir
from torch.utils.data import DataLoader, random_split
import pandas as pd
from timm import create_model
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report

import os

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    FitAndPad(),
    transforms.ToTensor()
])

# Load your dataset
img_treat_dir = '/samples_original'
original_dir = classification_dir+img_treat_dir

data = datasets.ImageFolder(original_dir+'/Unified', transform=transform)

seeds = [3] #, 5, 11, 19, 22, 28, 1947, 1989, 1990, 2017]
num_epochs = 20
num_classes = 10
max_val_acc = - np.inf

# model_names = ["vgg16.tv_in1k", "pvt_v2_b1.in1k"] #, "resnet18.tv_in1k"]
model_names = ["resnet18.tv_in1k"]

# Create data loaders
valid_size, test_size = 0.1, 0.1
n_test = int(test_size * len(data))
n_valid = int(valid_size * len(data))
n_train = len(data) - n_valid - n_test


if __name__ == '__main__':
    # Training loop
    criterion = nn.CrossEntropyLoss()

    for seed in seeds:
        train_ds, valid_ds, test_ds = random_split(data, [n_train, n_valid, n_test])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        for model_name in model_names:
            model = create_model(model_name, pretrained=True, num_classes=num_classes)
            model = model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            print(f"Begin to train model: {model_name}, Seed: {seed}")
            val_results = {'epoch': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'reports': []}
            test_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'reports': []}

            for epoch in range(num_epochs):
                best_f1 = - np.inf
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    train_preds = model(images)
                    loss = criterion(train_preds, labels)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_preds_all = []
                val_labels_all = []
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images = images.to(device)

                        # val_labels = labels.to(device)
                        val_preds = model(images)
                        _, predicted = torch.max(val_preds, 1)

                        val_labels_all.extend(labels.cpu().numpy())
                        val_preds_all.extend(predicted.cpu().numpy())

                    val_labels_all = torch.tensor(val_labels_all)
                    val_preds_all = torch.tensor(val_preds_all)

                val_f1 = f1_score(val_labels_all, val_preds_all, average='macro')
                val_precision = precision_score(val_labels_all, val_preds_all, average='macro')
                val_recall = recall_score(val_labels_all, val_preds_all, average='macro')
                val_accuracy = accuracy_score(val_labels_all, val_preds_all)
                val_report = classification_report(val_labels_all, val_preds_all)

                print(f"Seed {seed}, Epoch [{epoch + 1}/{num_epochs}] - "
                      f"Acc: {val_accuracy:.4}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    print("Best model, saving checkpoint...")
                    torch.save(model.state_dict(), models_dir + f"/{model_name}_{seed}_best_model_checkpoint.pt")

                val_results['epoch'].append(epoch + 1)
                val_results['accuracy'].append(val_accuracy)
                val_results['precision'].append(val_precision)
                val_results['recall'].append(val_recall)
                val_results['f1'].append(val_f1)
                val_results['reports'].append(val_report)

                df = pd.DataFrame(val_results)
                df.attrs['model_name'] = model_name
                df.to_csv(results_dir + f"/{model_name}_{seed}_val_results.csv", index=False)

            print("loading best model for test")
            model.load_state_dict(torch.load(models_dir + f"/{model_name}_{seed}_best_model_checkpoint.pt"))
            model.to(device)

            # Testing
            print('begin of testing')
            model.eval()
            test_preds_all = []
            test_labels_all = []
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)

                    # test_labels = labels.to(device)
                    test_preds = model(images)
                    _, predicted = torch.max(val_preds, 1)

                    test_labels_all.extend(labels.cpu().numpy())
                    test_preds_all.extend(predicted.cpu().numpy())

                test_labels_all = torch.tensor(val_labels_all)
                test_preds_all = torch.tensor(val_preds_all)

            test_f1 = f1_score(test_labels_all, test_preds_all, average='macro')
            test_precision = precision_score(test_labels_all, test_preds_all, average='macro')
            test_recall = recall_score(test_labels_all, test_preds_all, average='macro')
            test_accuracy = accuracy_score(test_labels_all, test_preds_all)
            test_reports = classification_report(test_labels_all, test_preds_all)

            test_results['accuracy'].append(test_accuracy)
            test_results['precision'].append(test_precision)
            test_results['recall'].append(test_recall)
            test_results['f1'].append(test_f1)
            test_results['reports'].append(test_reports)

            print(f"Seed: {seed}, Test F2: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

            # Save results to a file
            df = pd.DataFrame(test_results)
            df.to_csv(results_dir+f"/{model_name}_{seed}_test_results.csv", index=False)

