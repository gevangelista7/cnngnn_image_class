import numpy as np
import pandas as pd
from datasets import load_dataset
from timm import create_model
from utils import fit_and_pad_map, datasets_dir, models_dir, results_dir, classification_dir
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report

from torchvision import transforms, datasets

# seeds = [3, 5, 11, 19, 22, 28, 1947, 1989, 1990, 2017]
seeds = [3]

if __name__ == '__main__':
    img_treat_dir = classification_dir+"/samples_original"
    dataset = load_dataset("imagefolder",
                           task="image-classification",
                           data_dir=img_treat_dir+"/Test_1")

    dataset = dataset.map(fit_and_pad_map)
    dataset.set_format("torch")

    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        # k-fold
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Begin of folding")
        ds_split_train_test = dataset['train'].train_test_split(test_size=0.1, seed=seed)
        train_ds, test_ds = ds_split_train_test["train"], ds_split_train_test["test"]

        ds_split_train_val = train_ds.train_test_split(test_size=0.1 / 0.9, seed=seed)
        train_ds, val_ds = ds_split_train_val["train"], ds_split_train_val["test"]

        train_ds.save_to_disk(dataset_path=img_treat_dir+f"/unified_train_{seed}")
        # test_ds.save_to_disk(dataset_path=img_treat_dir+f"/unified_test_{seed}")
        # val_ds.save_to_disk(dataset_path=img_treat_dir+f"/unified_val{seed}")
