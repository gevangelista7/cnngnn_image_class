import os
from PIL import Image
import joblib
from itg_model import ITGProcessor
import sys
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet18
from utils import classification_dir, graphs_dir, models_dir
sys.path.append('.')


if __name__ == '__main__':
    """
    possible values
    img_treat:  
        'original', 'equalize', 'enhance2', 'autocontrast1';
    linking type: 
        'knn_w', 'knn', 'fc';
    superpixels methods:
        'slic', 'pixel_inf, 'vgg', 'resnet' 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', default=False, type=bool)
    args = parser.parse_args()

    test_mode = args.test_mode
    test_mode = False

    total_count = 0
    obj_counter = {}
    ds = []

    img_treat = 'original'
    linking_type = 'knn_w'
    superpixel_method = 'resnet_pretrain'
    # finetuned_model = '/ResNet18/ResNet_best_model_fitpad.pth'      # REMEMBER to put '/' in the beginning (placeholder: None)
    finetuned_model = None
    n_sp = 75
    cut_list = [1, 2, 5, 6, 8]

    original_dir = classification_dir+'/samples_'+img_treat
    cnn_dir = graphs_dir+'/cnn'
    destination_file = cnn_dir+'/'+img_treat+'_'+linking_type+'_'+str(n_sp)
    if test_mode:
        destination_file += 'TEST'

    if linking_type == 'knn' or linking_type == 'knn_w':
        knn = True
    else:
        knn = False

    if linking_type == 'knn_w':
        weighted = True
    else:
        weighted = False

    if not knn and not weighted:
        raise NotImplementedError()

    if finetuned_model:
        resnet18_full = resnet18()
        num_features = resnet18_full.fc.in_features
        resnet18_full.fc = nn.Linear(num_features, 10)
        resnet18_full = resnet18_full.load_state_dict(torch.load(models_dir+finetuned_model))
    else:
        resnet18_full = resnet18(pretrained=True)
    net_name = resnet18_full._get_name()

    for cut in cut_list:
        destination_file_cut = destination_file + '_' + net_name + '18_cut' + str(cut)
        if finetuned_model:
            destination_file_cut += '_ft'
        resnet18_partial = nn.Sequential(*list(resnet18(pretrained=True).children())[:cut])
        converter = ITGProcessor(resnet18_partial)
        graph_list = []
        c = 0

        for root, dirs, files in os.walk(original_dir):
            for file in files:
                if file.endswith(".bmp"):
                    img_path = os.path.join(root, file)

                    part_dir, label = os.path.split(root)
                    part = os.path.split(part_dir)[-1]
                    if part in ['Training_ext', 'Unified']:
                        continue

                    img = Image.open(img_path)

                    graph = converter.img2graph_cnn(
                        img=img,
                        label=label,
                        partition=part
                    )
                    graph_list.append(graph)

                    c += 1
                    if c % 100 == 0:
                        print('Cut: {}, Current {}: Part {}, File {}'.format(cut, c, part, file))

                if test_mode and c > 1:
                    break

        joblib.dump(graph_list, destination_file_cut)

        if test_mode:
            result = joblib.load(destination_file_cut)
            print(result[0])
