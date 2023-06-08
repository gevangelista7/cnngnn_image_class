import os
from PIL import Image
import joblib
from itg_model import ITGProcessor
import sys
sys.path.append('.')
import argparse

if __name__ == '__main__':
    """
    possible values
    img_treat:  
        'original', 'equalize', 'enhance2', 'autocontrast1';
    linking type: 
        'knn_w', 'knn', 'fc';
    superpixels methods:
        'slic', 'pixel_inf, 'vgg'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', default=False, type=bool)
    args = parser.parse_args()

    test_mode = args.test_mode

    total_count = 0
    obj_counter = {}
    ds = []

    root_dir = '../datasets/UATD_graphs' if os.name == 'nt' else '/home/gabriel/thesis/dataset/UATD_graphs'
    classification_dir = './UATD_classification'

    dataset_dir = '../datasets' if os.name == 'nt' else '/home/gabriel/thesis/dataset'

    img_treat = 'autocontrast1'
    linking_type = 'knn_w'
    superpixel_method = 'vgg'
    n_sp = 75

    original_dir = dataset_dir+'/UATD_classification/samples_'+img_treat
    destination_file = dataset_dir+'/UATD_graphs/'+superpixel_method+'/'+img_treat+'_'+linking_type+'_'+str(n_sp)

    c = 0

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

    converter = ITGProcessor()
    graph_list = []

    for root, dirs, files in os.walk(original_dir):
        for file in files:
            if file.endswith(".bmp"):
                img_path = os.path.join(root, file)

                part_dir, label = os.path.split(root)
                part = os.path.split(part_dir)[-1]

                img = Image.open(img_path)
                # img = np.array(img)

                # G = img2graph(img, knn=knn, weighted=weighted, n_sp=n_sp, pixel_dist=True)
                #
                # sample = {
                #     'file': file,
                #     'partition': part,
                #     'label': label,
                #     'graph': G
                # }
                # ds.append(sample)
                graph = converter.img2graph_vgg(
                    img=img,
                    label=label,
                    partition=part
                )
                graph_list.append(graph)

                c += 1
                if c % 100 == 0:
                    print('Atual {}: Part {}, File {}'.format(c, part, file))

            if test_mode and c > 2:
                break


    joblib.dump(graph_list, destination_file)

    if test_mode:
        result = joblib.load(destination_file)
        print(result[0])

    # pd.DataFrame.from_records(ds).to_pickle(destination_file)
    # joblib.dump(pd.DataFrame.from_records(ds), destination_file)
    