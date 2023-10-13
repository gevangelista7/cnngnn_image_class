from PIL import Image
import numpy as np
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


# Load the image
image_path = '../datasets/UATD_classification/samples_autocontrast1/Training/plane/293.bmp'
image = Image.open(image_path)
img = np.array(image)

if __name__ == '__main__':
    segments = slic(img, n_segments=75, slic_zero=True)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # SUPERPOSIÇÃO
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # plt.imshow(mark_boundaries(img, segments, color=[255, 0, 0]))
    # plt.scatter(centers[:, 1], centers[:, 0], c='y', alpha=0.5, linewidths=0.1)
    #
    # for i in range(bneighbors.shape[1]):
    #     y0, x0 = centers[bneighbors[0, i]-1]
    #     y1, x1 = centers[bneighbors[1, i]-1]
    #
    #     l = Line2D([x0, x1], [y0, y1], alpha=0.5)
    #     ax.add_line(l)
    #
    # plt.show()
    #
    # # MASCARAS
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # plt.imshow(mark_boundaries(np.ones_like(img)*255, segments, color=[255, 0, 0]))
    # plt.scatter(centers[:, 1], centers[:, 0], c='y', alpha=0.5, linewidths=0.1)
    #
    # plt.show()

    # GRAFO
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # for i in range(bneighbors.shape[1]):
    #     y0, x0 = centers[bneighbors[0, i]-1]
    #     y1, x1 = centers[bneighbors[1, i]-1]
    #
    #     l = Line2D([x0, x1], [y0, y1], alpha=0.5, zorder=-1)
    #     ax.add_line(l)
    #
    # plt.scatter(centers[:, 1], centers[:, 0], c='orange', alpha=1, linewidths=2, s=150)
    # ax.set_aspect('equal')
    # plt.show()

    # GRAFO GNN
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # for i in range(bneighbors.shape[1]):
    #     y0, x0 = centers[bneighbors[0, i] - 1]
    #     y1, x1 = centers[bneighbors[1, i] - 1]
    #
    #     l = Line2D([x0, x1], [y0, y1], alpha=0.5, zorder=-1)
    #     ax.add_line(l)
    #
    # for X, Y in zip(centers[:, 1], centers[:, 0]):
    #     plt.scatter(X, Y, c=np.random.rand(3,), alpha=1, linewidths=2, s=150)
    # ax.set_aspect('equal')
    # plt.show()

    # Bar graph
    df = pd.read_csv('tabela_comparativo_geral.txt',
                     sep='&',
                     skiprows=1,
                     skipfooter=2,
                     engine='python')
    numerical_columns = df.columns[1:]
    df[numerical_columns] = df[numerical_columns].replace('-', pd.NA)

    # Convert numerical columns to numeric data type
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

    melted_df = pd.melt(df, id_vars=['Unnamed: 0'], value_vars=df.columns,
                         var_name='Metric', value_name='Value')

    custom_palette = sns.color_palette("muted")
    custom_palette = ['red'] + custom_palette

    # Create the bar plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Unnamed: 0', y='Value', hue='Metric', data=melted_df, palette=custom_palette)

    # Customize the plot
    plt.ylabel('Recall')
    plt.xlabel('')
    plt.title('Taxa de Verdadeiro Positivo')
    plt.legend(title='Recall Classes')
    plt.grid()
    resnet_mAR = .897
    plt.plot([-0.5, 3], [0.897, 0.897], color='red', linestyle='--', linewidth=2, label='ResNet18')
    plt.text(3.2, resnet_mAR, f'ResNet-18 mAR: {resnet_mAR:.3f}', color='red', verticalalignment='center')
    # Show the plot
    plt.tight_layout()
    plt.show()

