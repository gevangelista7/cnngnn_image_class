import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from correlation_coord import CorrelationMatrixMounter
from img2point import Img2Point
import matplotlib.pyplot as plt
import skimage
import time
import networkx as nx
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from wavelet_based_superpixels import wavelet_superpixel_singlechannel, wavelet_superpixel


if __name__ == '__main__':
    img = Image.open('00001.bmp')
    img = ImageEnhance.Sharpness(img).enhance(2)
    img = ImageEnhance.Brightness(img).enhance(2)
    img = ImageEnhance.Contrast(img).enhance(2)
    ImageDraw.Draw(img).rectangle([(324, 1307), (394, 1402)], outline="yellow")     # ball
    ImageDraw.Draw(img).rectangle([(424, 1478), (564, 1596)], outline="yellow")  # circle cage
    img.show()

    circle_cage = img.crop((424, 1478, 564, 1596))
    ball = img.crop((324, 1307, 394, 1402))

    segments_slic = skimage.segmentation.slic(np.array(ball))


    new_w = 50
    original_size = img.size
    new_size = new_w, int(new_w*img.height/img.width)
    # img_red = img.resize(new_size)
    # img = np.asarray(img)


    # start = time.time()
    # segments_slic = skimage.segmentation.slic(img)
    # segments_quick = wavelet_superpixel_singlechannel(img)
    # end = time.time()
    #
    # print('elapsed time:', end-start)
    #
    # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    # # segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,
    # #                      start_label=1)
    # # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    # gradient = sobel(rgb2gray(img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    #
    # print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
    # print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
    # print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
    # print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')
    #
    # fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
    #
    # ax[0, 0].imshow(mark_boundaries(img, segments_fz))
    # ax[0, 0].set_title("Felzenszwalbs's method")
    # ax[0, 1].imshow(mark_boundaries(img, segments_slic))
    # ax[0, 1].set_title('SLIC')
    # ax[1, 0].imshow(mark_boundaries(img, segments_quick))
    # ax[1, 0].set_title('Quickshift')
    # ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
    # ax[1, 1].set_title('Compact watershed')
    #
    # for a in ax.ravel():
    #     a.set_axis_off()
    # #
    # plt.tight_layout()
    # plt.show()

    # new_w = 100
    # original_size = img.size
    # new_size = new_w, int(new_w*img.height/img.width)
    # img_red = img.resize(new_size)
    # np_img = np.array(img_red)
    #
    # start = time.time()
    # points, edges = Img2Point(np_img, threshold=10).get_graph()
    #
    # end = time.time()
    # print('elapsed time: {}'.format(end-start))
    #
    # G = nx.from_numpy_array(edges)
    # nx.set_node_attributes(G, pd.DataFrame(points.T))
    #
    # end = time.time()
    # print('elapsed time: {}'.format(end-start))
    #
    # nx.draw(G, pos=points[:, 1:]@np.array([[1900, 0], [0, 1000]]).T)
    # plt.show()
    # end = time.time()
    # print('elapsed time: {}'.format(end - start))

    # # np_img.shape
    #
    # # plt.hist(np_img.flatten(), bins=np.arange(255), log=True)
    # # plt.xlim([0,255])
    # # plt.show()
    #
    # valid_mask = np.where(np_img > 5, np_img, 0)
    # print(np.count_nonzero(valid_mask) / valid_mask.size)
    # print(valid_mask.shape)
    #
    # plt.hist(valid_mask.flatten(), bins=np.arange(255), log=True)
    # plt.xlim([1, 255])
    # plt.show()
    #
    # print(valid_mask.shape)
    #
    # Image.fromarray(valid_mask).show()
    #
    # # # print(np.array(img).count(0))

    # # def mount_graph(img):


    # new_w = 100
    # original_size = img.size
    # new_size = new_w, int(new_w*img.height/img.width)
    # img_red = img.resize(new_size)
    # img.show()
    #
    # for scale in (0.1, 1, 10, 100):
    #     start = time.time()
    #     im_mask = skimage.segmentation.felzenszwalb(skimage.util.img_as_float(img_red), sigma=1, scale=scale)
    #     end = time.time()
    #
    #     print('tempo: ', end-start)
    #
    #     segmented = Image.fromarray((im_mask*255/im_mask.max()).astype(np.uint8))
    #     seg_resize = segmented.resize(original_size)
    #     seg_resize.show()
    #     seg_resize.save("segmented_scale={}.png".format(scale))
    #
    # # new_w = 50
    # # new_size = new_w, int(new_w*img.height/img.width)
    # # img = img.resize(new_size)
    # #
    # # img = np.asarray(img)
    # #
    # # start = time.time()
    # # W = CorrelationMatrixMounter(img, 0.5).correlation_matrix
    # # end = time.time()
    # #
    # # print(W.shape)
    # # print(W)
    # # print('time = ', end - start, 's')
