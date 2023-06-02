import numpy as np


def normalize_img(img):
    return img / img.max()


class CorrelationMatrixMounter:
    def __init__(self, img, gamma):
        self.img = img[:, :, 0]
        self.gamma = gamma

        self.max_y, self.max_x, _ = img.shape
        self.normalized_img = normalize_img(self.img)

        # self.mean_dist_pix = self.calc_mean_dist_pix()
        # self.mean_dist_coord = self.calc_mean_dist_coord()
        self.correlation_matrix = self.calc_correlation_matrix()

    def calc_correlation_matrix(self):
        max_prod = self.max_x * self.max_y
        correlation_matrix = np.zeros((max_prod, max_prod), dtype=np.float32)
        i, j = 0, 0
        for i_coord, i_pix in np.ndenumerate(self.normalized_img):
            i = 0
            for j_coord, j_pix in np.ndenumerate(self.normalized_img):
                correlation_matrix[i, j] = \
                    self.gamma * self.coord_correlation(i_coord, j_coord) + \
                    (1 - self.gamma) * self.pix_correlation(i_pix, j_pix)
            # abs(j_pix - i_pix)
                i += 1
            j += 1

        return correlation_matrix

    def pix_correlation(self, pix_i, pix_j):
        corr = 1 / (1 + np.exp(- np.linalg.norm(pix_i - pix_j) ** 2))  # / self.mean_dist_pix[i]))
        return corr

    def coord_correlation(self, i, j):
        i_hat = np.array([i[0] / self.max_y, i[1] / self.max_x])
        j_hat = np.array([j[0] / self.max_y, j[1] / self.max_x])
        corr = np.exp(-np.linalg.norm(i_hat - j_hat) ** 2)  # / self.mean_dist_coord[i])

        return corr

    def calc_mean_dist_pix(self):
        mean_dist_pix = np.zeros_like(self.img, dtype=np.float32)
        for r in range(self.img.shape[0]):
            for c in range(self.img.shape[1]):
                mean_dist_pix[r, c] = (self.normalized_img[r, c] - self.normalized_img).__abs__().mean()

        return mean_dist_pix

    def calc_mean_dist_coord(self):
        mean_dist_coord = np.zeros_like(self.img, dtype=np.float32)
        for r in range(self.img.shape[0]):
            for c in range(self.img.shape[1]):
                for rr in range(self.img.shape[0]):
                    for cc in range(self.img.shape[1]):
                        mean_dist_coord[r, c] += \
                            np.linalg.norm(((r - rr) / self.max_y), ((c - cc) / self.max_x))

        mean_dist_coord /= mean_dist_coord.size

        return mean_dist_coord
