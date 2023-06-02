import numpy as np


class Img2Point:
    def __init__(self, img, threshold=5, connected_neighbors_half=3):
        self.np_img = np.array(img)[:, :, 0]
        self.threshold = threshold

        self.shape = self.np_img.shape

        self.x_pos = self.create_x_pos_mat()
        self.y_pos = self.create_y_pos_mat()
        self.points = self.create_points()
        self.edges = self.create_edges(connected_neighbors_half)

    def create_x_pos_mat(self):
        max_y, max_x = self.shape
        row = np.arange(max_x) / max_x
        x_pos_mat = np.tile(row, (max_y, 1))

        return x_pos_mat

    def create_y_pos_mat(self):
        max_y, max_x = self.shape
        column = np.arange(max_y) / max_y
        y_pos_mat = np.tile(column, (max_x, 1)).T

        return y_pos_mat

    def create_points(self):
        normalized_img = self.np_img / np.max(self.np_img)
        parametrized_img = np.stack((normalized_img, self.x_pos, self.y_pos), axis=-1)
        points = parametrized_img.T.reshape((-1, 3))
        points = points[points[:, 0] > self.threshold / np.max(self.np_img)]
        sortidx = np.argsort(np.linalg.norm(points, axis=1))
        points = points[sortidx]

        return points

    def create_edges(self, neighbors):
        n = len(self.points)
        edges = np.zeros((n, n))
        i, j = np.indices(edges.shape)
        for delta in range(1, neighbors+1):
            edges[i == j + delta] = 1
            edges[i == j - delta] = 1

        return edges

    def get_graph(self):
        return self.points, self.edges
