from .Track import *
from .QuadTree import *


class Cluster:

    def __init__(self, tracks, cluster_number, height, width, grid_size, h_region_factor,
                 h_percent_min_lines=None, h_min_lines=None, h_min_path=None):
        self.number = cluster_number
        self.tracks = tracks
        self.height = height
        self.width = width
        self.qtree = None
        self.grid_size = grid_size
        self.h_region_factor = h_region_factor
        if len(self.tracks) != 0:
            self.rep_path = self.find_rep_path(self.grid_size, h_percent_min_lines, h_min_lines, h_min_path)
        else:
            self.rep_path = []

    def get_num_tracks_for_frame(self, frame):
        count = 0
        for track in self.tracks:
            if track.is_in_frame(frame):
                count += 1
        return count

    def get_avg_confidence(self, frame=None):
        avg_confidence = 0
        count = 0
        for track in self.tracks:
            if frame is None or track.is_in_frame(frame):
                avg_confidence += track.confidence
                count += 1
        return avg_confidence/count if count != 0 else 0


    @staticmethod
    def find_rotation_matrix(segments):
        n_vectors = segments.shape[0]
        vectors = np.diff(segments, axis=1).reshape((n_vectors, 2))
        avg_vector = np.mean(vectors, axis=0)
        unit_avg_vector = avg_vector / np.linalg.norm(avg_vector)
        phi = np.arccos(np.clip(np.dot(unit_avg_vector, [1, 0]), -1.0, 1.0))
        return np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

    def get_all_segments(self):
        all_segments = []
        n_segments = 0
        for track in self.tracks:
            segments = []
            for i in range(len(track.coords) - 1):
                segments.append(np.array([track.coords[i], track.coords[i + 1]]))
            all_segments.append(segments)
            n_segments += len(segments)
        return np.concatenate(all_segments, axis=0).reshape((n_segments, 2, 2))

    def rotate_and_insert(self, rotation_matrix):
        center = np.array(np.matmul(rotation_matrix, [self.width / 2, self.height / 2])).T
        domain = Rect(*center, self.width, self.height)
        self.qtree = QuadTree(domain, 3)
        min_coord = [inf, inf]
        max_coord = [-inf, -inf]
        for track in self.tracks:
            rotated_coords = np.array(np.matmul(rotation_matrix, track.coords.T)).T
            for i in range(len(rotated_coords) - 1):
                start, end = rotated_coords[i], rotated_coords[i + 1]
                if start[0] < min_coord[0]:
                    min_coord = start
                if end[0] > max_coord[0]:
                    max_coord = end
                self.qtree.insert(Point(*start, end))
        return min_coord[0], max_coord[0], center

    def test_plot(self, points=()):
        video_reader = imageio.get_reader("data/cam_1.mp4")
        image = video_reader.get_data(0)
        video_reader.close()
        print('Number of points in the domain =', len(self.qtree))
        # plt.imshow(image)
        ax = plt.subplot()
        self.qtree.draw(ax)
        return ax

    def find_rep_path(self, gamma, h_percent_min_lines, h_min_lines, h_min_paths):
        min_lines = max(len(self.tracks) * h_percent_min_lines, h_min_lines)
        path_1, path_2 = [], []
        all_segments = self.get_all_segments()
        rotation_matrix = Cluster.find_rotation_matrix(all_segments)
        inverse = np.linalg.inv(rotation_matrix)
        sweep, last, center = self.rotate_and_insert(rotation_matrix)
        # ax = self.test_plot()
        # rotated_segments = np.array(np.matmul(rotation_matrix, all_segments.reshape((-1, 2)).T).T).reshape((-1, 2, 2))
        # sorted_segments = rotated_segments[rotated_segments[:, 0, 0].argsort()]
        # prev_1, prev_2 = None, None
        while sweep <= last:
            # ax = self.test_plot()
            num_p_1, num_p_2, avg_y_1, avg_y_2 = 0, 0, 0, 0
            search_region = Rect(sweep, center[1], self.grid_size * self.h_region_factor, self.height)
            # search_region.draw(ax, c='r')
            # plt.show()
            found_points = []
            self.qtree.query(search_region, found_points)
            increment = 5
            for point_obj in found_points:
                point = np.array([[point_obj.x, point_obj.y], point_obj.payload])
                if point[0][0] <= sweep < point[1][0]:
                    num_p_1 += 1
                    m = (point[1][1] - point[0][1]) / (point[1][0] - point[0][0])
                    b = point[0][1] - m * point[0][0]
                    avg_y_1 += m * sweep + b
                if point[1][0] <= sweep < point[0][0]:
                    num_p_2 += 1
                    m = (point[1][1] - point[0][1]) / (point[1][0] - point[0][0])
                    b = point[0][1] - m * point[0][0]
                    avg_y_2 += m * sweep + b
            if num_p_1 >= min_lines:
                increment = gamma
                # if prev_1 is not None:
                #     dist = fabs(start[0] - prev_1[0])
                #     if dist <= gamma:
                #         continue
                # prev_1 = start
                avg_y = avg_y_1 / num_p_1
                real_point = np.array(np.matmul(inverse, [sweep, avg_y]))
                path_1.append(real_point)
            if num_p_2 >= min_lines:
                increment = gamma
                # if prev_2 is not None:
                #     dist = fabs(start[0] - prev_2[0])
                #     if dist <= gamma:
                #         continue
                # prev_2 = start
                avg_y = avg_y_2 / num_p_2
                real_point = np.array(np.matmul(inverse, [sweep, avg_y]))
                path_2.append(real_point)
            sweep += increment
        if len(path_2) < h_min_paths:
            path_2 = []
        if len(path_1) < h_min_paths:
            path_1 = []
        paths = np.array(path_1 + path_2[::-1])
        smooth_path = []
        for point in paths:
            flag = True
            for other_point in paths:
                if not np.array_equal(point, other_point) and np.linalg.norm(other_point - point) < gamma:
                    flag = False
            if flag:
                smooth_path.append(point)
        return smooth_path

    def plot(self, s=15, c=None):
        if type(self.rep_path) == Track:
            x, y = np.hsplit(self.rep_path.coords, 2)
            color = "C" + str(self.number)
            if c is not None:
                color = c
            plt.plot(x, y, color=color, markersize=s / 2)
        elif len(self.rep_path) > 0:
            x, y = zip(*self.rep_path)
            color = "C" + str(self.number)
            if c is not None:
                color = c
            plt.plot(x, y, color=color, markersize=s / 2)
            # plt.scatter(x, y, color=color, s=s)
