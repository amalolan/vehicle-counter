from .Track import *


def temp_rep_path(tracks, cluster_number, min_lines=7, gamma=30):
    paths = []
    # cluster_tracks = tracks.get_tracks_by_cluster(cluster_number)
    # all_segments = tracks.get_all_segments(cluster_tracks)
    # rotation_matrix = find_rotation_matrix(all_segments)
    # inverse = np.linalg.inv(rotation_matrix)
    # rotated_segments = np.array(np.matmul(rotation_matrix, all_segments.reshape((-1, 2)).T).T).reshape((-1, 2, 2))
    # sorted_segments = rotated_segments[rotated_segments[:, 0, 0].argsort()]
    # colors = [i / len(sorted_segments) for i in range(len(sorted_segments))]
    # x, y = zip(*sorted_segments[:, 0])
    # plt.set_cmap('rainbow')
    # count = 0
    # prev = None
    # for i in range(len(sorted_segments)):
    #     start, end = sorted_segments[i]
    #     if prev is not None:
    #         dist = fabs(start[0] - prev[0])
    #         if dist <= gamma:
    #             continue
    #     num_p_1, num_p_2, avg_y_1, avg_y_2 = 0, 0, 0, 0
    #     for j in range(0, len(sorted_segments)):
    #         point = sorted_segments[j]
    #         if point[0][0] <= start[0] < point[1][0]:
    #             num_p_1 += 1
    #             m = (point[1][1] - point[0][1]) / (point[1][0] - point[0][0])
    #             b = point[0][1] - m * point[0][0]
    #             avg_y_1 += m * start[0] + b
    #         if point[1][0] <= start[0] < point[0][0]:
    #             num_p_2 += 1
    #             m = (point[1][1] - point[0][1]) / (point[1][0] - point[0][0])
    #             b = point[0][1] - m * point[0][0]
    #             avg_y_2 += m * start[0] + b
    #
    #     if num_p_1 >= min_lines:
    #         # for j in range(0, len(sorted_segments)):
    #         #     point = sorted_segments[j]
    #         #     if point[0][0] <= start[0] < point[1][0]:
    #         #         # plt.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], marker='o')
    #         prev = start
    #         avg_y = avg_y_1 / num_p_1
    #         real_point = np.array(np.matmul(inverse, np.array([[start[0], avg_y]]).T).T).squeeze()
    #         paths.append(real_point)
    #         # plt.plot(start[0], avg_y, marker='o', markersize=10, color="black")
    #         count += 1
    #         if count % 100 == 0:
    #             plt.imshow(tracks.image)
    #             plt.scatter(x, y, c=colors, s=.5)
    #             plt.show()
    #     if num_p_2 >= min_lines:
    #         # for j in range(0, len(sorted_segments)):
    #         #     point = sorted_segments[j]
    #         #     if point[1][0] <= start[0] < point[0][0]:
    #         #         plt.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], marker='o')
    #         avg_y = avg_y_2 / num_p_2
    #         real_point = np.array(np.matmul(inverse, np.array([[start[0], avg_y]]).T).T).squeeze()
    #         paths.append(real_point)
    #         # plt.plot(start[0], avg_y, marker='o', markersize=10, color="grey")
    #         if count % 100 == 0:
    #             plt.imshow(tracks.image)
    #             plt.scatter(x, y, c=colors, s=.5)
    #             plt.show()
    # x, y = zip(*paths)
    # plt.imshow(tracks.image)
    # plt.scatter(x, y)
    # plt.show()
