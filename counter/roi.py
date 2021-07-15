import numpy as np
import imageio
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from collections import Counter


def findGrid(detectionsFile, video_file_path,
             n_frames_percent, quantile=0.5):
    # Read the detections csv.
    df = pd.read_csv(detectionsFile,
                     dtype={'frame': int, 'x': int, 'y': int, 'w': int,
                            'h': int, 'confidence': float})
    n_frames = df['frame'].max()
    df = df[df['frame'] <= int(n_frames * n_frames_percent)]

    video_reader = imageio.get_reader(video_file_path)
    image = video_reader.get_data(0)
    height, width, channels = image.shape

    # max of quantiles in width or height
    box_size = int(max(df.quantile(quantile)['w'], df.quantile(quantile)['h']))
    # Create x,y coords for all the boxes (top-left corner coord).
    points = {}
    for i in range(0, width, box_size):
        for j in range(0, height, box_size):
            points[(i, j)] = []

    # For ever point, add it to its respective box.
    for index, row in df.iterrows():
        box_coord = (int(row['x'] // box_size) * box_size, int(row['y'] // box_size) * box_size)
        points[box_coord].append(row['confidence'])

    grid = {}
    # Now for every box, draw a rectangle with color corresponding to
    # the average YOLO detection confidence at that box.
    for point in points:
        avg = 0
        if len(points[point]):
            avg = sum(points[point]) / len(points[point])
        grid[point] = avg
    return grid, box_size, n_frames


def explore(grid, point, box_size, threshold, explored, cluster):
    explored[point] = cluster
    neighbors = {(point[0] - box_size, point[1] - box_size),
                 (point[0] - box_size, point[1]),
                 (point[0] - box_size, point[1] + box_size),
                 (point[0], point[1] - box_size),
                 (point[0], point[1]),
                 (point[0], point[1] + box_size),
                 (point[0] + box_size, point[1] - box_size),
                 (point[0] + box_size, point[1]),
                 (point[0] + box_size, point[1] + box_size), }
    for neighbor in neighbors:
        if neighbor in grid:
            if grid[neighbor] >= threshold and neighbor not in explored:
                explore(grid, neighbor, box_size, threshold, explored, cluster)


def dfs(grid, box_size, h_confidence_threshold=0.75):
    explored = dict()
    cluster = 1
    for point in grid:
        if grid[point] >= h_confidence_threshold and not point in explored:
            explore(grid, point, box_size, h_confidence_threshold, explored, cluster)
            cluster += 1
    return explored


def removeOutlierClusters(grid, box_size, clusters, h_outlier_threshold=0.25):
    sizes = Counter(clusters.values())
    avg = sum(sizes.values()) / len(sizes)
    print("Average cluster size: ", avg)
    for point in clusters:
        if sizes[clusters[point]] <= h_outlier_threshold * avg:
            clusters[point] = 0
    return clusters


def getPointsFromClusters(clusters, box_size):
    points = []
    for point in clusters:
        if clusters[point] != 0:
            points.append((point[0], point[1]))
            points.append((point[0] + box_size, point[1]))
            points.append((point[0], point[1] + box_size))
            points.append((point[0] + box_size, point[1] + box_size))
    return points


# plot the clusters and the convex hull (ROI) determined, and save it in results/hull.txt
def plotConvexHull(grid, box_size, clusters, video_file_path,
                   output_hull="results/hull.txt", output_image="results/hull.png",
                   cams_file="results/cams.txt"):
    video_reader = imageio.get_reader(video_file_path)
    image = video_reader.get_data(0)
    height, width, channels = image.shape

    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    print("YOLO Detection Confidence Clusters for box sizes of " + str(box_size) + " x " + str(box_size))
    implot = plt.imshow(image, cmap='viridis')
    cmap = plt.get_cmap('viridis')

    for point in grid:
        rect = patches.Rectangle(point, box_size, box_size, linewidth=1,
                                 facecolor=cmap(grid[point]), edgecolor='black', alpha=0.3)
        ax.add_patch(rect)
        if point in clusters:
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax.annotate(str(clusters[point]), (cx, cy), color='w',
                        fontsize=7, ha='center', va='center', weight="bold")
    points = np.array(getPointsFromClusters(clusters, box_size))
    hull = ConvexHull(points)
    hull_vertices = np.array([points[i] for i in hull.vertices])
    np.savetxt(output_hull, hull_vertices,
               fmt='%d', delimiter=',')
    print(hull_vertices)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k-', color="white")

    # Finally, append the color bar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(implot, cax=cax)
    cbar.set_ticks([0, 255 * 0.25, 255 * 0.5, 255 * 0.75, 255])
    cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])
    plt.savefig(output_image)
    return hull_vertices


def find_roi(video_file_path, detections_path, image_output_path,
             hull_output_path, h_confidence_threshold, h_outlier_threshold,
             n_frames_percent=1):
    # detections_csv is the output of running yolo on the video, so it contains all
    # detected objects and their confidences.
    grid, box_size, n_frames = findGrid(detections_path, video_file_path, n_frames_percent)
    clusters = dfs(grid, box_size, h_confidence_threshold=h_confidence_threshold)
    clusters = removeOutlierClusters(grid, box_size, clusters, h_outlier_threshold=h_outlier_threshold)
    hull_vertices = plotConvexHull(grid, box_size, clusters,
                                   output_hull=hull_output_path,
                                   output_image=image_output_path,
                                   video_file_path=video_file_path)
    # results will be saved in results/hull.txt
    return n_frames, box_size, hull_vertices
