import os
import imageio
from math import atan2, sqrt, fabs, inf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.neighbors import LocalOutlierFactor
from rdp import rdp
import time


class Track:
    i = 0

    def __init__(self, coords, width, height, cls="car"):
        self.coords = coords
        self.cls = cls
        self.width = width
        self.height = height
        self.cluster = None
        self.id = Track.i
        Track.i += 1

    def rotate(self, rotation_matrix):
        self.coords = np.array(np.matmul(rotation_matrix, self.coords.T).T)

    def get_cluster_value(self):
        first = self.coords[0]
        last = self.coords[-1]
        # distance = np.linalg.norm(last - first)
        dist = np.array([last[0] - first[0], last[1] - first[1]])
        angle = atan2(dist[1], dist[0])
        return np.hstack([first, last, dist, angle])

    def get_displacement(self):
        return np.linalg.norm(self.coords[-1] - self.coords[0])

    def get_distance(self):
        dist = 0
        for i in range(len(self.coords) - 1):
            dist += fabs(np.linalg.norm(self.coords[i + 1] - self.coords[i]))
        return dist

    def plot_vals(self, x_vals, y_vals, markersize):
        if self.cluster is None:
            color = "black"
        else:
            color = "C" + str(self.cluster)
        plt.plot(x_vals, y_vals, c=color, label=self.cluster)
        if markersize != 0:
            plt.plot(x_vals[0], y_vals[0], c=color, marker='o', markersize=markersize)
            plt.plot(x_vals[1], y_vals[1], c=color, marker='o', markersize=markersize)

    def plot_quick(self):
        x_vals = [self.coords[0][0], self.coords[-1][0]]
        y_vals = [self.coords[0][1], self.coords[-1][1]]
        self.plot_vals(x_vals, y_vals, 0)

    def plot(self, markersize=0):
        print(self.id)
        for i in range(len(self.coords) - 1):
            x_vals = [self.coords[i][0], self.coords[i + 1][0]]
            y_vals = [self.coords[i][1], self.coords[i + 1][1]]
            self.plot_vals(x_vals, y_vals, markersize)

    def is_matched(self, path, radius):
        for track_coord in self.coords:
            flag = False
            for coord in path:
                if fabs(np.linalg.norm(coord - track_coord)) < radius:
                    flag = True
                    break
            if not flag:
                return False
        return True


class Tracks:

    def __init__(self,h_angle_factor, image=None):
        self.tracks_list = []
        self.image = image
        self.h_angle_factor = h_angle_factor

    def append(self, track):
        self.tracks_list.append(track)

    def get_track_by_id(self, track_id):
        for track in self.tracks_list:
            if track.id == track_id:
                return track
        return None

    def remove_small_paths(self):
        new_tracks_list = []
        for track in self.tracks_list:
            size = max(track.width, track.height)
            min_detections = track.get_displacement() / size
            if min_detections > 1:
                new_tracks_list.append(track)
        self.tracks_list = new_tracks_list

    def remove_clustered_anomaly(self, clusters, radius):
        new_tracks_list = []
        # plt.imshow(self.image)
        for track in self.tracks_list:
            # size = max(track.width, track.height)
            # plt.imshow(self.image)
            # track.plot(2)
            cluster = clusters[track.cluster]
            # cluster.plot(5)
            if track.is_matched(cluster.rep_path, radius):
                # plt.savefig("data/test_figs/matched/"+str(track.id)+".png")
                new_tracks_list.append(track)
            # else:
            #     # track.plot(2)
            #     # cluster.plot(20, c='black')
            #     # plt.savefig("data/test_figs/removed/"+str(track.id)+".png")
            #
            #     print(track.id)
        self.tracks_list = new_tracks_list
        # plt.show()

    def remove_tracks(self, tracks_list):
        for track in tracks_list:
            self.tracks_list.remove(track)

    def normalize(self):
        cluster_values = np.array([track.get_cluster_value() for track in self.tracks_list])
        std_scaler = StandardScaler()
        normalized = std_scaler.fit_transform(cluster_values)
        normalized[:, -1] *= self.h_angle_factor
        return normalized


    def n_cluster(self, n, X):
        clusterer = KMeans(n_clusters=n, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print("n: " + str(n) + " score: " + str(silhouette_avg))
        return cluster_labels, silhouette_avg

    def update_clusters(self, cluster_labels):
        for i in range(len(self.tracks_list)):
            self.tracks_list[i].cluster = cluster_labels[i]

    def get_cluster_lengths(self, cluster_labels):
        lengths = [track.get_displacement() for track in self.tracks_list]
        df = pd.DataFrame({'cluster': cluster_labels, 'lengths': lengths})
        return df.groupby(['cluster'])['lengths'].mean()

    def get_tracks_by_cluster(self, cluster):
        tracks = []
        for i in self.tracks_list:
            if i.cluster == cluster:
                tracks.append(i)
        return tracks

    def plot_subset(self, tracks, markersize=5):
        plt.imshow(self.image)
        for track in tracks:
            track.plot(markersize)
        plt.show()

    def plot_quick(self, tracks=None):
        if tracks is None:
            tracks = self.tracks_list
        plt.imshow(self.image)
        for track in tracks:
            track.plot_quick()
        plt.show()

    def plot_track(self, track, markersize=10):
        plt.imshow(self.image)
        track.plot(markersize)
        plt.show()

    def plot_id(self, track_id, markersize=5):
        track = self.get_track_by_id(track_id)
        self.plot_track(track, markersize)

    def plot(self):
        self.plot_subset(self.tracks_list, 1)

    def plot_longest(self, n):
        lengths = [None for _ in range(n)]
        for track in self.tracks_list:
            if lengths[track.cluster] is None or \
                    track.get_distance() > lengths[track.cluster].get_distance():
                lengths[track.cluster] = track
        self.plot_subset(lengths)

    def plot_average_length(self, counts):
        n = len(counts)
        means = [0 for _ in range(n)]
        for track in self.tracks_list:
            means[track.cluster] += track.get_distance()
        means = [means[i] / counts[i] for i in range(n)]
        lengths = [None for _ in range(n)]
        for track in self.tracks_list:
            if lengths[track.cluster] is None or \
                    np.sqrt((track.get_distance() - means[track.cluster]) ** 2) \
                    < np.sqrt((lengths[track.cluster].get_distance() - means[track.cluster]) ** 2):
                lengths[track.cluster] = track
        self.plot_subset(lengths)

    def plot_coords(self, coords, markersize=5):
        plt.imshow(self.image)
        x_vals, y_vals = zip(*coords)
        plt.plot(x_vals, y_vals, marker='o', markersize=markersize)
        plt.show()
