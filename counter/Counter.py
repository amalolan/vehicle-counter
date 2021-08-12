from .Cluster import *


class Counter:
    def __init__(self, tracks, grid_size, h_region_factor):
        self.tracks = tracks
        self.grid_size = grid_size
        self.h_region_factor = h_region_factor
        self.clusters = []

    #
    # def remove_small_clusters(self, n, min_cluster_size=3):
    #     height, width, _ = self.tracks.image.shape
    #     self.clusters = [Cluster(self.tracks.get_tracks_by_cluster(i), i, height, width, self.grid_size)
    #                      for i in range(n)]
    #     new_clusters = []
    #     for cluster in self.clusters:
    #         if len(cluster.tracks) >= min_cluster_size:
    #             new_clusters.append(cluster)
    #         else:

    def plot(self, cluster_labels, n, title, plot_path=None, total_n=False):
        plt.clf()
        plt.imshow(self.tracks.image)
        print(title)
        clusters, counts = np.unique(cluster_labels, return_counts=True)
        if total_n:
            actual_clusters = [i for i in range(n)]
            actual_counts = []
            decrement = 0
            for i in range(n):
                if i not in clusters:
                    actual_counts.append(0)
                    decrement += 1
                else:
                    actual_counts.append(counts[i - decrement])
            print(clusters)
            print(counts)
            print(actual_clusters)
            print(actual_counts)
            clusters = actual_clusters
            counts = actual_counts
        [cluster.plot() for cluster in self.clusters]
        patches = [mpatches.Patch(color="C" + str(clusters[i]), label=str(clusters[i]) + " " + str(counts[i]))
                   for i in range(n)]
        plt.legend(handles=patches)
        if plot_path:
            plt.savefig(plot_path)
            plt.clf()
        else:
            plt.show()

    def cluster(self, h_min_cluster_size, h_percent_min_lines, h_min_lines, h_min_paths,
                min_n=None, max_n=None, fixed_n=None, plot_path=None, debug=False):
        height, width, _ = self.tracks.image.shape
        self.tracks.remove_small_paths()
        if len(self.tracks.tracks_list) == 0:
            return 0
        X = self.tracks.normalize()
        max_silhouette, best_n = 0, None
        if fixed_n is None:
            max_n = min(max_n, len(self.tracks.tracks_list))
            for n in range(min_n, max_n):
                cluster_labels, silhouette_avg = self.tracks.n_cluster(n, X)
                if debug:
                    self.tracks.update_clusters(cluster_labels)
                    self.tracks.plot_quick()
                if silhouette_avg > max_silhouette:
                    max_silhouette = silhouette_avg
                    best_n = n
        else:
            best_n = fixed_n
        if best_n == 1:
            cluster_labels, max_silhouette = [0 for _ in self.tracks.tracks_list], 0
        else:
            cluster_labels, max_silhouette = self.tracks.n_cluster(best_n, X)
        self.tracks.update_clusters(cluster_labels)
        # self.tracks.plot_quick()
        self.clusters = []
        for i in range(best_n):
            cluster_tracks = self.tracks.get_tracks_by_cluster(i)
            if len(cluster_tracks) < h_min_cluster_size:
                self.tracks.remove_tracks(cluster_tracks)
                return self.cluster(h_min_cluster_size, h_percent_min_lines,
                                    h_min_lines, h_min_paths, fixed_n=best_n - 1)
            self.clusters.append(
                Cluster(cluster_tracks, i, height, width, self.grid_size, self.h_region_factor,
                        h_percent_min_lines, h_min_lines, h_min_paths))

        self.plot(cluster_labels, best_n, "n: " + str(best_n) + " avg: " + str(max_silhouette), plot_path)
        return best_n

    def post_process(self):
        # Post-processing
        self.tracks.remove_clustered_anomaly(self.clusters, self.grid_size * self.h_region_factor)

    def update_track_clusters(self, computed_clusters, plot_path=None, debug=False, total_n=False):
        self.clusters = []
        path_cluster_values = []
        for computed_cluster in computed_clusters:
            new_cluster = Cluster([], computed_cluster.number, computed_cluster.height,
                                  computed_cluster.width, computed_cluster.grid_size,
                                  computed_cluster.h_region_factor)
            new_cluster.rep_path = Track(np.array(computed_cluster.rep_path),
                                         np.array([0]), 0, 0, cls="rep")
            if len(new_cluster.rep_path.coords) != 0:
                self.clusters.append(new_cluster)
        n = len(self.clusters)
        if n == 0:
            self.plot([], 0, "Test", plot_path, total_n)
            return
        for i in range(n):
            self.clusters[i].number = i
            path_cluster_values.append(self.clusters[i].rep_path.get_cluster_value())
        cluster_labels = []
        self.tracks.remove_small_paths()
        X = self.tracks.normalize(path_cluster_values)
        for j in range(len(self.tracks.tracks_list)):
            track = self.tracks.tracks_list[j]
            # self.tracks.tracks_list[j].cluster = 4
            if debug:
                self.tracks.plot_quick([track])
            track.cluster = min([i for i in range(n)],
                                key=lambda x: np.linalg.norm(X[(-n + x), :] -
                                                             X[j, :]))
            if debug:
                self.tracks.plot_quick([track])
            self.clusters[track.cluster].tracks.append(track)
            cluster_labels.append(track.cluster)

        self.tracks.plot_quick(self.tracks.tracks_list)
        self.post_process()
        self.plot(cluster_labels, n, "Test", plot_path, total_n)

