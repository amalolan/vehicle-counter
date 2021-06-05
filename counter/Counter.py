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

    def plot(self, cluster_labels, n, title, plot_path=None):
        plt.clf()
        plt.imshow(self.tracks.image)
        print(title)
        clusters, counts = np.unique(cluster_labels, return_counts=True)
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
                min_n=None, max_n=None, fixed_n=None, plot_path=None):
        height, width, _ = self.tracks.image.shape
        self.tracks.remove_small_paths()
        X = self.tracks.normalize()
        max_silhouette, best_n = 0, None
        if fixed_n is None:
            for n in range(min_n, max_n):
                cluster_labels, silhouette_avg = self.tracks.n_cluster(n, X)
                if silhouette_avg > max_silhouette:
                    max_silhouette = silhouette_avg
                    best_n = n
        else:
            best_n = fixed_n
        cluster_labels, max_silhouette = self.tracks.n_cluster(best_n, X)
        self.tracks.update_clusters(cluster_labels)
        # self.tracks.plot_quick()
        self.clusters = []
        for i in range(best_n):
            cluster_tracks = self.tracks.get_tracks_by_cluster(i)
            if len(cluster_tracks) < h_min_cluster_size:
                self.tracks.remove_tracks(cluster_tracks)
                self.cluster(h_min_cluster_size, h_percent_min_lines, h_min_lines, h_min_paths, fixed_n=best_n - 1)
                return best_n - 1
            self.clusters.append(
                Cluster(cluster_tracks, i, height, width, self.grid_size, self.h_region_factor,
                        h_percent_min_lines, h_min_lines, h_min_paths))

        self.plot(cluster_labels, best_n, "n: " + str(best_n) + " avg: " + str(max_silhouette), plot_path)
        return best_n

    def post_process(self):
        # Post-processing
        self.tracks.remove_clustered_anomaly(self.clusters, self.grid_size * self.h_region_factor)
