from counter.Counter import *
from counter.roi import find_roi
from roi_object_tracker import run_detector
from object_tracker import counter_helper
import pandas as pd
import numpy as np
import json
import time
import cv2
import os
import sys

WORKING_DIR = ""
PARENT_DIR = ""


class Module:
    def __init__(self, hyperparams_file, log_file):
        with open(WORKING_DIR + hyperparams_file, "r") as fp:
            self.hyperparams = json.load(fp)
        self.log_file = WORKING_DIR + log_file

    def log_run(self, param_set, outputs):
        with open(self.log_file, 'r') as fp:
            current_logs = json.load(fp)
        current_logs.append({**param_set, **outputs})
        with open(self.log_file, "w+") as fp:
            json.dump(current_logs, fp, indent=4)
        print(json.dumps(outputs, indent=2))

    def is_completed(self, param_set, other_data=None):
        if other_data is None:
            other_data = {}
        with open(self.log_file, "r") as fp:
            current_logs = json.load(fp)
        for log in current_logs:
            combined = {**param_set, **other_data}
            if combined.items() <= log.items():
                return True
        return False

    def get_log(self, data, required_keys):
        with open(self.log_file, "r") as fp:
            current_logs = json.load(fp)
        for log in current_logs:
            if data.items() <= log.items():
                if all(key in log for key in required_keys):
                    return log
        return None


class DetectionModule(Module):
    def run(self, cam_name):
        cam_num = "_".join(cam_name.split("_")[1:])
        print(cam_num)
        for i in range(len(self.hyperparams)):
            param_set = self.hyperparams[i]
            if self.is_completed(param_set, {"cam_num": cam_num, "detections_num": str(i)}):
                continue
            start = time.time()
            output_file = WORKING_DIR + "/outputs/detections/detections_" + cam_name + "_" + str(i) + ".csv"
            run_detector(PARENT_DIR + "/videos/cam_" + cam_num + ".mp4",
                         detections_output=output_file,
                         score_threshold=param_set["confidence"],
                         iou_threshold=param_set["iou"])
            self.log_run(param_set, {
                "cam_num": cam_num,
                "detections_num": str(i),
                "detection_time": time.time() - start,
                "detections_file": output_file
            })


class ROIModule(Module):
    def run(self, cam_name, detections_num="0"):
        cam_num = "_".join(cam_name.split("_")[1:])
        print(cam_num)
        for i in range(len(self.hyperparams)):
            param_set = self.hyperparams[i]
            if self.is_completed(param_set, {"cam_num": cam_num, "detections_num": detections_num,
                                             "roi_num": str(i)}):
                continue
            start = time.time()
            hull_file_prefix = WORKING_DIR + "/hull/hull_" + cam_name + "_" + detections_num + "_" + str(i)
            n_frames, box_size, hull_vertices = find_roi(
                PARENT_DIR + "/videos/cam_" + cam_name + ".mp4",
                WORKING_DIR + "/outputs/detections/detections_" + cam_name + "_" + detections_num + ".csv",
                hull_file_prefix + ".png", hull_file_prefix + ".txt",
                param_set["dfs_confidence"], param_set["outlier_threshold"]
            )
            self.log_run(param_set, {
                "cam_num": cam_num,
                "detections_num": detections_num,
                "roi_num": str(i),
                "n_frames": n_frames,
                "grid_size": box_size,
                "detection_time": time.time() - start,
                "hull_file": hull_file_prefix + ".txt"
            })


class TrackModule(Module):
    def run(self, cam_name, detections_num="0", roi_num="0"):
        cam_num = "_".join(cam_name.split("_")[1:])
        video_path = PARENT_DIR + "/videos/cam_" + cam_num + ".mp4"
        print(cam_num)
        for i in range(len(self.hyperparams)):
            param_set = self.hyperparams[i]
            if self.is_completed(param_set, {"cam_num": cam_num, "detections_num": detections_num,
                                             "roi_num": roi_num, "tracks_num": str(i)}):
                continue
            start = time.time()
            # frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
            cam_prefix = cam_name + "_" + detections_num + "_" + roi_num
            tracks_file = WORKING_DIR + "/outputs/tracks/tracks_" + cam_prefix + "_" + str(i) + ".csv"
            try:
                counter_helper([video_path,
                                PARENT_DIR + "/tracked_videos/tracked_" + cam_prefix + "_" + str(i) + ".avi",
                                str(param_set["score_threshold"]),
                                tracks_file,
                                WORKING_DIR + "/outputs/hull/hull_" + cam_prefix + ".txt",
                                str(param_set["max_iou_distance"]),
                                str(param_set["max_age"]),
                                str(param_set["n_init"]),
                                WORKING_DIR + "/outputs/detections/detections_" +
                                cam_name + "_" + detections_num + ".csv",
                                "dont_show"])
            except:
                print("Video ended or ERROR!!")
            self.log_run(param_set, {
                "cam_num": cam_num,
                "detections_num": detections_num,
                "roi_num": roi_num,
                "tracks_num": str(i),
                "tracks_file": tracks_file,
                "tracks_time": time.time() - start
            })
            # break  # TODO: REMOVE if running for all tracks


class CountModule(Module):
    def __init__(self, hyperparams_file, log_file, min_n=2, max_n=15):
        super().__init__(hyperparams_file, log_file)
        self.min_n = min_n
        self.max_n = max_n

    def read_tracks(self, cam_name, tracks_file, h_angle_factor):
        df = pd.read_csv(tracks_file)
        df.sort_values(by=['track', 'frame'], inplace=True)
        gb = df.groupby(["track"])
        video_reader = imageio.get_reader(PARENT_DIR + "/videos/" + cam_name + ".mp4")
        image = video_reader.get_data(0)
        video_reader.close()
        tracks = Tracks(image)
        for x in gb.groups:
            track_df = gb.get_group(x)
            coords = np.c_[track_df['x'].to_numpy(), track_df['y'].to_numpy()]
            track = Track(coords,
                          track_df['width'].mean(),
                          track_df['height'].mean(),
                          h_angle_factor,
                          cls=track_df.iloc[0]['class']
                          )
            tracks.append(track)
        return tracks

    def run(self, cam_name, grid_size, detections_num="0", roi_num="0", tracks_num="0"):
        cam_num = "_".join(cam_name.split("_")[1:])
        os.makedirs(WORKING_DIR + "/outputs/counts/" + cam_name, exist_ok=True)
        for i in range(len(self.hyperparams)):
            param_set = self.hyperparams[i]
            if self.is_completed(param_set, {"cam_num": cam_num, "roi_num": roi_num,
                                             "detections_num": detections_num, "tracks_num": tracks_num,
                                             "count_num": str(i)}):
                continue
            print(json.dumps(param_set, indent=2))
            cam_prefix = cam_name + "_" + detections_num + "_" + roi_num + "_" + tracks_num
            tracks_file = WORKING_DIR + "/outputs/tracks/tracks_" + cam_prefix + ".csv"
            plot_path = WORKING_DIR + "/outputs/counts/" + cam_name + "/counts_" + \
                        cam_prefix + "_" + str(i) + ".png"
            start = time.time()
            try:
                tracks = self.read_tracks(cam_name, tracks_file, param_set["angle_factor"])
                counter = Counter(tracks, grid_size, param_set["region_factor"])
                first_n = counter.cluster(param_set["min_cluster_size"],
                                          param_set["percent_min_lines"],
                                          param_set["min_lines"],
                                          param_set["min_paths"],
                                          self.min_n, self.max_n)
                counter.post_process()
                second_n = counter.cluster(param_set["min_cluster_size"],
                                           param_set["percent_min_lines"],
                                           param_set["min_lines"],
                                           param_set["min_paths"],
                                           self.min_n, self.max_n, plot_path=plot_path)
                if second_n != first_n:
                    tracks = self.read_tracks(cam_name, tracks_file, param_set["angle_factor"])
                    counter = Counter(tracks, grid_size, param_set["region_factor"])
                    counter.cluster(param_set["min_cluster_size"],
                                    param_set["percent_min_lines"],
                                    param_set["min_lines"],
                                    param_set["min_paths"],
                                    fixed_n=second_n)
                    counter.post_process()
                    counter.cluster(param_set["min_cluster_size"],
                                    param_set["percent_min_lines"],
                                    param_set["min_lines"],
                                    param_set["min_paths"],
                                    fixed_n=second_n, plot_path=plot_path)
                self.log_run(param_set, {
                    "cam_num": cam_num,
                    "detections_num": detections_num,
                    "tracks_num": tracks_num,
                    "count_num": str(i),
                    "cluster_data": [{"cluster_num": cluster.number, "tracks": len(cluster.tracks)}
                                     for cluster in counter.clusters],
                    "tracks_time": time.time() - start,
                    "plot_path": plot_path
                })
            except FileNotFoundError:
                print("File error!")


class Tuning:

    def __init__(self, video_folder,
                 track_hyperparams_file="/outputs/tuning/track_hyperparams.json",
                 count_hyperparams_file="/outputs/tuning/count_hyperparams.json",
                 roi_hyperparams_file="/outputs/tuning/roi_hyperparams.json",
                 detection_hyperparams_file="/outputs/tuning/detection_hyperparams.json",
                 track_log_file="/outputs/tuning/track_log.json",
                 count_log_file="/outputs/tuning/count_log.json",
                 roi_log_file="/outputs/tuning/roi_log.json",
                 detection_log_file="/outputs/tuning/detection_log.json"):
        self.video_folder = video_folder
        self.detection_module = DetectionModule(detection_hyperparams_file, detection_log_file)
        self.roi_module = ROIModule(roi_hyperparams_file, roi_log_file)
        self.track_module = TrackModule(track_hyperparams_file, track_log_file)
        self.count_module = CountModule(count_hyperparams_file, count_log_file)

    def by_video(self, cam_name, detection=False, roi=False, track=True, count=True):
        cam_num = "_".join(cam_name.split("_")[1:])
        if detection:
            self.detection_module.run(cam_name)
        if roi:
            for i in range(len(self.detection_module.hyperparams)):
                self.roi_module.run(cam_name, str(i))
        if track:
            for i in range(len(self.detection_module.hyperparams)):
                for j in range(len(self.roi_module.hyperparams)):
                    self.track_module.run(cam_name, str(i), str(j))
        if count:
            for i in range(len(self.detection_module.hyperparams)):
                for j in range(len(self.roi_module.hyperparams)):
                    for k in range(len(self.track_module.hyperparams)):
                        grid_size = self.roi_module.get_log({
                            "cam_num": cam_num, "roi_num": str(j), "detections_num": str(i)},
                            ["grid_size"])['grid_size']
                        self.count_module.run(cam_name, grid_size, str(i), str(j), str(k))


# 100, 5, 3, 0.05, 5, 3
if __name__ == '__main__':
    WORKING_DIR = os.getcwd()
    PARENT_DIR = os.path.abspath(os.path.join(sys.argv[2], os.pardir))
    tuner = Tuning(PARENT_DIR + "/" + sys.argv[3])
    tuner.by_video(sys.argv[1])

# python object_tracker.py --video ../data/cam_1.mp4 --output ../data/tracked_1_new.avi --model yolov4 --score 0.5
# --tracks_output ../data/tracks_1_new.csv --roi_file ../data/hull_1.txt --info
