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
    def run(self, cam_name, detections_num=None, single=False):
        cam_num = "_".join(cam_name.split("_")[1:])
        print("Detection: ", cam_num)
        for i in range(len(self.hyperparams)):
            if detections_num is not None:
                i = int(detections_num)
                single = True
            param_set = self.hyperparams[i]
            if self.is_completed(param_set, {"cam_num": cam_num}):
                if single:
                    break
                else:
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
            if single:
                break


class ROIModule(Module):
    def run(self, cam_name, n_frames_percent=1, detections_num="0", frames_num="0",
            roi_num=None, single=False):
        cam_num = "_".join(cam_name.split("_")[1:])
        print("ROI ", cam_num)
        for i in range(len(self.hyperparams)):
            if roi_num is not None:
                i = int(roi_num)
                single = True
            param_set = self.hyperparams[i]
            if self.is_completed(param_set, {"cam_num": cam_num, "detections_num": detections_num,
                                             "frames_num": frames_num}):
                if single:
                    break
                else:
                    continue
            start = time.time()
            hull_file_prefix = WORKING_DIR + "/outputs/hull/hull_" + cam_name + "_" + frames_num + \
                               "_" + detections_num + "_" + str(i)
            n_frames, box_size, hull_vertices = find_roi(
                PARENT_DIR + "/videos/" + cam_name + ".mp4",
                WORKING_DIR + "/outputs/detections/detections_" + cam_name + "_" + detections_num + ".csv",
                hull_file_prefix + ".png", hull_file_prefix + ".txt",
                param_set["dfs_confidence"], param_set["outlier_threshold"],
                n_frames_percent=n_frames_percent
            )
            print(param_set)
            self.log_run(param_set, {
                "cam_num": cam_num,
                "detections_num": detections_num,
                "frames_num": frames_num,
                "roi_num": str(i),
                "total_frames": n_frames,
                "grid_size": box_size,
                "detection_time": time.time() - start,
                "hull_file": hull_file_prefix + ".txt"
            })
            if single:
                break


class TrackModule(Module):
    def run(self, cam_name, detections_num="0", roi_num="0",
            frames_num="0", single=False, tracks_num=None):
        cam_num = "_".join(cam_name.split("_")[1:])
        video_path = PARENT_DIR + "/videos/cam_" + cam_num + ".mp4"
        print("Track ", cam_num)
        for i in range(len(self.hyperparams)):
            if tracks_num is not None:
                i = int(tracks_num)
                single = True
            param_set = self.hyperparams[i]
            if self.is_completed(param_set, {"cam_num": cam_num, "detections_num": detections_num,
                                             "frames_num": frames_num, "roi_num": roi_num}):
                if single:
                    break
                else:
                    continue
            start = time.time()
            # frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
            cam_prefix = cam_name + "_" + frames_num + "_" + detections_num + "_" + roi_num
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
                "frames_num": frames_num,
                "roi_num": roi_num,
                "tracks_num": str(i),
                "tracks_file": tracks_file,
                "tracks_time": time.time() - start
            })
            if single:
                break


class CountModule(Module):
    def __init__(self, hyperparams_file, log_file, min_n=2, max_n=15):
        super().__init__(hyperparams_file, log_file)
        self.min_n = min_n
        self.max_n = max_n

    def read_tracks(self, cam_name, tracks_file, h_angle_factor, n_frames):
        df = pd.read_csv(tracks_file)
        df.sort_values(by=['track', 'frame'], inplace=True)
        df = df[df['frame'] <= int(n_frames)]
        gb = df.groupby(["track"])
        video_reader = imageio.get_reader(PARENT_DIR + "/videos/" + cam_name + ".mp4")
        image = video_reader.get_data(0)
        video_reader.close()
        tracks = Tracks(h_angle_factor, image)
        count = 0
        for x in gb.groups:
            track_df = gb.get_group(x)
            confidence = float(track_df.tail(1)['confidence'])
            coords = np.c_[track_df['x'].to_numpy(), track_df['y'].to_numpy()]
            track = Track(coords,
                          track_df['frame'].to_numpy(),
                          track_df['width'].mean(),
                          track_df['height'].mean(),
                          confidence=confidence,
                          cls=track_df.iloc[0]['class'],
                          id=count
                          )
            tracks.append(track)
            count += 1
        return tracks

    def run(self, cam_name, grid_size, n_frames, total_frames,
            detections_num="0", roi_num="0", tracks_num="0", frames_num="0",
            single=False, analyze=False, count_num=None):
        cam_num = "_".join(cam_name.split("_")[1:])
        os.makedirs(WORKING_DIR + "/outputs/counts/" + cam_name, exist_ok=True)
        print("Count ", cam_num)
        for i in range(len(self.hyperparams)):
            if count_num is not None:
                i = int(count_num)
                single = True
            param_set = self.hyperparams[i]
            if self.is_completed(param_set, {"cam_num": cam_num, "roi_num": roi_num,
                                             "detections_num": detections_num,
                                             "tracks_num": tracks_num,
                                             "frames_num": frames_num}):
                if single:
                    break
                else:
                    continue
            print(json.dumps(param_set, indent=2))
            cam_prefix = cam_name + "_" + frames_num + "_" + detections_num + "_" + roi_num + "_" + tracks_num
            tracks_file = WORKING_DIR + "/outputs/tracks/tracks_" + cam_prefix + ".csv"
            plot_path = WORKING_DIR + "/outputs/counts/" + cam_name + "/counts_" + \
                        cam_prefix + "_" + str(i) + ".png"
            start = time.time()
            try:
                tracks = self.read_tracks(cam_name, tracks_file, param_set["angle_factor"], n_frames)
                # tracks.plot()
                counter = Counter(tracks, grid_size, param_set["region_factor"])
                counter.cluster(param_set["min_cluster_size"],
                                param_set["percent_min_lines"],
                                param_set["min_lines"],
                                param_set["min_paths"],
                                self.min_n, self.max_n)
                counter.post_process()
                # tracks.plot()
                counter.cluster(param_set["min_cluster_size"],
                                param_set["percent_min_lines"],
                                param_set["min_lines"],
                                param_set["min_paths"],
                                self.min_n, self.max_n, plot_path=plot_path)
                tracks_for_n_frames = len(tracks.tracks_list)

                all_tracks = self.read_tracks(cam_name, tracks_file, param_set["angle_factor"],
                                              total_frames)
                all_tracks.remove_small_paths()
                all_counter = Counter(all_tracks, grid_size, param_set['region_factor'])
                all_counter.update_track_clusters(counter.clusters, plot_path)
                log_object = {
                    "cam_num": cam_num,
                    "detections_num": detections_num,
                    "roi_num": roi_num,
                    "frames_num": frames_num,
                    "tracks_num": tracks_num,
                    "deep_sort_n_frames": tracks_for_n_frames,
                    "avg_confidence_n_frames": all_tracks.get_avg_confidence(n_frames),
                    "avg_confidence_total_frames": all_tracks.get_avg_confidence(),
                    "cluster_data": [{"cluster_num": cluster.number, "counted_tracks": len(cluster.tracks),
                                      "n_tracks_in_n_frames": cluster.get_num_tracks_for_frame(n_frames),
                                      "avg_confidence_in_n_frames": cluster.get_avg_confidence(n_frames)}
                                     for cluster in all_counter.clusters],
                    "tracks_time": time.time() - start,
                    "plot_path": plot_path
                }
                if analyze:
                    full_tracks = self.read_tracks(cam_name,
                                                   "/Users/malolan/Documents/Research/Traffic/final/yolov4-deepsort/outputs/tracks/tracks_" + cam_name + "_0_0_0_0.csv"
                                                   , param_set["angle_factor"], total_frames)
                    full_counter = Counter(full_tracks, grid_size, param_set["region_factor"])
                    full_counter.cluster(param_set["min_cluster_size"],
                                         param_set["percent_min_lines"],
                                         param_set["min_lines"],
                                         param_set["min_paths"],
                                         self.min_n, self.max_n)
                    full_counter.post_process()
                    # full_tracks.plot()
                    full_counter.cluster(param_set["min_cluster_size"],
                                         param_set["percent_min_lines"],
                                         param_set["min_lines"],
                                         param_set["min_paths"],
                                         self.min_n, self.max_n)
                    new_plot_path = plot_path[:-4] + "_frame_counts.png"
                    new_tracks = self.read_tracks(cam_name, tracks_file, param_set["angle_factor"], n_frames)
                    new_counter = Counter(new_tracks, grid_size, param_set["region_factor"])
                    new_counter.update_track_clusters(full_counter.clusters, new_plot_path,
                                                      debug=False, total_n=True)
                    log_object["n_tracks_based_on_total_frames"] = [
                        {"cluster_num": cluster.number,
                         "n_tracks_in_total_frames": len(cluster.tracks),
                         "avg_confidence_in_total_frames": cluster.get_avg_confidence()}
                        for cluster in new_counter.clusters]
                    tracks_df = all_tracks.save_tracks_df(old_cluster_map=full_tracks.get_id_cluster_map(),
                                                          path=WORKING_DIR + '/outputs/track_data/' +
                                                               cam_prefix + ".csv")

                self.log_run(param_set, log_object)

                if single:
                    break
            except FileNotFoundError:
                print("File error!")


class Tuning:

    def __init__(self, video_folder,
                 frame_hyperparams_file="/outputs/tuning/frame_hyperparams.json",
                 track_hyperparams_file="/outputs/tuning/track_hyperparams.json",
                 count_hyperparams_file="/outputs/tuning/count_hyperparams.json",
                 roi_hyperparams_file="/outputs/tuning/roi_hyperparams.json",
                 detection_hyperparams_file="/outputs/tuning/detection_hyperparams.json",
                 track_log_file="/outputs/tuning/track_log.json",
                 count_log_file="/outputs/tuning/count_log.json",
                 roi_log_file="/outputs/tuning/roi_log.json",
                 detection_log_file="/outputs/tuning/detection_log.json"):
        self.video_folder = video_folder
        with open(WORKING_DIR + frame_hyperparams_file, "r") as fp:
            self.frames = json.load(fp)
        self.detection_module = DetectionModule(detection_hyperparams_file, detection_log_file)
        self.roi_module = ROIModule(roi_hyperparams_file, roi_log_file)
        self.track_module = TrackModule(track_hyperparams_file, track_log_file)
        self.count_module = CountModule(count_hyperparams_file, count_log_file)

    def online(self, cam_name, detection=False, roi=True, track=True, count=True,
               detections_num="0", roi_num="0", count_num="5", tracks_num="7"):
        cam_num = "_".join(cam_name.split("_")[1:])
        for frame_obj in self.frames:
            if detection:
                self.detection_module.run(cam_num, detections_num=detections_num)
            if roi:
                self.roi_module.run(cam_name, n_frames_percent=frame_obj['n_frames_percent'],
                                    frames_num=frame_obj['frames_num'],
                                    detections_num=detections_num,
                                    roi_num=roi_num)
            if track:
                self.track_module.run(cam_name, frames_num=frame_obj['frames_num'],
                                      detections_num=detections_num,
                                      roi_num=roi_num, tracks_num=tracks_num)
            if count:
                logged_data = self.roi_module.get_log({
                    "cam_num": cam_num, "roi_num": roi_num, "detections_num": detections_num},
                    ["grid_size"])
                grid_size = logged_data['grid_size']
                n_frames = int(logged_data["total_frames"] * frame_obj["n_frames_percent"])
                self.count_module.run(cam_name, grid_size, n_frames, logged_data['total_frames'],
                                      frames_num=frame_obj['frames_num'],
                                      detections_num=detections_num,
                                      roi_num=roi_num, tracks_num=tracks_num,
                                      count_num=count_num)


# 100, 5, 3, 0.05, 5, 3
if __name__ == '__main__':
    WORKING_DIR = os.getcwd()
    PARENT_DIR = os.path.abspath(os.path.join(sys.argv[2], os.pardir))
    tuner = Tuning(PARENT_DIR + "/" + sys.argv[3])
    cam_names = [sys.argv[1]]
    if sys.argv[1] == 'all':
        cam_names = ['cam_1', 'cam_2', 'cam_4', 'cam_5', 'cam_8',
                     'cam_10', 'cam_11', 'cam_14', 'cam_15', 'cam_16']
    elif sys.argv[1] == 'debug':
        cam_names = ['cam_1']
    for cam_name in cam_names:
        tuner.online(cam_name)

# python object_tracker.py --video ../data/cam_1.mp4 --output ../data/tracked_1_new.avi --model yolov4 --score 0.5
# --tracks_output ../data/tracks_1_new.csv --roi_file ../data/hull_1.txt --info
