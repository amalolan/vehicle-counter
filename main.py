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

WORKING_DIR = ""
PARENT_DIR = ""


class Tuning:

    def __init__(self, video_folder,
                 track_hyperparams_file="/outputs/tuning/track_hyperparams.json",
                 count_hyperparams_file="/outputs/tuning/count_hyperparams.json",
                 roi_hyperparams_file="/outputs/tuning/roi_hyperparams.json",
                 track_log_file="/outputs/tuning/track_log.json",
                 count_log_file="/outputs/tuning/count_log.json",
                 roi_log_file="/outputs/tuning/roi_log.json"):
        self.video_folder = video_folder
        self.hyperparams = dict()
        with open(WORKING_DIR + roi_hyperparams_file, "r") as fp:
            self.hyperparams['roi'] = json.load(fp)
        with open(WORKING_DIR + track_hyperparams_file, "r") as fp:
            self.hyperparams['track'] = json.load(fp)
        with open(WORKING_DIR + count_hyperparams_file, "r") as fp:
            self.hyperparams['count'] = json.load(fp)
        self.log_file = {
            'roi': WORKING_DIR + roi_log_file,
            'track': WORKING_DIR + track_log_file,
            'count': WORKING_DIR + count_log_file
        }

    def log_tune(self, type, hyperparams, outputs):
        with open(self.log_file[type], "r") as fp:
            current_logs = json.load(fp)
        current_logs.append({**hyperparams, **outputs})
        with open(self.log_file[type], "w") as fp:
            json.dump(current_logs, fp)

    def get_roi_output(self, cam_num):
        with open(self.log_file['roi'], "r") as fp:
            roi_logs = json.load(fp)
        for roi_log in roi_logs:
            if roi_log["cam_num"] == cam_num:
                return roi_log

    def check_if_completed(self, type, hyperparams, other_data):
        with open(self.log_file[type], "r") as fp:
            current_logs = json.load(fp)
        for log in current_logs:
            combined = {**hyperparams, **other_data}
            if type == 'count':
                combined["track_num"] = 0
            if combined.items() <= log.items():
                return True
        return False

    def read_tracks(self, cam_num, track_file, h_angle_factor):
        df = pd.read_csv(track_file)
        df.sort_values(by=['track', 'frame'], inplace=True)
        gb = df.groupby(["track"])
        video_reader = imageio.get_reader(self.video_folder + "/cam_" + str(cam_num) + ".mp4")
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

    def count(self):
        min_n, max_n = 2, 15
        for filename in os.listdir(PARENT_DIR + "/videos"):
            cam_name = filename[:-4]
            cam_num = "_".join(cam_name.split("_")[1:])
            cam_dict = self.get_roi_output(cam_num)
            n_frames = int(cam_dict["n_frames"])
            grid_size = int(cam_dict["grid_size"])
            print(cam_num)
            os.makedirs(WORKING_DIR + "/outputs/counts/cam_" + cam_num, exist_ok=True)
            # TODO: Add loop over tracks
            track_file = WORKING_DIR + "/outputs/tracks/tracks_cam_" + cam_num + "_" + str(0) + ".csv"
            for index in range(len(self.hyperparams['count'])):
                hyperparams = self.hyperparams['count'][index]
                if self.check_if_completed('count', hyperparams, {"cam_num": cam_num}):
                    continue
                print(hyperparams)

                plot_path = WORKING_DIR + "/counts/cam_" + cam_num + "/counts_cam_" + \
                            cam_num + "_" + str(0) + "_" + str(index) + ".png"
                start = time.time()
                try:
                    tracks = self.read_tracks(cam_num, track_file, hyperparams["angle_factor"])
                    counter = Counter(tracks, grid_size, hyperparams["region_factor"])
                    first_n = counter.cluster(hyperparams["min_cluster_size"],
                                              hyperparams["percent_min_lines"],
                                              hyperparams["min_lines"],
                                              hyperparams["min_paths"],
                                              min_n, max_n)
                    counter.post_process()
                    second_n = counter.cluster(hyperparams["min_cluster_size"],
                                               hyperparams["percent_min_lines"],
                                               hyperparams["min_lines"],
                                               hyperparams["min_paths"],
                                               min_n, max_n, plot_path=plot_path)
                    if second_n != first_n:
                        tracks = self.read_tracks(cam_num, track_file, hyperparams["angle_factor"])
                        counter = Counter(tracks, grid_size, hyperparams["region_factor"])
                        counter.cluster(hyperparams["min_cluster_size"],
                                        hyperparams["percent_min_lines"],
                                        hyperparams["min_lines"],
                                        hyperparams["min_paths"],
                                        fixed_n=second_n)
                        counter.post_process()
                        counter.cluster(hyperparams["min_cluster_size"],
                                        hyperparams["percent_min_lines"],
                                        hyperparams["min_lines"],
                                        hyperparams["min_paths"],
                                        fixed_n=second_n, plot_path=plot_path)
                    end = time.time()
                    print("fps: ", (n_frames / (end - start)))
                    cam_dict["clustering_fps"] = n_frames / (end - start)
                    cam_dict["track_hyperparam_index"] = 0
                    cam_dict["cluster_hyperparam_index"] = index
                    cam_dict["plot_path"] = plot_path
                    self.log_tune('count', hyperparams, cam_dict)
                except FileNotFoundError:
                    print("File error!")

            break  # TODO: REMOVE if running for all tracks

    # def track(self):
    #     for filename in os.listdir(PARENT_DIR + "/videos"):
    #         cam_name = filename[:-4]
    #         cam_num = "_".join(cam_name.split("_")[1:])
    #         print(cam_num)
    #         for hyperparams in self.hyperparams['track']:
    #             if self.check_if_completed('track', hyperparams, {"cam_num": cam_num}):
    #                 continue
    #             print(hyperparams)
    #             start = time.time()
    #             video_path = PARENT_DIR + "/videos/cam_" + cam_num + ".mp4"
    #             frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    #             try:
    #                 counter_helper([video_path,
    #                                 PARENT_DIR + "/tracked_videos/tracked_cam_" + cam_num + "_" + str(index) + ".avi",
    #                                 str(row[0]),
    #                                 WORKING_DIR + "/outputs/tracks/tracks_cam_" + cam_num + "_" + str(index) + ".csv",
    #                                 WORKING_DIR + "/outputs/hull/hull_cam_" + cam_num + ".txt",
    #                                 str(row[1]),
    #                                 str(int(row[2])),
    #                                 str(int(row[3])),
    #                                 WORKING_DIR + "/outputs/detections/detections_cam_" + cam_num + ".csv",
    #                                 "dont_show"])
    #             except:
    #                 print("Video ended or ERROR!!")
    #             all_fps.append(frames / (time.time() - start))
    #             fps_file.write(json.dumps(all_fps) + "\n")
    #             break  # TODO: REMOVE if running for all tracks


def by_video(roi=False, track=True, count=True):
    pass


def roi_main():
    completed = []
    cam_data = []
    for filename in os.listdir(PARENT_DIR + "/videos"):
        cam_name = filename[:-4]
        cam_num = "_".join(cam_name.split("_")[1:])
        print(cam_num)
        if cam_name in completed:
            continue
        start = time.time()
        # run_detector("../videos/cam_" + cam_num + ".mp4",
        #              video_output="../detections/detected_cam_" + cam_num + ".avi",
        #              detections_output="../detections/detections_cam_" + cam_num + ".csv",
        #              dont_show=True)
        n_frames, box_size, fps, hull_vertices = find_roi(cam_num)
        end = time.time()
        detection_roi_fps = (end - start) / n_frames
        cam_obj = {
            "cam_num": cam_num,
            "n_frames": n_frames,
            "grid_size": box_size,
            "roi_fps": fps,
            "detection_roi_fps": detection_roi_fps
        }
        cam_data.append(cam_obj)
    with open(WORKING_DIR + '/outputs/log.json', 'w+') as fp:
        json.dump(cam_data, fp)


def tracking_main():
    completed = []
    hyperparams = pd.read_csv(WORKING_DIR + "/outputs/hyperparams_track.csv", index_col=False)
    fps_file = open(WORKING_DIR + "/outputs/fps.txt", "a+")
    all_fps = []
    print(hyperparams)
    for filename in os.listdir(PARENT_DIR + "/videos"):
        cam_name = filename[:-4]
        cam_num = "_".join(cam_name.split("_")[1:])
        print(cam_num)
        if cam_name in completed:
            continue
        for index, row in hyperparams.iterrows():
            print(row)
            start = time.time()
            video_path = PARENT_DIR + "/videos/cam_" + cam_num + ".mp4"
            frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
            try:
                counter_helper([video_path,
                                PARENT_DIR + "/tracked_videos/tracked_cam_" + cam_num + "_" + str(index) + ".avi",
                                str(row[0]),
                                WORKING_DIR + "/outputs/tracks/tracks_cam_" + cam_num + "_" + str(index) + ".csv",
                                WORKING_DIR + "/outputs/hull/hull_cam_" + cam_num + ".txt",
                                str(row[1]),
                                str(int(row[2])),
                                str(int(row[3])),
                                WORKING_DIR + "/outputs/detections/detections_cam_" + cam_num + ".csv",
                                "dont_show"])
            except:
                print("Video ended or ERROR!!")
            all_fps.append(frames / (time.time() - start))
            fps_file.write(json.dumps(all_fps) + "\n")
            break  # TODO: REMOVE if running for all tracks
    fps_file.close()


def counter_main():
    track_hyperparams_len = pd.read_csv(WORKING_DIR + "/outputs/hyperparams_track.csv", index_col=False).shape[0]
    hyperparams = pd.read_csv(WORKING_DIR + "/outputs/hyperparams_cluster.csv", index_col=False)
    min_n, max_n = 2, 15
    print(hyperparams)
    with open(WORKING_DIR + "/outputs/log.json", "r") as fp:
        cam_data = json.load(fp)
    final_cam_data = []
    for i in range(track_hyperparams_len):
        for cam_dict in cam_data:
            cam_num = cam_dict["cam_num"]
            os.makedirs(WORKING_DIR + "/outputs/counts/cam_" + cam_num, exist_ok=True)
            track_file = WORKING_DIR + "/outputs/tracks/tracks_cam_" + cam_num + "_" + str(i) + ".csv"
            n_frames = int(cam_dict["n_frames"])
            grid_size = int(cam_dict["grid_size"])

            for index, row in hyperparams.iterrows():
                print(row)
                plot_path = WORKING_DIR + "/counts/cam_" + cam_num + "/counts_cam_" + \
                            cam_num + "_" + str(i) + "_" + str(index) + ".png"
                start = time.time()
                try:
                    tracks = read_tracks(cam_num, track_file, row[0])
                    counter = Counter(tracks, grid_size, row[1])
                    first_n = counter.cluster(*(row[2:]), min_n, max_n)
                    counter.post_process()
                    second_n = counter.cluster(*(row[2:]), min_n, max_n, plot_path=plot_path)
                    if second_n != first_n:
                        tracks = read_tracks(cam_num, track_file, row[0])
                        counter = Counter(tracks, grid_size, row[1])
                        counter.cluster(*(row[2:]), fixed_n=second_n)
                        counter.post_process()
                        counter.cluster(*(row[2:]), fixed_n=second_n, plot_path=plot_path)
                    end = time.time()
                    print("fps: ", (n_frames / (end - start)))
                    cam_dict["clustering_fps"] = n_frames / (end - start)
                    cam_dict["track_hyperparam_index"] = i
                    cam_dict["cluster_hyperparam_index"] = index
                    cam_dict["plot_path"] = plot_path
                    final_cam_data.append(cam_dict.copy())
                except FileNotFoundError:
                    print("File error!")

        break  # TODO: REMOVE if running for all tracks

    with open(WORKING_DIR + '/outputs/final_log.json', 'w+') as fp:
        json.dump(final_cam_data, fp)


"""
Used to test clustering to see results if true k was used instead of finding
k with k-nn and silhouette index.
"""


# def test_true_k(k_file="/outputs/true_k.json", angle_factor=100, region_factor=5,
#                 min_cluster_size=3, percent_min_lines=0.05,
#                 min_lines=5, min_paths=3):
#     with open(WORKING_DIR + "/outputs/log.json", "r") as fp:
#         cam_data = json.load(fp)
#     with open(WORKING_DIR + k_file, "r") as fp:
#         cam_k = json.load(fp)
#     for cam_dict in cam_data:
#         cam_num = cam_dict["cam_num"]
#         track_file = WORKING_DIR + "/outputs/tracks/tracks_cam_" + cam_num + "_0.csv"
#         n_frames = int(cam_dict["n_frames"])
#         grid_size = int(cam_dict["grid_size"])
#         k = cam_k["cam_" + cam_num.split("_")[0]]
#         tracks = read_tracks(cam_num, track_file, angle_factor)
#         counter = Counter(tracks, grid_size, region_factor)
#         first_n = counter.cluster(min_cluster_size, percent_min_lines, min_lines, min_paths,
#                                   fixed_n=k)
#         counter.post_process()
#         second_n = counter.cluster(min_cluster_size, percent_min_lines, min_lines, min_paths,
#                                    fixed_n=k, plot_path=WORKING_DIR + "/outputs/true_k/cam_" + cam_num + ".png")


# 100, 5, 3, 0.05, 5, 3
if __name__ == '__main__':
    WORKING_DIR = os.getcwd()
    PARENT_DIR = os.path.abspath(os.path.join(WORKING_DIR, os.pardir))
    tuner = Tuning(PARENT_DIR + "/videos")
    tuner.count()
    # roi_main()
    # tracking_main()
    # counter_main()
    # test_true_k()

# python object_tracker.py --video ../data/cam_1.mp4 --output ../data/tracked_1_new.avi --model yolov4 --score 0.5
# --tracks_output ../data/tracks_1_new.csv --roi_file ../data/hull_1.txt --info
