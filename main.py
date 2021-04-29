from counter.Counter import *
from counter.roi import find_roi
from roi_object_tracker import run_detector
from object_tracker import counter_helper
import json
import time


def read_tracks(cam_num, track_file, h_angle_factor):
    df = pd.read_csv(track_file)
    df.sort_values(by=['track', 'frame'], inplace=True)
    gb = df.groupby(["track"])
    video_reader = imageio.get_reader("../videos/cam_" + str(cam_num) + ".mp4")
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


def roi_main():
    completed = []
    cam_data = []
    for filename in os.listdir("../videos"):
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
    with open('../log.json', 'w+') as fp:
        json.dump(cam_data, fp)


def tracking_main():
    completed = []
    hyperparams = pd.read_csv("../hyperparams_track.csv", index_col=False)
    print(hyperparams)
    for filename in os.listdir("../videos"):
        cam_name = filename[:-4]
        cam_num = "_".join(cam_name.split("_")[1:])
        print(cam_num)
        if cam_name in completed:
            continue
        for index, row in hyperparams.iterrows():
            print(row)
            try:
                counter_helper(["../videos/cam_" + cam_num + ".mp4",
                                "../tracked_videos/tracked_cam_" + cam_num + "_" + str(index) + ".avi",
                                str(row[0]),
                                "../tracks/tracks_cam_" + cam_num + "_" + str(index) + ".csv",
                                "../hull/hull_cam_" + cam_num + ".txt",
                                str(row[1]),
                                str(int(row[2])),
                                str(int(row[3])),
                                "../detections/detections_cam_" + cam_num + ".csv",
                                "dont_show"])
            except:
                print("Video ended or ERROR!!")
            break  # TODO: REMOVE if running for all tracks


def counter_main():
    track_hyperparams_len = pd.read_csv("../hyperparams_track.csv", index_col=False).shape[0]
    hyperparams = pd.read_csv("../hyperparams_cluster.csv", index_col=False)
    min_n, max_n = 2, 15
    print(hyperparams)
    with open("../log.json", "r") as fp:
        cam_data = json.load(fp)
    final_cam_data = []
    for i in range(track_hyperparams_len):
        for cam_dict in cam_data:
            cam_num = cam_dict["cam_num"]
            os.makedirs("../counts/cam_"+cam_num, exist_ok=True)
            track_file = "../tracks/tracks_cam_" + cam_num + "_" + str(i) + ".csv"
            n_frames = int(cam_dict["n_frames"])
            grid_size = int(cam_dict["grid_size"])

            for index, row in hyperparams.iterrows():
                print(row)
                plot_path = "../counts/cam_"+cam_num+"/counts_cam_" + cam_num + "_" + str(i) + "_" + str(index) + ".png"
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

    with open('../final_log.json', 'w+') as fp:
        json.dump(final_cam_data, fp)

# def test():
#


if __name__ == '__main__':
    # roi_main()
    tracking_main()
    counter_main()

# python object_tracker.py --video ../data/cam_1.mp4 --output ../data/tracked_1_new.avi --model yolov4 --score 0.5
# --tracks_output ../data/tracks_1_new.csv --roi_file ../data/hull_1.txt --info
