import pandas as pd
import imageio
import json


def read_detections(cam_name, detections_file, frame_file, frame_log_file):
    with open(frame_file, "r") as fp:
        frame_data = json.load(fp)
    df = pd.read_csv(detections_file,
                     dtype={'frame': int, 'x': int, 'y': int, 'w': int,
                            'h': int, 'confidence': float})
    n_frames = df['frame'].max()
    with open(frame_log_file, 'r') as fp:
        current_logs = json.load(fp)
    print(current_logs)
    for frames in frame_data:
        frames['n_frames'] = int(frames['n_frames_percent'] * n_frames)
        frames['n_detections'] = sum(df['frame'] <= frames['n_frames'])
        frames["cam_num"] = "_".join(cam_name.split("_")[1:])
        current_logs.append(frames)
    # current_logs.append({frame_data})
    print(current_logs)
    with open(frame_log_file, "w+") as fp:
        json.dump(current_logs, fp, indent=4)

if __name__ == "__main__":
    read_detections("cam_14", "outputs/detections/detections_cam_14_0.csv",
                    "outputs/tuning/frame_hyperparams.json", "outputs/tuning/frame_log.json")

