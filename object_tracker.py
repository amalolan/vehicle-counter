import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from scipy.spatial import Delaunay
import os

import cv2
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string('roi_file', None, 'ROI file for YOLO detections')
flags.DEFINE_string('tracks_output', None, 'path to output track information from video')
flags.DEFINE_float('max_iou_distance', 0.9, 'max iou distance')
flags.DEFINE_integer('max_age', 60, 'max age')
flags.DEFINE_integer('n_init', 6, 'max age')
flags.DEFINE_string('detections_file', None, 'pre-computed YOLO detections')


def counter_helper(_argv):
    app.run(main, _argv)


def main(_argv):
    if len(_argv) > 0:
        FLAGS.video = _argv[0]
        FLAGS.output = _argv[1]
        FLAGS.score = float(_argv[2])
        FLAGS.tracks_output = _argv[3]
        FLAGS.roi_file = _argv[4]
        FLAGS.max_iou_distance = float(_argv[5])
        FLAGS.max_age = int(_argv[6])
        FLAGS.n_init = int(_argv[7])
        FLAGS.detections_file = _argv[8]
        if len(_argv) == 10:
            FLAGS.dont_show = True
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric, max_iou_distance=FLAGS.max_iou_distance, max_age=FLAGS.max_age, n_init=FLAGS.n_init)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.tracks_output:
        tracks_file = open(FLAGS.tracks_output, 'w+')
        tracks_file.write("track,frame,x,y,class,width,height\n")

    if FLAGS.roi_file:
        roi = np.genfromtxt(FLAGS.roi_file, delimiter=',')
        hull = Delaunay(roi)

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    # while video is running

    detections_df = pd.read_csv(FLAGS.detections_file, index_col=False)
    detections_df = detections_df[detections_df["confidence"] > FLAGS.score]
    detections_df = detections_df[detections_df.apply(
        lambda row: hull.find_simplex(np.array((row['x'], row['y']))) >= 0, axis=1)]

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('Frame #: ', frame_num, " / ", length, video_path)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        detections_subset = detections_df[detections_df["frame"] == frame_num]
        bboxes = detections_subset.apply(lambda row: np.array([int(row['x'] - row['w'] / 2), int(row['y'] - row['h']
                                                                                                 / 2),
                                                           row['w'], row['h']]), axis=1).to_numpy()

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, 'car', feature) for bbox, score, feature in
                      zip(bboxes, detections_subset["confidence"], features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                                                                                    class_name, (
                                                                                                        int(bbox[0]),
                                                                                                        int(bbox[1]),
                                                                                                        int(bbox[2]),
                                                                                                        int(bbox[3]))))
            if FLAGS.tracks_output:
                center = ((int(bbox[0]) + int(bbox[2])) // 2,
                          (int(bbox[1]) + int(bbox[3])) // 2)
                width = int(bbox[2] - bbox[0])
                height = int(bbox[3] - bbox[1])
                tracks_file.write(str(track.track_id) + "," + str(frame_num) + ","
                                  + str(center[0]) + "," + str(center[1]) + "," + str(class_name) +
                                  "," + str(width) + "," + str(height) + "\n")

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        # result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    if FLAGS.tracks_output:
        tracks_file.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
