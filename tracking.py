import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import scipy.io
import predict
from pathlib import Path

from utils import CentroidTracker
from collections import OrderedDict
from typing import List, Tuple

# models/research/object_detection muss im PYTHONPATH sein

parser = argparse.ArgumentParser(description='Track spines in the whole stack',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', '--images', required=False,
                    help='Path to input images')
parser.add_argument('-t', '--threshold',
                    help='Threshold for detection', default=0.5, type=float)
parser.add_argument('-a', '--appeared',
                    help='appeared counter', default=0, type=int)
parser.add_argument('-d', '--disappeared',
                    help='disappeared counter', default=1, type=int)
parser.add_argument('-th', '--theta',
                    help='Threshold for theta (detection similarity threshold)', default=0.5, type=float)
parser.add_argument('-ta', '--tau',
                    help='Threshold for tau (tracking threshold)', default=0.3, type=float)
parser.add_argument('-m', '--model',
                    help='Path to model you want to analyze with')
parser.add_argument('-c', '--csv', required=False,
                    help='Single file or folder of csv files for previous prediction.'
                    'If this flag is set, no model prediction will be executed')
parser.add_argument('-s', '--save-images', action='store_true',
                    help='Activate this flag if images should be saved')
parser.add_argument('-o', '--output', required=False,
                    help='Path where tracking images and csv should be saved, default: output/tracking/MODEL')
parser.add_argument('-f', '--file-save',
                    help="Name of tracked data csv file", default="data_tracking.csv")
parser.add_argument('-mc', '--metric', default='iom',
                    help='Metric which should be used for evaluating. Currently available: iom, iou.'
                    'Own metric can be implemented as lambda function which takes two arguments and returns one.')
parser.add_argument('-uo', '--use-offsets', action='store_true',
                    help='whether offsets should be used or not')


def draw_boxes(img: np.ndarray, objects: OrderedDict) -> np.ndarray:
    """Draw boxes onto image

    Args:
        img (np.ndarray): image input to draw on
        objects (OrderedDict): Dictionary of objects of format (cX, cY, w, h, conf)

    Returns:
        np.ndarray: output image with drawn boxes
    """
    for key in objects:
        # w, h = 512, 512
        cX, cY, width, height, conf = objects[key]
        x1, x2 = int(cX-width/2), int(cX+width/2)
        y1, y2 = int(cY-height/2), int(cY+height/2)
        # correct colored rectangle
        # opencv : BGR!!!! NO RGB!!
        # linear from (0,0,255) to (255,255,0)

        # color = (255*(1-conf), 255*conf, 255*conf)
        color = (0, 255, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # green filled rectangle for text
        color = (0, 255, 0)
        text_color = (0, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x1+25, y1-12), color, thickness=-1)

        # text
        img = cv2.putText(img, '{:02.0f}%'.format(
            conf*100), (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
    return img


def csv_to_boxes(df: pd.DataFrame) -> Tuple[List]:
    """Convert a dataframe into the relevant detection information

    Args:
        df (pd.DataFrame): Dataframe of interest

    Returns:
        Tuple[List]: Tuple containing boxes, scores, classes, num detections 
    """
    boxes, scores, classes = [], [], []
    for i in range(len(df)):
        if len(df.iloc[i]) == 8:
            filename, w, h, class_name, x1, y1, x2, y2 = df.iloc[i]
            score = 1.0
        else:
            filename, w, h, class_name, score, x1, y1, x2, y2 = df.iloc[i]
        scores.append(score)
        classes.append(1)  # all are spines
        # boxes are in y1, x1, y2, x2 format!!!
        boxes.append([x1/w, y1/h, x2/w, y2/h])
    boxes = [boxes]
    scores = [scores]
    classes = [classes]
    num_detections = [len(scores[0])]
    return boxes, scores, classes, num_detections


if __name__ == '__main__':
    args = parser.parse_args()
    # Max diff -> (minimum) diff so that two following bboxes are connected with each other
    # iom thresh -> min iom that two boxes are considered the same in the same frame!
    MAX_DIFF = args.tau
    IOM_THRESH = args.theta
    THRESH = args.threshold
    MIN_APP = args.appeared
    MAX_DIS = args.disappeared
    METRIC = args.metric
    NUM_CLASSES = 1
    MAX_VOL = 2000

    if args.images is None:
        raise ValueError('You need specify input images or input tif stack!')

    # save_folder: folder where tracking csv file will be saved
    # folder: name of folder which is used in csv file for generating filename-column
    if args.model is not None:
        model_name = args.model.split(
            "/")[-1] if args.model.split("/")[-1] != "" else args.model.split("/")[-2]
    if args.output is None:
        args.output = os.path.join('output/tracking', model_name)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    img_output_path = os.path.join(args.output, 'images')
    csv_output_path = os.path.join(args.output, args.file_save)
    if args.save_images and not os.path.exists(img_output_path):
        os.makedirs(img_output_path)

    # to get some annotations on the first images too, make the same backwards
    all_imgs = sorted(glob.glob(args.images))

    all_dicts = []
    total_boxes = []
    total_classes = []
    total_scores = []
    nr_imgs = len(list(all_imgs))
    objects = dict()

    # if it's just a single csv file, load all data before iterating over images
    if args.csv is not None:
        all_csv_files = glob.glob(args.csv)
        if len(all_csv_files) == 0:
            raise ValueError(
                'No csv files with valid prediction data are available.')
        csv_path = args.csv

    # get all boxes, scores and classes at the start if prediction is necessary:
    if args.csv is None:
        detection_graph = predict.load_model(args.model)
        all_boxes, all_scores, all_classes, all_num_detections = predict.predict_images(
            detection_graph, args.images, img_output_path, csv_output_path, THRESH, save_csv=False, return_csv=True)
    all_csv_paths = list(Path().rglob(args.csv))

    ct = CentroidTracker(maxDisappeared=MAX_DIS, minAppeared=MIN_APP,
                         maxDiff=MAX_DIFF, iomThresh=IOM_THRESH, maxVol=MAX_VOL, metric=METRIC)

    # get offsets if we want to use them
    if args.use_offsets:
        sr, neuron, dend, day = 52, 1, 1, 1
        arrx = scipy.io.loadmat(
            f'data/offsets/SR{sr}N{neuron}D{dend}offsetX.mat')[f'SR{sr}N{neuron}D{dend}offsetX']
        arry = scipy.io.loadmat(
            f'data/offsets/SR{sr}N{neuron}D{dend}offsetY.mat')[f'SR{sr}N{neuron}D{dend}offsetY']

        # get offset for each stack
        offsets = np.array(
            list(zip(arrx[:, day-1], arry[:, day-1]))).astype(int)

        # double offsets so that it can easily be added to bounding boxes
        # divide through width = height = 512 to get correct box-offset
        offsets = np.concatenate((offsets, offsets), axis=1) / 512

        # make offset positive by subtracting possible negative offsets (including first offset of 0)
        offsets = offsets - np.min(offsets, axis=0)

    # use given prediction for all images, if csv is available
    for i, img in enumerate(all_imgs):
        orig_img = Path(img).name
        if args.csv is not None:
            if len(all_csv_paths) > 1:
                csv_path = [
                    elem for elem in all_csv_paths if orig_img[:-4] == elem.name[:-4]]
                if len(csv_path) == 0:
                    # no corresponding csv file for this image
                    continue
                else:
                    csv_path = csv_path[0]
                try:
                    new_df = pd.read_csv(csv_path)
                    boxes, scores, classes, num_detections = csv_to_boxes(
                        new_df)
                except:
                    continue
            else:
                try:
                    new_df = pd.read_csv(args.csv)

                    # load only data from interesting image
                    new_df = new_df[new_df.apply(lambda row: os.path.splitext(
                        orig_img)[0] in row['filename'], axis=1)]  # axis=1 for looping through rows

                    boxes, scores, classes, num_detections = csv_to_boxes(
                        new_df)
                except:
                    continue
        else:
            # just load data from saved list
            # this works as all_imgs from this file and sorted(glob.glob(args.images)) from predict sort all
            # image paths so they are perfectly aligned
            boxes, scores, classes, num_detections = all_boxes[
                i], all_scores[i], all_classes[i], all_num_detections[i]
        boxes = boxes[0]

        # look if there are some boxes
        if len(boxes) == 0:
            continue

        # convert all detections from different stacks into one stack (via offset matlab files)
        if args.use_offsets:
            # format of img name: SR52N1D1day1stack1-xx.png
            stack_nr = int(orig_img[-8])
            boxes += offsets[stack_nr - 1]

        scores = scores[0]
        num_detections = int(num_detections[0])

        image_np = cv2.imread(img)
        h, w = image_np.shape[:2]
        # Real tracking part!
        rects = np.array([[boxes[i][0]*w, boxes[i][1]*h,
                           boxes[i][2]*w, boxes[i][3]*h, scores[i]] for i in range(num_detections)
                          if scores[i] >= THRESH])

        objects = ct.update(rects)  # y1, x1, y2, x2 - format

        # Start with non-empty lists
        boxes = []
        classes = []
        scores = []

        # DO NOT USE absolute path for images!
        total_path = os.path.join(img_output_path, img.split('/')[-1])
        for key in objects:
            orig_dict = {'filename': total_path,
                         'width': w, 'height': h, 'class': 'spine'}

            # Making boxes, classes, scores correct
            cX, cY, width, height, conf = objects[key]
            x1, x2 = (cX-width/2)/w, (cX+width/2)/w
            y1, y2 = (cY-height/2)/h, (cY+height/2)/h
            boxes.append([x1, y1, x2, y2])
            classes.append(1)
            scores.append(conf)

            orig_dict.update({'id': key, 'ymin': round(y1*h, 2), 'ymax': round(y2*h, 2), 'xmin': round(x1*w, 2),
                              'xmax': round(x2*w, 2), 'score': conf})

            all_dicts.append(orig_dict)

        boxes = np.array(boxes)
        classes = np.array(classes)
        scores = np.array(scores)
        total_boxes.append(boxes)
        total_classes.append(classes)
        total_scores.append(scores)

        if args.save_images:
            image_np = cv2.imread(img)
            image_np = draw_boxes(image_np, objects)

    # delete all double elements
    all_dicts = [dict(tup)
                 for tup in {tuple(set(elem.items())) for elem in all_dicts}]
    df = pd.DataFrame(all_dicts, columns=[
                      'id', 'filename', 'width', 'height', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.sort_values(by='filename', inplace=True)
    df.to_csv(csv_output_path, index=False)

    # count real spines (does NOT correspond to max_key, but to number of keys!)
    nr_all_ind = len(df.groupby('id'))
    print(f"Nr of spines found: {nr_all_ind}")

    print('[INFO] Written predictions to '+csv_output_path+'.')
