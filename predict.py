import numpy as np
import os
import glob
import sys
import tensorflow as tf
import cv2
import argparse
import pandas as pd
import time
from collections import OrderedDict
from utils import calc_IoM
from scipy.spatial.distance import pdist
sys.path.append("..")
sys.path.extend(["models/research/object_detection", "model/research", "models/research/slim"])

# remove deprecation warnings
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

# from utils import label_map_util

# TODO: get image_decode into write_to_df function!

parser = argparse.ArgumentParser(description='Make prediction on images')
parser.add_argument('-m', '--model', \
    help='Model used for prediction (without frozen_inference_graph.pb!) or folder where csv files are saved')
parser.add_argument('-t', '--threshold',
    help='Threshold for detection', default=0.5, type=float)
parser.add_argument('-th', '--theta',
    help='Threshold for theta (detection similarity threshold)', default=0.5, type=float)
parser.add_argument('-C', '--use_csv', action='store_true', \
    help='activate this flag if you want to use the given csv files')
# parser.add_argument('-l', '--labelmap', \
#     help='Path to labelmap file ending with .pbtxt', default='data/spine_label_map.pbtxt')
parser.add_argument('-i ', '--input', \
    help='Path to input image(s), ready for prediction. Path can contain wildcards but must start and end with "')
parser.add_argument('-s', '--save_images', action='store_true', \
    help='Activate this flag if images should be saved')
parser.add_argument('-o', '--output', required=False, \
    help='Path where prediction images and csvs should be saved, default: output/prediction/MODEL')

def image_load_encode(img_path):
    # function to read img from given path and convert to get correct 512x512 format
    #new_img = np.zeros((512, 512, 3))
    #new_img[:256, :] = image[:, :512]
    #new_img[256:, :] = image[:, 512:]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    return img.copy(), w, h

def image_decode(img=None, rect=None):
    # function to decode img or detection, depending which type is provided to get original img/detection back
    # rects have x/y values between 0 and 512 and are of type xmin, ymin, xmax, ymax
    # convert img back to 1024/256
    # img = np.zeros((256, 1024, 3))
    # img[:, :512] = orig_img[:256, :]
    # img[:, 512:] = orig_img[256:, :]
    
    if img is None:
        return np.array(rect).astype(int)
    if rect is None:
        return img
    else:
        return img, rect

def postprocess(boxes, scores, theta=0.5):
    # postprocess boxes and scores:
    # if multiple boxes have an iom >= theta -> consider as the same box and get 
    # expected averaged box out of it
    final_boxes = []
    final_scores = []
    cluster_ids = list(range(len(boxes)))
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            iom = calc_IoM(rect1=boxes[i], rect2=boxes[j])
            if iom >= theta:
                # set cluster id of following point to that of the current pnt
                # this order is correct, as that is already fixed from previous rounds
                cluster_ids[j] = cluster_ids[i]

    cluster_ids = np.array(cluster_ids)
    # for all clusters calculate an average box
    for cluster_id in sorted(set(cluster_ids)):
        # get all indices with that cluster id
        indices = np.where(cluster_ids == cluster_id)[0]
        
        # only average if there are more than one box in that cluster
        if len(indices) > 1:
            new_box = np.sum(boxes[indices]*scores[indices].reshape(len(indices),1), axis=0)/np.sum(scores[indices])
            max_score = np.max(scores[indices])

            # score calculates as follows: max_score + weight*extra, weight = 1-max_score (to stay <= 1)
            # extra = sum(weight_i * score_i) with weight_i = score_i/(sum(scores) - max_score)
            # extra is <= 1 as well, it corresponds to the average of scores without max_score
            new_score = max_score + (1-max_score)*(np.sum(scores[indices]**2) - max_score**2)/(np.sum(scores[indices])-max_score)
            final_boxes.append(new_box.astype(np.int64))
            final_scores.append(new_score)
        else:
            final_boxes.append(boxes[indices[0]])
            final_scores.append(scores[indices[0]])
    return np.array(final_boxes), np.array(final_scores)

def write_to_df(df, img_path, w, h, csv_path, class_label, boxes, scores, thresh=0.0, disable_thresh=False):
    #'filename', 'width', 'height', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'
    # boxes are in format: [y1, x1, y2, x2] between 0 and 1 !!!!
    #print("Boxes: ", boxes, "scoreS: ", scores, "len: ", len(boxes), len(scores))
    dict_list = []
    for i in range(len(boxes)):
        if not disable_thresh and scores[i] < thresh:
            continue
        box = image_decode(rect=boxes[i])
        dict_list.append({'filename': img_path, 'width': w, 'height': h, 'class': 'spine', \
            'score': scores[i], 'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3]})
    if len(dict_list) != 0:    
        df = df.append(dict_list)
    # be aware of windows adding \\ for folders in paths!
    csv_filepath = os.path.join(csv_path, img_path.split('/')[-1].split('\\')[-1][:-4]+'.csv')
    df.to_csv(csv_filepath, index=False)
    return df
    # print("[INFO] Detections saved in csv file "+csv_filepath+".")

def draw_boxes(orig_img, boxes, scores, thresh=0.3, disable_thresh=False):
    img = image_decode(img=orig_img)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        x1, y1, x2, y2 = image_decode(rect=(x1, y1, x2, y2))
        if not disable_thresh and conf < thresh:
            continue
        
        # correct colored rectangle
        # opencv : BGR!!!! NO RGB!!
        # linear from (0,0,255) to (255,255,0)
        #color = (255*(1-conf), 255*conf, 255*conf)
        color = (0, 255, 0)
        text_color = (0, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)
        
        # green filled rectangle for text and adding border as well
        # width of rect depends on width of text
        text_width = 23 if conf < 0.995 else 30
        img = cv2.rectangle(img, (x1, y1), (x1+text_width, y1-12), color, thickness=-1)
        img = cv2.rectangle(img, (x1, y1), (x1+text_width, y1-12), color, thickness=1)
    
        # text
        img = cv2.putText(img, '{:02.0f}%'.format(conf*100), (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
    return img

def df_to_data(df):
    # get rects (boxes+scores) and classes for this specific dataframe
    rects = np.zeros((len(df), 5))
    scores = np.zeros(len(df))

    if len(df) == 0:
        return rects, scores
    #print('df: ', len(df))
    fi = df.first_valid_index()
    w, h = df['width'][fi], df['height'][fi]
    for i in range(len(df)):
        rects[i] = np.array([df['xmin'][fi+i], df['ymin'][fi+i], df['xmax'][fi+i], df['ymax'][fi+i], 1.0])
        classes[i] = 1.0 #df['class'][fi+1]

    return rects, classes

def load_model(path):
    # Load frozen model
    print("[INFO] Loading model ...")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# save_csv flag only False if used in tracking.py!
def predict_images(detection_graph, image_path, output_path, output_csv_path, threshold=0.3, theta=0.5, save_csv=True, return_csv=False):
    data = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    all_boxes, all_scores, all_classes, all_num_detections = [], [], [], []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for img in sorted(glob.glob(image_path)):
                image_np, orig_w, orig_h = image_load_encode(img)
                h, w = image_np.shape[:2]
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                
                #print('Image expanded: ', image_np.shape, image_np_expanded.shape)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                #print('Boxes: ', boxes, 'scores', scores, 'classes', classes, 'num dets', num_detections)

                if return_csv:
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_classes.append(classes)
                    all_num_detections.append(num_detections)

                boxes = boxes[0] * np.array([[512, 512, 512, 512]]); scores = scores[0]; classes = classes[0]; num_detections = int(num_detections[0])

                # convert boxes to xmin, ymin, xmax, ymax. Currently it is ymin, xmin, ymax, xmax
                boxes = boxes[:, [1, 0, 3, 2]]

                # find out where scores are greater than at threshold and change everything according to that
                thresh_indices = np.where(scores >= threshold)[0]
                boxes = boxes[thresh_indices]; scores = scores[thresh_indices]; classes = classes[thresh_indices]
               
                boxes, scores = postprocess(boxes, scores, theta=args.theta)

                # Visualization of the results of a detection, but only if output_path is provided
                if output_path is not None:
                    image_np = draw_boxes(image_np, boxes, scores, disable_thresh=True)     
                    orig_name = img.split('/')[-1].split('\\')[-1]
                    img_output_path = os.path.join(output_path, orig_name) 
                    cv2.imwrite(img_output_path, image_np)

                # always saving data to dataframe
                if save_csv:
                    _ = write_to_df(data, img, orig_w, orig_h, output_csv_path, 'spine', boxes, scores, disable_thresh=True)

                print('[INFO] Finished detection of image '+img+'.')
        
    if return_csv:
        return all_boxes, all_scores, all_classes, all_num_detections

if __name__ == '__main__':
    start = time.time()
    args = parser.parse_args()

    # if it doesn't make sense, print warning
    if args.use_csv is not None and not args.save_images:
        print("[WARNING] As you are using csv files, not saving any detections will result in doing nothing. So images are saved.")
        args.save_images = True

    # save_images true/false, output None/path wo images/csvs
    model_name = args.model.split("/")[-1] if args.model.split("/")[-1] != "" else args.model.split("/")[-2]
    if args.output is None:
        args.output = os.path.join("output/prediction/", model_name)
    output_path = os.path.join(args.output, "images")
    
    # create folder for prediction csvs if not already done
    if not args.use_csv:
        csv_path = os.path.join(args.output, 'csvs')
    else:
        csv_path = args.use_csv
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.save_images and not os.path.exists(output_path):
        os.makedirs(output_path)

    print(output_path, model_name, csv_path)
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = os.path.join('own_models', args.model, 'frozen_inference_graph.pb')

    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = args.labelmap

    NUM_CLASSES = 1

    # Decide whether to predict the bboxes or to load from csv
    if not args.use_csv:
        detection_graph = load_model(PATH_TO_CKPT)
    else:
        print("[INFO] Loading detections from csv file ...")
        df = pd.read_csv(args.model)
    after_loading_model = time.time()
    
    # Load categories
    # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)

    # Make prediction
    nr_imgs = len(list(glob.glob(args.input)))
    print("[INFO] Starting predictions ...")
    if not args.use_csv:
        predict_images(detection_graph, args.input, output_path, csv_path, args.threshold, args.theta)
    else:
        changed_df = False
        for img in glob.glob(args.input):
            image_np, orig_w, orig_h = image_load_encode(img)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            # At first change the folder in front of the filenames to the folder the images are inside
            if not changed_df:
                df['filename'] = [df['filename'][i].split('/')[-1] for i in range(len(df))]
                folder = img.replace(img.split('/')[-1], '')
                df['filename'] = folder+df['filename']
                changed_df = True
            img_df = df[df.filename==img & df.score >= args.threshold]
            
            # Read boxes, classes and scores
            rects, classes = df_to_data(img_df)

            # Visualization of the results of a detection.
            image_np = draw_boxes(image_np, rects)     
            orig_name = os.path.abspath(img).split('/')[-1]
            img_output_path = os.path.join(output_path, orig_name) 
            cv2.imwrite(img_output_path, image_np)            

            # no writing to csv-file necessary as a csv file was already provided

    finished = time.time()
    print(f"Model read in {after_loading_model-start}sec")
    print(f"Predicted {nr_imgs} images in {finished-after_loading_model}sec")
    # with open("log_predict.txt", "w+") as f:
    #     f.write(f"Model read in {after_loading_model-start}sec")
    #     f.write(f"Predicted {nr_imgs} in {finished-after_loading_model}sec -> {(finished-after_loading_model)/nr_imgs}sec per img")
