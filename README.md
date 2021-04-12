# Dendritic-Spine-Detection
Code for detecting dendritic spines in three dimensions, retraining and evaluating models.

Structure of this guide:
1. Installation
2. Folder structure
3. Prediction on 2D-images
4. Prediction and tracking on 3D-images
   - File format for prediction and tracking csv
5. Re-Training with new dataset
   - Prepare dataset
   - Training
   - Prepare model for inference
6. Model evaluation

## Installation
All necessary packages are listed in the `requirements.txt` file. To install packages with pip simply run
```
pip install -r requirements.txt
```
The model, training and evaluation images are not saved in GitHub, but can be downloaded [here](https://drive.google.com/uc?export=download&id=1r9bGOeOxEd6Clg8Sw5ajcxLO-C_xC9K8). The model and images should then be extracted into the `own_models/default_model` and `data/raw` folder. Training and tracking evaluation are also available as shell scripts with some predefined arguments.

## Folder structure
This github repository provides all necessary files to predict and track dendritic spines as described in the paper TODO. Retraining on another dataset is possible as well. The mainly relevant files and structures of this repository are:
```
|-- config_files
|   |-- custom_model.config
|   `-- default_model.config
|-- convert_data
|   |-- via.html
|   |-- via_to_csv.py
|   |-- split_csv_train_valid_test.py
|   `-- generate_tfrecord.py
|-- data
|   |-- raw
|   `-- spine_label_map.pbtxt
|-- models
|   `-- research
|       |-- slim
|       `-- object_detection
|           `-- export_inference_graph.py
|-- output
|   |-- prediction
|   |   |-- custom_model
|   |   |   |-- images
|   |   |   `-- csvs
|   |   `-- default_model
|   |       |-- images
|   |       `-- csvs
|   `-- tracking
|       |-- custom_model
|       |   |-- images
|       |   `-- data_tracking.csv
|       `-- default_model
|           |-- images
|           `-- data_tracking.csv
|-- own_models
|   |-- custom_model
|   |   `-- frozen_inference_graph.pbtxt
|   `-- default_model
|       `-- frozen_inference_graph.pbtxt
|-- utils.py
|-- train.py
|-- predict.py
|-- requirements.py
`-- tracking.py
```
The `default_model` folders and files do already exist, this is the model reaching human performance. Other retrained models can be added as well, marked with `custom_model`.

## Prediction on 2D-images
Before predicting on 2D-images, the images must be converted into the correct format. A few conversion scripts can be found in the `convert_data/` directory. Making sure that our model performs best, the images should be in `.png` Format and should have a size of 512x512 pixel.

The name of the model which the user wants to use is referred to as `MODEL_NAME` for the following paragraphs. Under `own_models/MODEL_NAME` the `frozen_inference_graph.pb` of this specific model must be saved. By default the `default_model` is used.

An example prediction will look like this:
```
python predict.py --model=MODEL_NAME --input="data/raw_images/SR052*.png" --save_images
```
All images with their predictions will be saved in the folder `output/prediction/MODEL_NAME/images` and all csv files will be saved in `output/prediction/MODEL_NAME/csvs`.


## Prediction and tracking on 3D-images
If a 3D-stack of images should be analyzed there are two possibilities to do that:
1. Predict and track everything in one single command:
    ```
    python tracking.py --model=MODEL_NAME --images="data/raw_images/SR052*.png" --save_images
    ```
2. Predict first or choose different prediction files and use the tracking algorithm to get total 3D-trajectories. Two commands must be executed for that:
    ```
    python predict.py --model=MODEL_NAME --input="data/raw_images/SR052*.png"

    python tracking.py --model=MODEL_NAME --images="data/raw_images/SR052*.png" --csv="output/prediction/MODEL_NAME/csvs/*.csv" --save_images
    ```
    Every csv-file in a valid format can be used at the `--csv`-flag to determine which predictions the tracking algorithm should take.

Default settings will save all images in the folder `output/tracking/MODEL_NAME/images` together with the single tracking file `output/tracking/MODEL_NAME/data_tracking.csv`. Only the images resulting of the tracking algorithm are of interest, therefore the `--save_images` flag is removed from the prediction part.

### File format for prediction and tracking csv
The prediction csv files are named exactly as the images they are saving the detections for. Column names are `filename,width,height,class,score,xmin,ymin,xmax,ymax`:

- `filename`: relative path of image the detection is made on
- `width,height`: width and heigh of this image
- `class`: as only one class is detected, this should always be 'spine'
- `score`: detection confidence
- `xmin,ymin,xmax,ymax`: top left and bottom right corner of the bounding box

The tracking file is built very similar. The only change is an additional column at the beginning, named `id`:
- `id`: Spine-ID in the 3D-stack

## Re-Training with new dataset
### Prepare dataset
The labeling can be done with the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) using the provided `via.html` script. Already created labels can be load using the menu button `Annotations - Import Annotations (from csv)`. If the labels are not in via-format, but already in the format used for training, these files can be converted using the `convert_data/via_to_csv.py` script.

**Attention**: To load the images correctly, the `Default Path` in the settings menu has to be set to the path where the images are saved.

After creating all labels, the data has first to be converted to the correct format with `convert_data/via_to_csv.py` and split into train, validation and test datasets using `convert_data/split_csv_train_valid_test.py`.

After saving all data in csv files, `.tfrecord` files must be created. The necessary files `train.csv`, `valid.csv` and `test.csv` are saved in the `data/PATH_TO_CSV` folder and the images the csv-files are referring to are inside  the folder `data/PATH_TO_IMAGES`. The conversion can be done by executing the following command three times and each time replacing `FILENAME.csv` with either `train.csv`, `valid.csv` or `test.csv`:
```
python convert_data/generate_tfrecord.py --csv_input=data/PATH_TO_CSV/FILENAME.csv --output_path=data/PATH_TO_CSV/FILENAME.record --image_dir=data/PATH_TO_IMAGES
```

### Training
Given a config file `CONFIG_NAME.config`, the output training folder `TRAINING_NAME` and the number of steps `NR_STEPS` that should be trained for, training can be started by executing the following command:
```
python train.py --model_dir=own_models/TRAINING_NAME --pipeline_config_path=config_files/CONFIG_NAME.config --num_train_steps=NR_STEPS --alsologtostderr
```
For a better overview of config files and training folders it is recommended to use the same name: `TRAINING_NAME=CONFIG_NAME`. The original script can be found in `model/research/object_detection` and is named `model_main.py`.

### Prepare model for inference
Before being able to use inference and let the model predict spines as described in previous sections, the model needs to be converted to a frozen inference graph. After training the model for `NR_STEPS` amount of steps, the folder structure should look like this:
```
`-- own_models
    `-- MODEL_NAME
        |-- checkpoint
        |-- graph.pbtxt
        |-- model.ckpt-NR_STEPS.data-00000-of-00001
        |-- model.ckpt-NR_STEPS.index
        `-- model.ckpt-NR_STEPS.meta
```
For inference the `frozen_inference_graph.pb` is needed. This can be obtained by calling
```
python models/research/object_detection/export_inference_graph.py --pipeline_config_path=config_files/MODEL_NAME.config --trained_checkpoint_prefix=own_models/MODEL_NAME/model.ckpt-NR_STEPS --output_directory=own_models/MODEL_NAME
```

## Model evaluation
After using the model `MODEL_NAME` together with the tracker to create a `DATA_TRACKING_FILE` one can compare the results to a given groundtruth example by using the following command:
```
python evaluate_tracking.py --detfolder output/tracking/MODEL_NAME --gtfolder output/tracking/GT/data_tracking.csv --tracking DATA_TRACKING_FILE.csv --savename SAVE_NAME
```
The output csv-file will be saved under `results/SAVE_NAME.csv`. Its format has colum names `timestamp,detection_threshold,fscore,precision,recall,nr_detected,nr_gt,nr_gt_detected`:

- `timestamp`: Timestamp of model evaluation
- `detection_threshold`: Reminder of the used detection threshold. Default set to 0.5 and has to be adjusted manually according to the used detection threshold while tracking
- `fscore,precision,recall`: ![formula](https://render.githubusercontent.com/render/math?math=F^{3D}_1)-score, Precision and Recall of the 3D detected spines using the ![formula](https://render.githubusercontent.com/render/math?math=IoM)
- `nr_detected,nr_gt,nr_gt_detected`: number of spines only detected by model, only detected by groundtruth or detected by both.

<!-- - <img src="https://latex.codecogs.com/gif.latex?F^{3D}_1 " /> -->

<!-- ![equation](https://latex.codecogs.com/gif.latex?F^{3D}_1) -->
