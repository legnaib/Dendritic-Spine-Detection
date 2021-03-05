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

## Folder structure
This github repository provides all necessary files to predict and track dendritic spines as described in the paper TODO. Retraining on another dataset is possible as well. The mainly relevant files and structures of this repository are:
```
|-- config_files
|   |-- custom_model.config
|   `-- default_model.config
|-- convert_data
|   `-- generate_tfrecord.py
|-- data
|   |-- raw
|   `-- spine_label_map.pbtxt
|-- models
|   `-- research
|       |-- slim
|       `-- object_detection
|           |-- export_inference_graph.py
|           `-- model_main.py
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
|-- CentroidTracker.py
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
TODO.
After saving all data in csv files, `.tfrecord` files must be created. The necessary files `train.csv`, `valid.csv` and `test.csv` are saved in the `data/PATH_TO_CSV` folder and the images the csv-files are referring to are inside  the folder `data/PATH_TO_IMAGES`. The conversion can be done by executing the following command three times and each time replacing `FILENAME.csv` with either `train.csv`, `valid.csv` or `test.csv`:
```
python convert_data/generate_tfrecord.py --csv_input=data/PATH_TO_CSV/FILENAME.csv --output_path=data/PATH_TO_CSV/FILENAME.record --image_dir=data/PATH_TO_IMAGES
```

### Training
### Prepare model for inference
Before being able to use inference and let the model predict spines as described in previous sections, the model needs to be converted to a frozen inference graph. After training the model for `NR_STEPS` amount of steps, the folder structure should look like this:
```
`-- own_models
    `-- MODEL_NAME
        |-- checkpoint
        |-- graph.pbtxt
        |-- model.ckpt-NR_STEPS.data-00000-of-00001
        |-- model.ckpt-NR_STEPS.index
        |-- model.ckpt-NR_STEPS.meta
```
For inference the `frozen_inference_graph.pb` is needed. This can be obtained by calling
```
python models/research/object_detection/export_inference_graph.py --pipeline_config_path=config_files/MODEL_NAME.config --trained_checkpoint_prefix=own_models/MODEL_NAME/model.ckpt-NR_STEPS --output_directory=own_models/MODEL_NAME
```

## Model evaluation