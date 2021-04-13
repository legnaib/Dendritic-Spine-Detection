export PYTHONPATH=$PYTHONPATH:/models/research/slim:/models/research:/models/research/object_detection
export PYTHONPATH=$PYTHONPATH:models/research/slim:models/research:models/research/object_detection
for NAME in train valid test
do
    python3 convert_data/generate_tfrecord.py --csv_input data/default_annotations/$NAME.csv --output_path data/default_annotations/$NAME.record --image_dir .
done