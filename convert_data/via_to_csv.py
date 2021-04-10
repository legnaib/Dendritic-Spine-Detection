import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, \
    description='Convert the downloaded via_region_data.csv to csv file for NNs')

parser.add_argument('-i', '--input', \
    help='path to input via_region_data.csv file', default='data/raw/via_region_data.csv')
parser.add_argument('-o', '--output', \
    help='path to output file', default='data/annotations/data.csv')

if __name__ == '__main__':
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Delete all parts with no annotations and some unimportant columns
    df = df[df['region_count'] != 0]
    df.drop(columns=['file_size', 'file_attributes', 'region_count', 'region_id', 'region_attributes'], inplace=True)

    # Create x1, y1, x2, y2, class_name columns
    df['x1'] = 0; df['y1'] = 0; df['x2'] = 0; df['y2'] = 0; df['class_name'] = 'spine'

    # Convert the region_shape_attributes correctly into x1, y1, x2, y2 values
    # QUESTION: IS X, Y MIDDLEPOINT OF RECT OR UPPER LEFT?
    df['region_shape_attributes'] = df['region_shape_attributes'].apply(eval)
    df['x1'] = df['region_shape_attributes'].apply(lambda d: d['x'])
    df['y1'] = df['region_shape_attributes'].apply(lambda d: d['y'])
    df['x2'] = df['region_shape_attributes'].apply(lambda d: d['width']) + df['x1']
    df['y2'] = df['region_shape_attributes'].apply(lambda d: d['height']) + df['y1']
    df['filename'] = '/home/fvogel/Documents/data/person5/'+df['filename']

    # Delete the last column
    df.drop(columns=['region_shape_attributes'], inplace=True)

    df.to_csv(args.output, index=False)
    print('[INFO] csv file ' +args.input+ ' correctly formatted to ' +args.output+'.')
    