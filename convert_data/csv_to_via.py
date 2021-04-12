import pandas as pd
import argparse
import numpy as np
import glob
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Convert normal csv files to format used for VIA Annotation Tool")

parser.add_argument('-i', '--input',
    help='path to input csv files', default='*.csv')
parser.add_argument('-o', '--output',
    help='path to output file')

def csv_to_via(input_files, output_path):
    print("[INFO] Preparing data...")
    all_data = []
    for f in glob.glob(args.input):
        all_data.append(pd.read_csv(f))
    
    print("[INFO] All data is ready for conversion.")
    # total data saved in df and sort by filename, so that region_id and region_count work properly
    df = pd.concat(all_data)
    df.sort_values(by=['filename'], inplace=True)

    # add the easy data
    df['file_attributes'] = '{}'
    df['region_attributes'] = '{}'
    
    # convert filename that it works on my windows:
    #own_models/labeling/extra_images_1
    df['file_size'] = list(map(os.path.getsize, df['filename']))
    df['filename'] = [df['filename'].iloc[i].split('/')[-1] for i in range(len(df))]
    df['region_count'] = 0
    df['region_id'] = 0
    df['region_shape_attributes'] = ''
    #print(df[['filename', 'file_size']])

    print("[INFO] Minor conversions done.")
    # set region_count, region_id and boxes
    curr_file = ''
    curr_nr = 0
    start_index = 0
    region_id = [0 for i in range(len(df))]
    region_count = [0 for i in range(len(df))]
    region_shape_attributes = ['' for i in range(len(df))]
    for i in range(len(df)):
        # file is named before -> region id is higher
        if curr_file == df['filename'].iloc[i]:
            curr_nr += 1
            region_id[i] = curr_nr
        else:
            # set region_count of all files before correctly
            for j in range(start_index, i):
                region_count[j] = curr_nr+1

            curr_file = df['filename'].iloc[i]
            region_id[i] = 0
            start_index = i
            curr_nr = 0

        # set boxes
        x, y = df['xmin'].iloc[i], df['ymin'].iloc[i]
        w, h = df['xmax'].iloc[i]-x, df['ymax'].iloc[i]-y
        region_shape_attributes[i] = \
            '{"name":"rect","x":'+str(x)+',"y":'+str(y)+',"width":'+str(w)+',"height":'+str(h)+'}'
    
    df['region_count'] = region_count
    df['region_id'] = region_id
    df['region_shape_attributes'] = region_shape_attributes
    # finally set filename correctly
    df['filename'] = list(map(lambda x: x.split('/')[-1], df['filename']))#.iloc[i].split('/')[-1]

    print("[INFO] Almost completely converted.")
    # get columns in correct order
    df = df[['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']]
    df.to_csv(output_path, index=False)
    print("[INFO] Csv files are converted into "+output_path+".")

if __name__ == '__main__':
    args = parser.parse_args()

    csv_to_via(args.input, args.output)
    # Column order VIA:
    # filename, file_size, file_attributes, region_count, region_id, region_shape_attributes, region_attributes
    # where:
    # filename = only name of file (not whole path)
    # file_attributes: "{}", region_attributes: "{}"
    # region_count: nr of boxes in one image
    # region_id: id of box in same image (starting with 0)
    # region_shape_attributes: "{""name"":""rect"",""x"":nr,""y"":nr,""width"":nr,""height"":nr}"

    # orig file:
    # filename,width,height,class,xmin,ymin,xmax,ymax (width, height of image)
    # filename: whole path of image