import pandas as pd
import argparse
import os
import glob
import random

parser = argparse.ArgumentParser(description="Convert one csv file to train, valid and test csv file")
parser.add_argument('-i', '--input', \
    help="Path to original csv file", default="data/annotations/")
parser.add_argument('-o', '--output', \
    help="Path to output csv files train, valid, test", default="data/annotations")
parser.add_argument('-p', '--probabilities', \
    help="Comma separated values for the probabilities", default='0.8,0.1,0.1')
parser.add_argument('-n', '--names', \
    help="Names of final files. If sum of probabilities > 1 all files will be created independently", default='train,valid,test')

if __name__ == "__main__":
    args = parser.parse_args()
    
    names = args.names.split(',')
    probs = list(map(float, args.probabilities.split(',')))

    if len(names) != len(probs):
        raise ValueError('Number of files '+len(names)+' does not match with number of probabilites '+len(probs)+'.')
    
    nr_files = len(names)
    # Load all csv files
    #total_data = [[], [], []] # train, valid, test
    total_data = [[] for i in range(len(names))]

    # probabilities for [train, valid] and test set
    #probs = [0.8, 0.1]
    
    df = pd.read_csv(args.input)
    imgs = df.groupby('filename')

    # Look if you want to divide or independently collect data
    if round(sum(probs), 6) == 1.0:
        # One image shouldn't be split into two different parts
        for img in imgs:
            data = img[1]
            rndm = random.random()
            if rndm < probs[0]:
                total_data[0].append(data)
            elif rndm < probs[0]+probs[1]:
                total_data[1].append(data)
            else:
                total_data[2].append(data)
    else:
        # One image shouldn't be split into two different parts
        for img in imgs:
            data = img[1]
            for i in range(nr_files):
                rndm = random.random()
                if rndm < probs[i]:
                    total_data[i].append(data)

    
    # Concatenate and write
    concat_data = [pd.concat(total_data[i], ignore_index=True) for i in range(nr_files)]
    
    #train_data = pd.concat(total_data[0], ignore_index=True)
    #val_data = pd.concat(total_data[1], ignore_index=True)
    #test_data = pd.concat(total_data[2], ignore_index=True)

    #total_data = pd.concat(concat_data, ignore_index=True)
    #total_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

    for i in range(nr_files):
        concat_data[i].to_csv(args.output+names[i]+'.csv', index=False)
    #train_data.to_csv(args.output+'train.csv', index=False)
    #val_data.to_csv(args.output+'valid.csv', index=False)
    #test_data.to_csv(args.output+'test.csv', index=False)

    print("[INFO] Csv file split into "+args.output+"-"+str(names)+".csv.")