import os, re, sys, json, subprocess
import cv2
import imghdr
import numpy as np
from random import shuffle
from math import floor


### ISTRUZIONI:
# fai partire il venv datloader nella cartella workspace
# (cd in cartella dyc)
# (fai partire docker con up)
# (muovi i dataset nella cartella dyc/dataset se non ci sono)
# poi vai in cartella dyc e fai partire dyc_batch_py.py
###

def run_dyc(path_in, path_out):
    command = f'docker-compose exec opencv ./app {path_in} {path_out}'
    #print(command)
    process = subprocess.run(command.split())

# remember that Pratheepan dataset has one file with comma in the filename
csv_sep = '?'

#datas = ['dataset/Pratheepan', 'dataset/ECU', 'dataset/HGR_small', 'dataset/HGR_big',
#        'dataset/Uchile', 'dataset/Schmugge', 'dataset/abd-skin', 'dataset/VDM']

datas = ['dataset/ECU', 'dataset/HGR_small', 'dataset/Schmugge']

for ds in datas:
    csv_file = os.path.join(ds, 'data.csv')
    ds_name = os.path.basename(ds).lower()
    pred_dir = os.path.join('predicted', ds_name, 'p')
    y_dir = os.path.join('predicted', ds_name, 'y')
    #pred_dir = os.path.join('predicted', ds.split('/')[1])
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)

    # read csv lines
    file3c = open(csv_file)
    triples = file3c.read().splitlines()
    file3c.close()

    # check if there is a test set
    has_test = False
    for entry in triples: # oriname.ext, gtname.ext
        #ori_path = entry.split(csv_sep)[0]
        #gt_path = entry.split(csv_sep)[1]
        note = entry.split(csv_sep)[2]
        #ori_name, ori_ext = os.path.splitext(os.path.basename(ori_path))
        #gt_name, gt_ext = os.path.splitext(os.path.basename(gt_path))
        if note == 'te':
            has_test = True
    
    if has_test: # predict only test images
        for entry in triples: # oriname.ext, gtname.ext
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note = entry.split(csv_sep)[2]
            ori_name, ori_ext = os.path.splitext(os.path.basename(ori_path))
            pred_path = os.path.join(pred_dir, ori_name + '.png')

            if note == 'te':
                #save y
                im_y = cv2.imread(gt_path)
                y_path = os.path.join(y_dir, ori_name + '.png')
                cv2.imwrite(y_path, im_y)
                # predict
                run_dyc(ori_path, pred_path)
    else: # predict all dataset
        for entry in triples: # oriname.ext, gtname.ext
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            ori_name, ori_ext = os.path.splitext(os.path.basename(ori_path))
            pred_path = os.path.join(pred_dir, ori_name + '.png')
            #save y
            im_y = cv2.imread(gt_path)
            y_path = os.path.join(y_dir, ori_name + '.png')
            cv2.imwrite(y_path, im_y)
            # predict
            run_dyc(ori_path, pred_path)
