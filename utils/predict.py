import os
import sys
import time
import traceback
from shutil import copyfile

from utils.logmanager import *

method_name = 'rulebased'
predictions_dir = 'predictions'

def get_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def pred_name(name: str) -> str:
    return name.lower().replace('hgr_small', 'hgr')

def pred_dir(type: str, timestr: str, name: str) -> str:
    '''
    Return a proper directory to store predictions given
    prediction type, timestamp string, and predictions name/title
    '''
    name = pred_name(name)
    if type == 'base':
        return os.path.join(predictions_dir, timestr, method_name, type, name)
    elif type == 'bench':
        return os.path.join(predictions_dir, type, timestr, name)
    else: # default
        return os.path.join(predictions_dir, name)

def pred_out(path_x: str, out_dir: str) -> list:
    # use the x filename for all saved images filenames (x, y, p)
    filename, x_ext = os.path.splitext(os.path.basename(path_x))
    # the masks and predictions will be saved LOSSLESS as PNG
    out_p = os.path.join(out_dir, 'p', filename + '.png')
    out_y = os.path.join(out_dir, 'y', filename + '.png')
    out_x = os.path.join(out_dir, 'x', filename + x_ext)
    return (out_p, out_y, out_x)

# out_bench is the file in which append inference performance data
def predict(path_x, path_y, out_dir, out_bench: str = ''):
    '''
    Create a single prediction image

    Also copy the original image and grountruth

    #### No-grountruth image prediction
    In case of a prediction over a single image which has no groundtruth, hence
    is not a dataset image, the behaviour is a little different:

    `path_y` will be None

    `out_dir` is the prediction output filename

    Original image and grountruth files are not copied
    '''
    if path_y is not None:
        out_p, out_y, out_x = pred_out(path_x, out_dir)
    else:
        out_p = out_dir

    # Save p
    run_skin_detector(path_x, out_p, out_bench)

    # Copy x and y
    if path_y is not None: # if its a dataset image (has a groudntruth)
        copyfile(path_x, out_x)
        copyfile(path_y, out_y)

def run_skin_detector(path_in, path_out, bench_out = None):
    '''Print commands that will be executed via pipe'''
    if bench_out == None:
        command = f'./app {path_in} {path_out}'
    else:
        command = f'./app {path_in} {path_out} {bench_out}'
    
    print(command)

def make_predictions(image_paths, out_dir, out_bench: str = ''):
    '''Detect skin pixels over a list of images'''
    info('Data collection completed')

    # make dirs
    for basedir in ('p', 'y', 'x'):
        os.makedirs(os.path.join(out_dir, basedir), exist_ok=True)
    
    for i in image_paths:
        im_path = i[0]
        y_path = i[1]

        # Try predicting
        try:
            predict(im_path, y_path, out_dir, out_bench)
        # File not found, prediction algo fail, ..
        except Exception:
            error(f'Failed to infer on image: {im_path}')
            print(traceback.format_exc(), file=sys.stderr)

def base_preds(timestr: str, datasets: list, only_test_set: bool = True):
    '''
    Base predictions
    By default, predictions are performed on the test set of each dataset.
    To target whole datasets, set `only_test_set` to False
    '''
    # Iterate each dataset
    for predict_db in datasets:
        assert os.path.isdir(predict_db.dir), 'Dataset has no directory: ' + predict_db.name

        # Make predictions
        if only_test_set:
            image_paths = predict_db.get_test_paths() # predict on testing set
            out_dir = pred_dir('base', timestr, predict_db.name)
        else:
            image_paths = predict_db.get_all_paths() # predict the whole dataset
            out_dir = pred_dir('base', timestr, predict_db.name + '_all')
        make_predictions(image_paths, out_dir)
