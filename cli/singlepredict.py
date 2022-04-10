import os
import time

import click
from utils.db_utils import *
from utils.ECU import ECU_bench
from utils.logmanager import *
from utils.metrics_utils import read_performance
from utils.predict import (base_preds, get_timestamp, make_predictions,
                           pred_dir, pred_name, predict)


@click.group()
def cli_predict():
    pass

@cli_predict.command(short_help='Skin detection over N datasets')
@click.option('--target' , '-t',  multiple=True, required = True,
              type=click.Choice(skin_databases_names(get_datasets()), case_sensitive=False),
              help = 'Datasets to use (eg. -t ECU -t HGR_small -t medium)')
@click.option('--whole/--no-whole', '-w', 'whole', default=False,
              help = 'Whether to predict only on test set or whole dataset')
def batch(target, whole):
    '''BATCH: skin detection over N datasets'''
    timestr = get_timestamp()
    targets = [get_db_by_name(x) for x in target]
    #targets = skin_databases_names(targets)

    if whole:
        base_preds(timestr, targets, only_test_set=False)
    else:
        base_preds(timestr, targets)

@cli_predict.command(short_help='Measure inference time')
@click.option('--size', '-s', type=int, default = 15, show_default=True,
              help='Benchmark set size, in images (-1 is whole db)')
@click.option('--observations', '-o', type=int, default = 5, show_default=True,
              help='Observations to register for the benchmark set')
def bench(size, observations):
    '''BENCHMARK: measure inference time'''
    timestr = get_timestamp()

    # Use first 15 ECU images as test set        
    ECU_bench().reset(amount=size)
    image_paths = ECU_bench().get_test_paths()
    out_dir = pred_dir('bench', timestr, 'observation{}')

    # Do multiple observations
    # The predictions will be the same but performance will be logged 5 different times
    for k in range(observations):
        assert os.path.isdir(ECU_bench().dir), 'Dataset has no directory: ' + ECU_bench().name
        make_predictions(image_paths, out_dir.format(k),
            out_bench=os.path.join(out_dir.format(k), '..', f'bench{k}.txt'))
    
    time.sleep(5) # wait for predictions to complete

    # Print inference times
    read_performance(os.path.join(out_dir.format(0), '..'))


@cli_predict.command(short_help='Skin detection over 1 dataset')
@click.option('--target', '-t', required=True,
              type=click.Choice(skin_databases_names(get_datasets()), case_sensitive=False))
@click.option('--whole/--no-whole', '-w', 'whole', default=False,
              help = 'Whether to predict only on test set or whole dataset')
@click.option('--output', '-o', default = '',
              type=click.Path(exists=False),
              help = 'Define the directory in which to save predictions')
def single(target, whole, output):
    '''SINGLE: skin detection over 1 dataset'''
    target_dataset = get_db_by_name(target)
    if whole:
        image_paths = target_dataset.get_all_paths()
        out_foldername = target + '_all'
    else:
        image_paths = target_dataset.get_test_paths()
        out_foldername = target

    assert os.path.isdir(target_dataset.dir), 'Dataset has no directory: ' + target_dataset.name
    # Make predictions
    if output == '':
        out_dir = pred_dir(None, None, name = out_foldername)
    else:
        name = pred_name(out_foldername)
        out_dir = os.path.join(output, name)
        os.makedirs(output, exist_ok=True)
    make_predictions(image_paths, out_dir)

@cli_predict.command(
    short_help='Skin detection over a single image')
@click.option('--path', '-p',
              type=click.Path(exists=True), required=True,
              help = 'Path to the image to predict on')
def image(path):
    '''
    IMAGE: skin detection over a single image.
    Image may not have a grountruth.
    Result will be placed in the same directory as the input image.
    '''
    ori_name = os.path.basename(path)
    ori_filename, _ = os.path.splitext(ori_name)

    im_dir = os.path.dirname(path)
    p_out = os.path.join(im_dir, ori_filename + '_p.png')

    assert os.path.isfile(path), 'Image file not existing: ' + path
    # Make predictions
    predict(path, None, p_out)
