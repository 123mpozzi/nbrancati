import json
import os

import click
from utils.metrics import *
from utils.predict import method_name
from utils.metrics_utils import calc_metrics, calc_mean_metrics

dump_dir = 'dumps'
dump_filename = os.path.join(dump_dir, 'metrics_{}_{}.json')

@click.group()
def cli_measure():
    pass

@cli_measure.command(short_help='Evaluate skin detector performance')
@click.option('--path', '-p',
              type=click.Path(exists=True), required=True,
              help = 'Path to the folder containing the predictions dir (eg. ECU_on_Schmugge)')
@click.option('--dump/--no-dump', '-d', default=False, help = 'Whether to dump results to files')
def eval(path, dump):
    # Define metric functions used to evaluate
    #metrics = [f1_m, f1fin, f2fin, iou, dprs_m, mcc, recall, precision, specificity]
    metrics = [f1_medium, f1, f2, iou, iou_logical, dprs_medium, dprs, mcc, recall, precision, specificity]

    # Get folders containing grountruth and prediction IMAGES
    y_path = os.path.join(path, 'y') # Path eg. 'predictions/HGR_small_on_ECU/y'
    p_path = os.path.join(path, 'p') # Path eg. 'predictions/HGR_small_on_ECU/p'

    singles = calc_metrics(y_path, p_path, metrics)
    avg = calc_mean_metrics(singles, metrics, desc=path, method=method_name)

    if dump:
        path_bn = os.path.basename(os.path.normpath(path))
        os.makedirs(dump_dir, exist_ok=True)

        with open(dump_filename.format(path_bn, 'average'), 'w') as f:
            json.dump(avg, f, sort_keys = True, indent = 4)
        with open(dump_filename.format(path_bn, 'singles'), 'w') as f:
            json.dump(singles, f, sort_keys = True, indent = 4)
