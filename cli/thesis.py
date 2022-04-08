import os
import subprocess

import click
from utils.predict import get_timestamp, method_name, predictions_dir
from utils.db_utils import get_datasets
from utils.ECU import ECU
from utils.HGR import HGR
from utils.Schmugge import Schmugge, dark, light, medium

# Run the necessary predictions to get tables featured in the thesis


def gen_cmd(skintones: bool, timestr: str) -> str:
    targets_args = '-t dark -t medium -t light' if skintones else '-t ECU -t HGR_small -t Schmugge'
    cmd = 'batch'

    method = method_name + '_st' if skintones else method_name
    out_dir = os.path.join(predictions_dir, timestr, method, 'base')

    return 'python main.py {} {} -o {}'.format(cmd, targets_args, out_dir)

@click.group()
def cli_thesis():
    pass

@cli_thesis.command(name='thesis', short_help='Reproduce tables featured in the thesis')
def thesis():
    targets = ECU(), HGR(), Schmugge(), dark(), medium(), light()

    for t in targets:
        assert t in get_datasets(), f'Necessary dataset not found: {t}'

    # Reset datasets with predefined splits
    #for t in targets:
    #    t.reset(predefined=True)

    timestr = get_timestamp()
    # Call each command synchronously with subprocess.call(), wait for it to end
    commands = []
    for pred_type in [True, False]:
        commands.append(gen_cmd(skintones=pred_type, timestr=timestr))
    
    for cmd in commands:
        subprocess.call(cmd.split(), shell=True, stdin=None, stdout=None, stderr=None)
