import os, time, subprocess, sys
import cv2
#from tqdm import tqdm
from shutil import copyfile


### ISTRUZIONI:
# fai partire il venv datloader nella cartella workspace
# (cd in cartella dyc)
# (fai partire docker con up)
# (muovi i dataset nella cartella dyc/dataset se non ci sono)
# poi vai in cartella dyc e fai partire dyc_batch_py.py <mode>
###

# remember that Pratheepan dataset has one file with comma in the filename
csv_sep = '?'

def run_dyc(path_in, path_out):
    command = f'docker-compose exec opencv ./app {path_in} {path_out}'
    #print(command)
    process = subprocess.run(command.split())

def get_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

# load a schmugge skintone split by replacing the data.csv file
def load_skintone_split(skintone):
    os.remove('./dataset/Schmugge/data.csv')

    if skintone == 'light':
        print(f'loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/light2305_1420.csv', './dataset/Schmugge/data.csv')
    elif skintone == 'medium':
        print(f'loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/medium2305_1323.csv', './dataset/Schmugge/data.csv')
    elif skintone == 'dark':
        print(f'loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/dark2305_1309.csv', './dataset/Schmugge/data.csv')
    else:
        print(f'skintone type invalid: {skintone}')



if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    if n == 2: # predict over the same dataset of the model
        mode = sys.argv[1] # first argument, argv[0] is the name of the script
    else:
        exit('''There must be 1 argument!
        Usage: python dyc_batch_py.py <mode>
        mode can be: normal or skintones''')

    #datas = ['dataset/Pratheepan', 'dataset/ECU', 'dataset/HGR_small', 'dataset/HGR_big',
    #        'dataset/Uchile', 'dataset/Schmugge', 'dataset/abd-skin', 'dataset/VDM']

    #mode = 'normal' # can be: 'normal' or 'skintones'

    if mode == 'normal':
        datas = ['dataset/ECU', 'dataset/HGR_small', 'dataset/Schmugge']
    elif mode == 'skintones':
        datas = ['dark', 'medium', 'light']
    else:
        exit('Invalid mode! Possible values are: normal, skintones')
    
    timestr = get_timestamp()

    for ds in datas:
        if mode == 'skintones':
            ds_name = ds
            ds = f'./dataset/Schmugge'
        else:
            ds_name = os.path.basename(ds).lower()

        csv_file = os.path.join(ds, 'data.csv')
        #ds_name = os.path.basename(ds).lower()

        if mode == 'skintones':
            load_skintone_split(ds_name)

        # prepare directories
        out_dir = f'./predictions/{timestr}/dyc/base/{ds_name}'
        pred_dir = f'{out_dir}/p'
        x_dir = f'{out_dir}/x'
        y_dir = f'{out_dir}/y'
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(x_dir, exist_ok=True)
        os.makedirs(y_dir, exist_ok=True)

        # read csv lines
        file3c = open(csv_file)
        triples = file3c.read().splitlines()
        file3c.close()

        # check if there is a test set
        has_test = False
        for entry in triples: # oriname.ext, gtname.ext
            note = entry.split(csv_sep)[2]
            if note == 'te':
                has_test = True
                break
        
        #for entry in tqdm(triples): # oriname.ext, gtname.ext
        for entry in triples: # oriname.ext, gtname.ext
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note = entry.split(csv_sep)[2]
            ori_name, ori_ext = os.path.splitext(os.path.basename(ori_path))
            pred_path = os.path.join(pred_dir, ori_name + '.png')

            # predict all dataset if there is not a test split
            if has_test == False:
                note = 'te'
            
            # predict only test images
            if note == 'te':
                #save x
                im_x = cv2.imread(ori_path)
                x_path = os.path.join(x_dir, ori_name + ori_ext)
                cv2.imwrite(x_path, im_x)
                #save y
                im_y = cv2.imread(gt_path)
                y_path = os.path.join(y_dir, ori_name + '.png')
                cv2.imwrite(y_path, im_y)
                # predict
                run_dyc(ori_path, pred_path)
