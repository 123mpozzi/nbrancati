import os
import unittest

from cli.singlepredict import batch, single
from click.testing import CliRunner
from utils.predict import predictions_dir
from utils.db_utils import gen_pred_folders, get_db_by_name
from utils.hash_utils import hash_dir
from utils.logmanager import *
from utils.Schmugge import light, medium

from tests.helper import search_subdir, set_working_dir, rm_folder

# xxh3_64 hashes of prediction folders already generated for thesis
hashes = {
    # base
    'ECU_on_ECU' : 'e1dc03ee2bfbb903',
    'HGR_small_on_HGR_small' : '40752ebb0f0b4410',
    'Schmugge_on_Schmugge' : '860c1665a6a03c70',
    # skintone base
    'dark_on_dark' : 'f1db4259767f19ed',
    'light_on_light' : 'adb974e9c9a49eb9',
    'medium_on_medium' : 'bf0d93cbc416c526',
}

class TestMultipredict(unittest.TestCase):
    '''Functional testing for multipredict commands'''


    def check_predictions_folders(self, predictions: list):
        '''
        Resulting predictions have same hashes as the ones registered in the thesis,
        for datasets featured in it

        Prediction images are the same number as in the csv file
        '''
        # Check if can find all predictions folders
        folder_matches = {}
        for pred in predictions:
            match_ = search_subdir(predictions_dir, pred)
            # Assert predictions folder exists
            self.assertIsNotNone(match_, f'No match found for prediction folder named {pred}')
            folder_matches[pred] = match_

        for pred in predictions:
            match = folder_matches[pred]

            info('Testing for ' + pred)
            # for datasets featured in thesis
            if pred in hashes:
                match_hash = hash_dir(match)
                info(f'{match_hash} - {hashes[pred]}')
                # Resulting predictions have same hashes
                self.assertEqual(match_hash, hashes[pred])
                info('Hash corresponding for ' + pred)
            
            #  Predictions folder it has same number of files as defined in csv
            target = pred.split('_on_')[1]
            target = get_db_by_name(target)
            images_to_predict = len(target.get_test_paths())
            images_predicted = len(os.listdir(os.path.join(match, 'p'))) # images in prediction dir
            self.assertEqual(images_predicted, images_to_predict,
                f'Number of images predicted != number of images to predict: {images_predicted} != {images_to_predict}')

    def test_batchm(self):
        '''
        Command run without errors

        Resulting predictions have same hashes as the ones registered in the thesis,
        for datasets featured in it.

        Prediction images are the same number as in the csv file
        '''
        set_working_dir(self)

        runner = CliRunner()

        # NOTE: uncomment to test all datasets (longer time)
        #targets = get_datasets()
        targets = [medium(), light()]
        predictions = gen_pred_folders(targets)
        info('predictions folder to check for:')
        for pred in predictions:
            info(pred)
            # remove previous predictions
            rm_folder(os.path.join(predictions_dir, pred))

        datasets_args = []
        for db in targets: # get models
            datasets_args.append('-t')
            datasets_args.append(db.name)
            # NOTE: reset or else hashes may not be the same if randomized
            db.reset(predefined=True)

        info('TESTING BATCHM COMMAND...')
        # run command
        result = runner.invoke(batch, datasets_args)
        # Command has no errors on run
        self.assertEqual(result.exit_code, 0,
            f'Error running the command with "{" ".join(datasets_args)}"\nResult: {result}')
        
        info('TESTING RESULTING PREDICTIONS...')
        self.check_predictions_folders(predictions)

    def test_singlem(self):
        '''
        Command run without errors

        Resulting predictions have same hashes as the one registered in the thesis,
        for datasets featured in it.

        Prediction images are the same number as in the csv file
        '''
        set_working_dir(self)

        runner = CliRunner()

        predictions = []
        info('TESTING SINGLEM COMMAND...')
        # NOTE: uncomment to test all datasets (longer time)
        #for m in get_datasets(): # get targets
        db = medium()
        # NOTE: reset or else hashes may not be the same if randomized
        db.reset(predefined=True)
        # save runned prediction in the list
        pred_name = db.name
        predictions.append(pred_name)
        # remove previous predictions
        rm_folder(os.path.join(predictions_dir, pred_name))
        # run command
        result = runner.invoke(single, ['-t', db.name])
        # Command has no errors on run
        self.assertEqual(result.exit_code, 0,
            f'Error running the command with "-t {db.name}"\nResult: {result}')
        info(predictions)
        
        info('TESTING RESULTING PREDICTIONS...')
        self.check_predictions_folders(predictions)


if __name__ == '__main__':
    unittest.main()
