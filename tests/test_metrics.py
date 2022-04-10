import unittest

import numpy as np
from utils.metrics import *
from utils.logmanager import *


class TestMetrics(unittest.TestCase):
    '''Unit testing for metrics measurements'''


    def metrics_unittesting(self):
        # singles
        y_true = np.array([[1,0,0],[0,1,0],[0,0,1]]) > 0 # > 0 to cast as bool
        y_pred = np.array([[1,0,0],[1,0,0],[0,0,0]]) > 0
        tp = 1
        tn = 5
        fp = 1
        fn = 2
        cs = confmat_scores(y_true, y_pred)
        pr = tp / (tp+fp) # 0.5
        re = tp / (tp+fn) # 0.33
        sp = tn / (tn+fp) # 0.83
        f1_ = 2*re*pr / (re+pr)
        overlap = y_true * y_pred
        union =   y_true | y_pred # with bitwise or it would work even without casting matrix as bool
        iou_ = overlap.sum() / (union.sum())
        a = (1 - pr)**2 # 0.25
        b = (1 - re)**2 # 0.4489
        c = (1 - sp)**2 # 0.0289
        dprs_ = math.sqrt(a + b + c)
        #           3               /  V        2       *    3      *     6     *    7
        mcc_ = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        info(cs)
        self.assertEqual(round(pr, 2), 0.5)
        self.assertEqual(round(pr, 2), round(precision(cs), 2), 'pr not equal to its function')
        self.assertEqual(round(re, 2), 0.33)
        self.assertEqual(round(re, 2), round(recall(cs), 2), 're not equal to its function')
        self.assertEqual(round(sp, 2), 0.83)
        self.assertEqual(round(sp, 2), round(specificity(cs), 2), 'sp not equal to its function')
        self.assertEqual(round(f1_, 2), 0.40)
        self.assertEqual(round(f1_, 2), round(f1(cs), 2), 'f1 not equal to its function')
        self.assertEqual(round(iou_, 2), 0.25)
        self.assertEqual(round(iou_, 2), round(iou(cs), 2), 'iou not equal to its function')
        self.assertEqual(round(iou_, 2), round(iou_logical(y_true, y_pred), 2), 'iou not equal to its function (alt)')
        self.assertEqual(round(dprs_, 2), 0.85)
        self.assertEqual(round(dprs_, 2), round(dprs(cs), 2), 'dprs not equal to its function')
        self.assertEqual(round(mcc_, 2), 0.19)
        self.assertEqual(round(mcc_, 2), round(mcc(cs), 2), 'mcc not equal to its function')
        # medium avg
        #metrics = [f1_medium, f1, f2, iou, iou_logical, dprs_medium, dprs, mcc, recall, precision, specificity]
        #rpd = pd_metrics(docs_y_path, docs_p_path, metrics)
        #res = print_pd_mean(rpd, metrics, desc='unit testing')
        pr_1 = 0.51
        pr_2 = 0.82
        pr_avg = (pr_1 + pr_2) /2   # 0.67
        re_1 = 0.61
        re_2 = 0.45
        re_avg = (re_1 + re_2) /2   # 0.53
        sp_1 = 0.14
        sp_2 = 0.62
        sp_avg = (sp_1 + sp_2) /2   # 0.38
        f1_med_avg = pr_avg * re_avg * 2 / (pr_avg + re_avg)
        a_ = (1 - pr_avg)**2 # 0.10
        b_ = (1 - re_avg)**2 # 0.22
        c_ = (1 - sp_avg)**2 # 0.38
        dprs_med_avg = math.sqrt(a_ + b_ + c_)
        self.assertEqual(round(f1_med_avg, 2), 0.59)
        self.assertEqual(round(f1_med_avg, 2), round(f1_medium(pr_avg, re_avg, sp_avg), 2), 'f1-medium not equal to its function')
        self.assertEqual(round(dprs_med_avg, 2), 0.85)
        self.assertEqual(round(dprs_med_avg, 2), round(dprs_medium(pr_avg, re_avg, sp_avg), 2), 'dprs-medium not equal to its function')

    def mcc_unittest(self):
        '''
        Unit testing based on MCC's paper analysis
        
        The analysis shows how F1 doesn't care much about TN and could signal
        over-optimistic data to the classifier
        '''
        # Use Case A1: Positively imbalanced dataset
        data = {}
        data['ap'] = 91   # 91 sick patients
        data['an'] = 9    # 9 healthy individuals
        data['se'] = 99
        data['tp'] = 90   # algorithm is good at predicting positive data
        data['fp'] = 9
        data['tn'] = 0
        data['fn'] = 1    # algorithm is bad at predicting negative data
        # F1 measures an almost perfect score, MCC instead measures a bad score
        # F1 0.95    MCC -0.03
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.95)
        self.assertEqual(mcc_, -0.03)

        # Use Case A2: Positively imbalanced dataset
        data = {}
        data['ap'] = 75   # 75 positives
        data['an'] = 25   # 25 negatives
        data['se'] = 11
        data['tp'] = 5    # classifier unable to predict positives
        data['fp'] = 6
        data['tn'] = 19   # classifier was able to predict negatives
        data['fn'] = 70
        # In this case both the metrics measure a bad score
        # F1 0.12    MCC -0.24
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.12)
        self.assertEqual(mcc_, -0.24)

        # Use Case B1: Balanced dataset
        data = {}
        data['ap'] = 50   # 50 positives
        data['an'] = 50   # 50 negatives
        data['se'] = 92
        data['tp'] = 47   # classifier able to predict positives
        data['fp'] = 45
        data['tn'] = 5    # classifier was unable to predict negatives
        data['fn'] = 3
        # F1 measures a good score, MCC doesn't
        # F1 0.66    MCC 0.07
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.66)
        self.assertEqual(mcc_, 0.07)

        # Use Case B2: Balanced dataset
        data = {}
        data['ap'] = 50   # 50 positives
        data['an'] = 50   # 50 negatives
        data['se'] = 14
        data['tp'] = 10   # classifier was unable to predict positives
        data['fp'] = 4
        data['tn'] = 46    # classifier able to predict negatives
        data['fn'] = 40
        # F1 measures a good score, MCC doesn't
        # F1 0.31    MCC 0.17
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.31)
        self.assertEqual(mcc_, 0.17)

        # Use Case C1: Negatively imbalanced dataset
        data = {}
        data['ap'] = 10   # 10 positives
        data['an'] = 90   # 90 negatives
        data['se'] = 98
        data['tp'] = 9    # classifier was unable to predict positives
        data['fp'] = 89
        data['tn'] = 1    # classifier able to predict negatives
        data['fn'] = 1
        # Both the scores gives bad measure
        # F1 0.17    MCC -0.19
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.17)
        self.assertEqual(mcc_, -0.19)

        # Use Case C2: Negatively imbalanced dataset
        data = {}
        data['ap'] = 11   # 10 positives
        data['an'] = 89   # 89 negatives
        data['se'] = 3
        data['tp'] = 2   # classifier was unable to predict positives
        data['fp'] = 1
        data['tn'] = 88    # classifier able to predict negatives
        data['fn'] = 9
        # Both the scores gives bad measure
        # F1 0.29    MCC 0.31
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.29)
        self.assertEqual(mcc_, 0.31)

if __name__ == '__main__':
    unittest.main()
