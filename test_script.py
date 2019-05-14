import unittest
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score

from seqmml import SeqUD
from seqmml import GPEIOPT
from seqmml import SMACOPT
from seqmml import TPEOPT

sx = MinMaxScaler()
dt = datasets.load_breast_cancer()
x = sx.fit_transform(dt.data)
y = dt.target

ParaSpace = {'C':    {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
         'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

estimator = svm.SVC()
score_metric = make_scorer(accuracy_score, True)
cv = KFold(n_splits=5, random_state=0, shuffle=True)

class TestSeqMML(unittest.TestCase):
    """
    SeqMM test class.
    """
    def setUp(self):
        """ Initialise test suite - no-op. """
        pass

    def tearDown(self):
        """ Clean up test suite - no-op. """
        pass

    def test_SeqUD(self):
        """ Test SeqUD. """

        try: 
            clf = SeqUD(ParaSpace, level_number = 20, max_runs = 10, estimator = estimator, cv = cv, 
                         scoring = score_metric, n_jobs = 1, refit = True, verbose = False)
            clf.fit(x, y)
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)

    def test_GPEI(self):
        """ Test GPEI. """
        try: 
            clf = GPEIOPT(ParaSpace, max_runs = 10, time_out = 10, estimator = estimator, cv = cv, refit = True, verbose = False)
            clf.fit(x, y)
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
        
    def test_SMAC(self):
        """ Test SMAC. """
        try: 
            clf = SMACOPT(ParaSpace, max_runs = 10, estimator = estimator, cv = cv, refit = True, verbose = False)
            clf.fit(x,y)    
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
        
    def test_TPE(self):
        """ Test TPE. """
        try: 
            clf = TPEOPT(ParaSpace, max_runs = 10, estimator = estimator, cv = cv, refit = True, verbose = False)
            clf.fit(x,y)    
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
        
if __name__ == '__main__':
    unittest.main()
