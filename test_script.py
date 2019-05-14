import unittest
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score

from seqmm import SeqUD
from seqmm import GPEIOPT
from seqmm import SMACOPT
from seqmm import TPEOPT

sx = MinMaxScaler()
dt = datasets.load_breast_cancer()
x = sx.fit_transform(dt.data)
y = dt.target

ParaSpace = {'C':    {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
         'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

estimator = svm.SVC()
score_metric = make_scorer(accuracy_score, True)
cv = KFold(n_splits=5, random_state=0, shuffle=True)

class TestSeqMM(unittest.TestCase):
    """
    SeqMM test class.
    """
    def setUp(self):
        """ Initialise test suite - no-op. """
        pass

    def tearDown(self):
        """ Clean up test suite - no-op. """
        pass

    def test_SeqUDSklearn(self):
        """ Test SeqUDSklearn. """
        
        Level_Number = 20
        try: 
            clf = SeqUDSklearn(ParaSpace, Level_Number, max_runs = 10, estimator = estimator, cv = cv, 
                         scoring = score_metric, n_jobs = 1, refit = True, verbose = False)
            clf.fit(x, y)
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)

    def test_GPEISklearn(self):
        """ Test GPEISklearn. """
        try: 
            clf = GPEISklearn(ParaSpace, max_runs = 10, time_out = 10, estimator = estimator, cv = cv, refit = True, verbose = False)
            clf.fit(x, y)
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
        
    def test_SMACSklearn(self):
        """ Test SMACSklearn. """
        try: 
            clf = SMACSklearn(ParaSpace, max_runs = 10, estimator = estimator, cv = cv, refit = True, verbose = False)
            clf.fit(x,y)    
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
        
    def test_TPESklearn(self):
        """ Test TPESklearn. """
        try: 
            clf = TPESklearn(ParaSpace, max_runs = 10, estimator = estimator, cv = cv, refit = True, verbose = False)
            clf.fit(x,y)    
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
        
if __name__ == '__main__':
    unittest.main()
