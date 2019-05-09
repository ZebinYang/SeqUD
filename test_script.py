import unittest
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score

from SeqMM.pySeqUD import SeqUDSklearn
from SeqMM.pySeqUD import SeqUDOptimizer
from SeqMM.pyBayOpt.skGPEI import GPEISklearn
from SeqMM.pyBayOpt.sksmac import SMACSklearn
from SeqMM.pyBayOpt.sktpe import TPESklearn

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
            clf = SeqUDSklearn(estimator, cv, ParaSpace, Level_Number, max_runs = 10, 
                         scoring = score_metric, n_jobs = 1, refit = True, verbose = True)
            clf.fit(x, y)
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
    
    def test_SeqUDOptimizer(self):
        """ Test SeqUDOptimizer. """
        
        def cliff(parameters):
            x1 = parameters['x1']
            x2 = parameters['x2']

            term1 = -0.5*x1**2/100
            term2 = -0.5*(x2+0.03*x1**2-3)**2
            y = np.exp(term1 + term2)
            return  y
        
        Level_Number = 20
        ParaSpace = {'x1': {'Type': 'continuous', 'Range': [-20,20], 'Wrapper': lambda x: x}, 
                 'x2': {'Type': 'continuous', 'Range': [-10,5], 'Wrapper': lambda x: x}}
        try: 
            clf = SeqUDOptimizer(cliff, ParaSpace, Level_Number, 
                          max_runs = 10, n_jobs = 1, verbose = True)
            clf.search()    
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)

    def test_GPEISklearn(self):
        """ Test GPEISklearn. """
        try: 
            clf = GPEISklearn(estimator, cv, ParaSpace, max_runs = 10, 
                        time_out = 10, refit = True, verbose = True)
            clf.fit(x, y)
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
        
    def test_SMACSklearn(self):
        """ Test SMACSklearn. """
        try: 
            clf = SMACSklearn(estimator, cv, ParaSpace, 
                        max_runs = 10, refit = True, verbose = True)
            clf.fit(x,y)    
            succeed = True
        except:
            print(clf)
            succeed = False
        self.assertTrue(succeed)
        
    def test_TPESklearn(self):
        """ Test TPESklearn. """
        try: 
            clf = TPESklearn(estimator, cv, ParaSpace, 
                        max_runs = 10, refit = True, verbose = True)
            print(clf)
            clf.fit(x,y)    
            succeed = True
        except:
            succeed = False
        self.assertTrue(succeed)
        
if __name__ == '__main__':
    unittest.main()
