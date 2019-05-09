import time 
import numpy as np
import pandas as pd 
from sklearn.model_selection import cross_val_score

from .base import BaseSeqUD

class SeqUDSklearn(BaseSeqUD):
    """ 
    Hyperparameter optimization based on Sequential Uniform Design and Sklearn interface. 
    
    Parameters
    ----------
    :type  estimator: estimator object
    :param estimator: This is assumed to implement the scikit-learn estimator interface.
    
    :type  cv: cross-validation method, an sklearn object.
    :param cv: e.g., `StratifiedKFold` and KFold` is used.
    
    :type  para_space: dict or list of dictionaries
    :param para_space: It has three types:
    
        Continuous: 
            Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
            `Wrapper`, a callable function for wrapping the values.  
        Integer:
            Specify `Type` as `integer`, and include the keys of `Mapping` (a list with all the sortted integer elements).
        Categorical:
            Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories).
    
    :type level_number: int
    :param level_number: The positive integer which represent the number of levels in generating uniform design. 
    
    :type max_runs: int
    :param max_runs: The maximum number of trials to be evaluated. When this values is reached, 
        then the algorithm will stop. 
    
    :type max_search_iter: int 
    :param max_search_iter: The maximum number of iterations used to generate uniform design or augmented uniform design.
    
    :type scoring: string, callable, list/tuple, dict or None, optional, default: None
    :param scoring: A sklearn type scoring function. 
        If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.
    
    :type n_jobs: int or None, optional, optional, default=None
    :param n_jobs: Number of jobs to run in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging. See the package `joblib` for details.
    
    :type refit: boolean, or string, optional, default=True
    :param refit: It controls whether to refit an estimator using the best found parameters on the whole dataset.
    
    :type rand_seed: int, optional, default=0
    :param rand_seed: The random seed for optimization.
    
    :type verbose: boolean, optional, default = False
    :param verbose: It controls whether the searching history will be printed. 


    Examples   
    ----------
    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> from pySeqUD.skSeqUD import SeqUDSklearn
    >>> from sklearn.model_selection import KFold
    >>> iris = datasets.load_iris()
    >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
    >>> Level_Number = 20
    >>> estimator = svm.SVC()
    >>> cv = KFold(n_splits=5, random_state=1, shuffle=True)
    >>> clf = SeqUDSklearn(estimator, cv, ParaSpace, Level_Number, n_jobs = 10, refit = True, verbose = True)
    >>> clf.fit(iris.data, iris.target)

    Attributes
    ----------
    :ivar best_score_: float
        The best average cv score among the evaluated trials.  

    :ivar best_params_: dict
        Parameters that reaches `best_score_`.

    :ivar best_estimator_: 
        The estimator refitted based on the `best_params_`. 
        Not available if `refit=False`.

    :ivar search_time_consumed_: float
        Seconds used for whole searching procedure.

    :ivar refit_time_: float
        Seconds used for refitting the best model on the whole dataset.
        Not available if `refit=False`.
    """    

    def __init__(self, estimator, cv, para_space, level_number, max_runs = 100, max_search_iter = 100,
                 scoring=None, n_jobs=None, refit=False, rand_seed = 0, verbose=False):

        super(SeqUDSklearn,self).__init__(para_space,level_number,max_runs,max_search_iter,n_jobs,rand_seed,verbose)
        
        self.cv = cv
        self.refit = refit
        self.scoring = scoring
        self.estimator = estimator        

    def fit(self, x, y = None):
        """
        Run fit with all sets of parameters.
        
        Parameters
        ----------
        :type x: array, shape = [n_samples, n_features] 
        :param x: input variales.
        
        :type y: array, shape = [n_samples] or [n_samples, n_output], optional
        :param y: target variable.
        
        """
        def obj_func(parameters):
            self.estimator.set_params(**parameters)
            out = cross_val_score(self.estimator, x, y, cv = self.cv, scoring = self.scoring)
            score = np.mean(out)
            return score

        self._run(obj_func)

        if self.refit:
            self.best_estimator_ = self.estimator.set_params(**self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(x, y)
            else:
                self.best_estimator_.fit(x)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time
