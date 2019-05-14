import warnings
warnings.filterwarnings("ignore")

import os
import time
import shutil
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import cross_val_score

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from .bayopt_base import BayoptBase

class SMACOPT(BayoptBase):
    """ 
    Sklearn Hyperparameter optimization interface based on SMAC (Bayesian Optimization). 
    
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
    
    :type max_runs: int, optional, default = 100
    :param max_runs: The maximum number of trials to be evaluated. When this values is reached, 
        then the algorithm will stop. 
        
    :type scoring: string, callable, list/tuple, dict or None, optional, default = None
    :param scoring: A sklearn type scoring function. 
        If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.
    
    :type refit: boolean, or string, optional, default = True
    :param refit: It controls whether to refit an estimator using the best found parameters on the whole dataset.
    
    :type rand_seed: int, optional, default = 0
    :param rand_seed: The random seed for optimization.
    
    :type verbose: boolean, optional, default = False
    :param verbose: It controls whether the searching history will be printed. 


    Examples
    ----------
    >>> import numpy as np
    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> from seqmm.pybayopt import SMACOPT
    >>> from sklearn.model_selection import KFold
    >>> iris = datasets.load_iris()
    >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
    >>> estimator = svm.SVC()
    >>> cv = KFold(n_splits=5, random_state=0, shuffle=True)
    >>> clf = SMACOPT(estimator, cv, ParaSpace, refit = True, verbose = True)
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

    def __init__(self, estimator, cv, para_space, max_runs = 100, 
                 scoring=None, refit=False, rand_seed = 0, verbose = False):

        super(SMACOPT,self).__init__(estimator, cv, para_space, max_runs, scoring, 
                       refit, rand_seed, verbose)
        

    def _para_mapping(self):
        """
        This function configures different hyperparameter types of spearmint. 
        
        """

        self.cs = ConfigurationSpace()
        for item, values in self.para_space.items():
            if values['Type'] =="continuous":
                para = UniformFloatHyperparameter(item, values['Range'][0], values['Range'][1])
            elif values['Type'] =="integer":
                para = UniformIntegerHyperparameter(item, min(values['Mapping']), max(values['Mapping']))
            elif values['Type'] =="categorical":
                para = CategoricalHyperparameter(item, values['Mapping'])
            self.cs.add_hyperparameter(para)

    def _run(self, obj_func):
        """
        Main loop for searching the best hyperparameters. 
        
        """        

        file_dir = "./temp"  + "/" + str(time.time()) + str(np.random.rand(1)[0]) + "/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": self.max_runs,  # maximum function evaluations
                         "cs": self.cs,               # configuration space
                         "deterministic": "true",
                         "output_dir": file_dir,
                         "abort_on_first_run_crash":False})

        self.smac = SMAC(scenario=scenario, rng=np.random.seed(self.rand_seed), tae_runner=obj_func)
        self.smac.solver.intensifier.tae_runner.use_pynisher = False # turn off the limit for resources
        incumbent = self.smac.optimize()
        shutil.rmtree(file_dir)