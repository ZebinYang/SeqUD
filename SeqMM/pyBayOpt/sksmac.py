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

class SMACSklearn():
    """ 
     Sklearn Hyperparameter optimization interface based on SMAC (Bayesian Optimization). 
    
        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface.
        cv : cross-validation method, an sklearn object.
            e.g., `StratifiedKFold` and KFold` is used.
        para_space : dict or list of dictionaries
            It has three types:
            Continuous: 
                Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
                `Wrapper`, a callable function for wrapping the values.  
            Integer:
                Specify `Type` as `integer`, and include the keys of `Mapping` (a list with all the sortted integer elements).
            Categorical:
                Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories).
        max_runs: int, optional, default = 100
            The maximum number of trials to be evaluated. When this values is reached, 
            then the algorithm will stop. 
        scoring : string, callable, list/tuple, dict or None, optional, default = None
            A sklearn type scoring function. 
            If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.
        refit : boolean, or string, optional, default=True
            It controls whether to refit an estimator using the best found parameters on the whole dataset.
        verbose : boolean, optional, default = False
            It controls whether the searching history will be printed. 


        Examples
        --------
        >>> from sklearn import svm
        >>> from sksmac import SMACSklearn
        >>> from sklearn.model_selection import KFold
        >>> iris = datasets.load_iris()
        >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'wrapper': np.exp2}, 
                   'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'wrapper': np.exp2}}
        >>> estimator = svm.SVC()
        >>> cv = KFold(n_splits=5, random_state=0, shuffle=True)
        >>> clf = SMACSklearn(estimator, cv, ParaSpace, refit = True, verbose = True)
        >>> clf.fit(iris.data, iris.target)

        Attributes
        ----------
        best_score_ : float
            The best average cv score among the evaluated trials.  

        best_params_ : dict
            Parameters that reaches `best_score_`.

        best_estimator_: 
            The estimator refitted based on the `best_params_`. 
            Not available if `refit=False`.

        search_time_consumed_: float
            Seconds used for whole searching procedure.

        refit_time_: float
            Seconds used for refitting the best model on the whole dataset.
            Not available if `refit=False`.
    """    

    def __init__(self, estimator, cv, para_space, max_runs = 100, 
                 scoring=None, refit=False, rand_seed = 0, verbose=False):

        self.estimator = estimator        
        self.cv = cv
        
        self.para_space = para_space
        self.rand_seed = rand_seed
        self.max_runs = max_runs

        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.factor_number = len(self.para_space)
        self.para_names = list(self.para_space.keys())
        self.iteration = 0
        self.logs = pd.DataFrame()
        
    def plot_scores(self):
        if self.logs.shape[0]>0:
            cum_best_score = self.logs["score"].cummax()
            fig = plt.figure(figsize = (6,4))
            plt.plot(cum_best_score)
            plt.xlabel('# of Runs')
            plt.ylabel('Best Scores')
            plt.title('The best found scores during optimization')
            plt.grid(True)
            plt.show()
        else:
            print("No available logs!")
            
    def _summary(self):
        """
        This function summarizes the evaluation results and makes records. 
        
        """

        self.best_index_ = self.logs.loc[:,"score"].idxmax()
        self.best_params_ = {self.logs.loc[:,self.para_names].columns[j]:\
                             self.logs.loc[:,self.para_names].iloc[self.best_index_,j] 
                              for j in range(self.logs.loc[:,self.para_names].shape[1])}
        
        self.best_score_ = self.logs.loc[:,"score"].iloc[self.best_index_]

        if self.verbose:
            print("Search completed in %.2f seconds."%self.search_time_consumed_)
            print("The best score is: %.5f."%self.best_score_)
            print("The best configurations are:")
            print("\n".join("%-20s: %s"%(k, v if self.para_space[k]['Type']=="categorical" else round(v, 5))
                            for k, v in self.best_params_.items()))

    def _para_mapping(self):
        """
        This function configures different hyperparameter types of spearmint. 
        
        """

        self.cs = ConfigurationSpace()
        for items, values in self.para_space.items():
            if values['Type'] =="continuous":
                para = UniformFloatHyperparameter(items, values['Range'][0], values['Range'][1])
            elif values['Type'] =="integer":
                para = UniformIntegerHyperparameter(items, min(values['Mapping']), max(values['Mapping']))
            elif values['Type'] =="categorical":
                para = CategoricalHyperparameter(items, values['Mapping'])
            self.cs.add_hyperparameter(para)

    def _smac_run(self, obj_func):
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

    def fit(self, x, y = None):
        """
        Run fit with all sets of parameters.
        Parameters
        ----------
        x : array, shape = [n_samples, n_features], input variales
        y : array, shape = [n_samples] or [n_samples, n_output], optional target variable
        """
        def obj_func(cfg):
            
            cfg = {k : cfg[k] for k in cfg}
            next_params = pd.DataFrame(cfg, columns = self.para_names, index = [0])
            parameters = {}
            for items, values in self.para_space.items():
                if (values['Type']=="continuous"):
                    parameters[items] = float(values['Wrapper'](float(next_params[items].iloc[0])))
                elif (values['Type']=="integer"):
                    parameters[items] = int(next_params[items].iloc[0]) 
                elif (values['Type']=="categorical"):
                    parameters[items] = next_params[items].iloc[0]

            self.estimator.set_params(**parameters)
            out = cross_val_score(self.estimator, x, y, cv=self.cv, scoring = self.scoring)
            score = np.mean(out)

            logs_aug = parameters
            logs_aug.update({"score":score})
            logs_aug = pd.DataFrame(logs_aug, index = [self.iteration])
            self.logs = pd.concat([self.logs, logs_aug]).reset_index(drop=True)

            self.pbar.update(1)
            self.iteration += 1
            self.pbar.set_description("Iteration %d:" %self.iteration)
            self.pbar.set_postfix_str("Current Best Score = %.5f"% (self.logs.loc[:,"score"].max()))
            return -score 

        self.iteration = 0
        self.logs = pd.DataFrame()
        self.pbar = tqdm(total=self.max_runs) 
        
        search_start_time = time.time()
        self._para_mapping()
        self._smac_run(obj_func)
        search_end_time = time.time()
        self.search_time_consumed_ = search_end_time - search_start_time
        
        self.pbar.close()
        self._summary()
        
        if self.refit:
            self.best_estimator_ = self.estimator.set_params(**self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(x, y)
            else:
                self.best_estimator_.fit(x)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time