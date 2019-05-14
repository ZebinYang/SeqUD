import warnings
warnings.filterwarnings("ignore")

import os
import time
import shutil
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from matplotlib import pylab as plt
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import cross_val_score

class BayoptBase():
    """ 
    Abstract Batch design class for Sklearn Hyperparameter optimization. 

    """    

    def __init__(self, estimator, cv, para_space, max_runs = 100, 
                 scoring=None, n_jobs=None, refit=False, rand_seed = 0, verbose=False):

        self.estimator = estimator        
        self.cv = cv
        self.para_space = para_space
        self.max_runs = max_runs
        self.scoring = scoring

        self.n_jobs = n_jobs
        self.refit = refit
        self.rand_seed = rand_seed
        self.verbose = verbose

        self.factor_number = len(self.para_space)
        self.para_names = list(self.para_space.keys())
        self.iteration = 0
        self.logs = pd.DataFrame()

    def plot_scores(self):
        """
        Visualize the scores history.
        """
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

    def _para_mapping(self, para_set_ud):
        
        """
        This function maps trials points to required forms. 
        """
        raise NotImplementedError  

        
    def _run(self, obj_func):
        """
        Main loop for searching the best hyperparameters. 
        
        """        
        raise NotImplementedError
        
        
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
        
        def obj_func(cfg):
            next_params = pd.DataFrame(np.array([cfg]), columns = self.para_names)
            parameters = {}
            for item, values in self.para_space.items():
                if (values['Type']=="continuous"):
                    parameters[item] = values['Wrapper'](float(next_params[item].iloc[0]))
                elif (values['Type']=="integer"):
                    parameters[item] = int(next_params[item].iloc[0]) 
                elif (values['Type']=="categorical"):
                    parameters[item] = next_params[item][0]
            self.estimator.set_params(**parameters)
            out = cross_val_score(self.estimator, x, y, cv = self.cv, scoring = self.scoring)
            score = np.mean(out)

            logs_aug = parameters
            logs_aug.update({"score":score})
            logs_aug = pd.DataFrame(logs_aug, index = [self.iteration])
            self.logs = pd.concat([self.logs, logs_aug]).reset_index(drop=True)

            if self.verbose:
                self.pbar.update(1)
                self.iteration += 1
                self.pbar.set_description("Iteration %d:" %self.iteration)
                self.pbar.set_postfix_str("Current Best Score = %.5f"% (self.logs.loc[:,"score"].max()))
            return -score
        
        if self.verbose:
            self.pbar = tqdm(total=self.max_runs) 

        self.iteration = 0
        self.logs = pd.DataFrame()
        search_start_time = time.time()
        self._para_mapping()
        self._run(obj_func)
        search_end_time = time.time()
        self.search_time_consumed_ = search_end_time - search_start_time
        
        self._summary()
        if self.verbose:
            self.pbar.close()

        if self.refit:
            self.best_estimator_ = self.estimator.set_params(**self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(x, y)
            else:
                self.best_estimator_.fit(x)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time