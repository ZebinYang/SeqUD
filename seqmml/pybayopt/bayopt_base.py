import time
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from tqdm import tqdm_notebook as tqdm

from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score

class BayoptBase(ABC):
    """ 
    Abstract Batch design class for Sklearn Hyperparameter optimization. 

    """    

    def __init__(self, para_space, max_runs = 100, verbose=False):

        self.para_space = para_space
        self.max_runs = max_runs
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
            print("%s completed in %.2f seconds."%(self.method, self.search_time_consumed_))
            print("The best score is: %.5f."%self.best_score_)
            print("The best configurations are:")
            print("\n".join("%-20s: %s"%(k, v if self.para_space[k]['Type']=="categorical" else round(v, 5))
                            for k, v in self.best_params_.items()))
        
    @abstractmethod
    def _run(self):
        """
        Main loop for searching the best hyperparameters. 
        
        """        
        pass
                    
            
    def fmin(self, wrapper_func):
        """
        Search the optimal value of a function. 
        
        Parameters
        ----------
        :type func: callable function
        :param func: the function to be optimized.
         
        """  
        np.random.seed(self.rand_seed)
        self.iteration = 0
        self.logs = pd.DataFrame()
        if self.verbose:
            self.pbar = tqdm(total=self.max_runs) 

        search_start_time = time.time()
        self._run(wrapper_func)
        search_end_time = time.time()
        self.search_time_consumed_ = search_end_time - search_start_time
        
        self._summary()
        if self.verbose:
            self.pbar.close()

        
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

        def sklearn_wrapper(parameters):
            self.estimator.set_params(**parameters)
            out = cross_val_score(self.estimator, x, y, cv = self.cv, scoring = self.scoring)
            score = np.mean(out)
            return score
        
        if self.verbose:
            self.pbar = tqdm(total=self.max_runs) 

        np.random.seed(self.rand_seed)
        self.iteration = 0
        self.logs = pd.DataFrame()
        search_start_time = time.time()
        self._run(sklearn_wrapper)
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