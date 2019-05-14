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

from pyDOE import lhs

class BatchBase():
    """ 
    Abstract Batch design class for Sklearn Hyperparameter optimization. 

    """    

    def __init__(self, para_space, max_runs = 100, n_jobs = None, verbose=False):

        self.para_space = para_space
        self.max_runs = max_runs
        self.n_jobs = n_jobs
        self.verbose = verbose
    
        self.para_ud_names = []
        self.variable_number = [0]
        self.factor_number = len(self.para_space)
        self.para_names = list(self.para_space.keys())
        for item, values in self.para_space.items():
            if (values['Type']=="categorical"):
                self.variable_number.append(len(values['Mapping']))
                self.para_ud_names.extend([item + "_UD_" + str(i+1) for i in range(len(values['Mapping']))])
            else:
                self.variable_number.append(1)
                self.para_ud_names.append(item+ "_UD")
        self.extend_factor_number = sum(self.variable_number)  

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
        This function maps trials points in UD space ([0, 1]) to original scales. 
        
        There are three types of variables: 
          - continuous：Perform inverse Maxmin scaling for each value. 
          - integer: Evenly split the UD space, and map each partition to the corresponding integer values. 
          - categorical: The UD space uses one-hot encoding, and this function selects the one with the maximal value as class label.
          
        Parameters
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point, 
                and columns are used to represent variables. 
        
        Returns
        ----------
        para_set: The transformed variables.
        """
        
        para_set = pd.DataFrame(np.zeros((para_set_ud.shape[0],self.factor_number)), columns = self.para_names) 
        for item, values in self.para_space.items():
            if (values['Type']=="continuous"):
                para_set[item] = values['Wrapper'](para_set_ud[item+"_UD"]*(values['Range'][1]-values['Range'][0])+values['Range'][0])
            elif (values['Type'] == "integer"):
                temp = np.linspace(0, 1, len(values['Mapping'])+1)
                for j in range(1,len(temp)):
                    para_set.loc[(para_set_ud[item+"_UD"]>=temp[j-1])&(para_set_ud[item+"_UD"]<temp[j]),item] = values['Mapping'][j-1]
                para_set.loc[para_set_ud[item+"_UD"]==1,item] = values['Mapping'][-1]
                para_set[item] = para_set[item].round().astype(int)
            elif (values['Type'] == "categorical"):
                column_bool = [item in para_name for para_name in self.para_ud_names]
                col_index = np.argmax(para_set_ud.loc[:,column_bool].values, axis = 1).tolist()
                para_set[item] = np.array(values['Mapping'])[col_index]
        return para_set  

    def _run(self, obj_func):
        """
        Main loop for searching the best hyperparameters. 
        
        """        
        raise NotImplementedError

    def fmin(self, wrapper_func):
        """
        Search the optimal value of a function. 
        
        Parameters
        ----------
        :type func: callable function
        :param func: the function to be optimized.
         
        """   
        np.random.seed(self.rand_seed)
        search_start_time = time.time()
        self._run(wrapper_func)
        search_end_time = time.time()
        self.search_time_consumed_ = search_end_time - search_start_time


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
        
        np.random.seed(self.rand_seed)
        search_start_time = time.time()
        self._run(sklearn_wrapper)
        search_end_time = time.time()
        self.search_time_consumed_ = search_end_time - search_start_time
       
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