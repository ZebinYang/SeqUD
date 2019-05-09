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

class LHSSklearn():
    """ 
    Sklearn Hyperparameter optimization interface based on Latin Hypercube Sampling. 

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
    
    :type max_runs: int
    :param max_runs: The maximum number of trials to be evaluated. When this values is reached, 
        then the algorithm will stop. 
        
    :type scoring: string, callable, list/tuple, dict or None, optional, default: None
    :param scoring: A sklearn type scoring function. 
        If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.
    
    :type refit: boolean, or string, optional, default=True
    :param refit: It controls whether to refit an estimator using the best found parameters on the whole dataset.
    
    :type rand_seed: int, optional, default=0
    :param rand_seed: The random seed for optimization.
    
    :type verbose: boolean, optional, default = False
    :param verbose: It controls whether the searching history will be printed. 

    Examples
    ----------
    >>> import numpy as np
    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> from SeqMM.pybatdoe.skLHS import LHSSklearn
    >>> from sklearn.model_selection import KFold
    >>> iris = datasets.load_iris()
    >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
    >>> estimator = svm.SVC()
    >>> cv = KFold(n_splits=5, random_state=0, shuffle=True)
    >>> clf = LHSSklearn(estimator, cv, ParaSpace, max_runs = 100, refit = True, verbose = True)
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
                 scoring=None, n_jobs=None, refit=False, rand_seed = 0, verbose=False):

        self.estimator = estimator        
        self.cv = cv
        
        self.para_space = para_space
        self.rand_seed = rand_seed
        self.max_runs = max_runs

        self.n_jobs = n_jobs
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose

        self.iteration = 0
        self.logs = pd.DataFrame()
        
        self.para_ud_names = []
        self.variable_number = [0]
        self.factor_number = len(self.para_space)
        self.para_names = list(self.para_space.keys())
        for items, values in self.para_space.items():
            if (values['Type']=="categorical"):
                self.variable_number.append(len(values['Mapping']))
                self.para_ud_names.extend([items + "_UD_" + str(i+1) for i in range(len(values['Mapping']))])
            else:
                self.variable_number.append(1)
                self.para_ud_names.append(items+ "_UD")
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
        for items, values in self.para_space.items():
            if (values['Type']=="continuous"):
                para_set[items] = values['Wrapper'](para_set_ud[items+"_UD"]*(values['Range'][1]-values['Range'][0])+values['Range'][0])
            elif (values['Type'] == "integer"):
                temp = np.linspace(0, 1, len(values['Mapping'])+1)
                for j in range(1,len(temp)):
                    para_set.loc[(para_set_ud[items+"_UD"]>=temp[j-1])&(para_set_ud[items+"_UD"]<temp[j]),items] = values['Mapping'][j-1]
                para_set.loc[para_set_ud[items+"_UD"]==1,items] = values['Mapping'][-1]
                para_set[items] = para_set[items].round().astype(int)
            elif (values['Type'] == "categorical"):
                column_bool = [items in para_name for para_name in self.para_ud_names]
                col_index = np.argmax(para_set_ud.loc[:,column_bool].values, axis = 1).tolist()
                para_set[items] = np.array(values['Mapping'])[col_index]
        return para_set  

    def _lhs_run(self, obj_func):
        """
        Main loop for searching the best hyperparameters. 
        
        """        
        para_set_ud = lhs(n = self.extend_factor_number, samples = self.max_runs, criterion='centermaximin')
        para_set_ud = pd.DataFrame(para_set_ud, columns = self.para_ud_names)
        para_set = self._para_mapping(para_set_ud)
        para_set_ud.columns = self.para_ud_names
        candidate_params = [{para_set.columns[j]: para_set.iloc[i,j] 
                             for j in range(para_set.shape[1])} 
                            for i in range(para_set.shape[0])] 
        out = Parallel(n_jobs=self.n_jobs)(delayed(obj_func)(parameters)
                                for parameters in tqdm(candidate_params))
        self.logs = para_set_ud.to_dict()
        self.logs.update(para_set)
        self.logs.update(pd.DataFrame(out, columns = ["score"]))
        self.logs = pd.DataFrame(self.logs).reset_index(drop=True)
        if self.verbose:
            print("Search completed (%d/%d) with best score: %.5f."
                %(self.logs.shape[0], self.max_runs, self.logs["score"].max()))

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
        
        search_start_time = time.time()
        self._lhs_run(obj_func)
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