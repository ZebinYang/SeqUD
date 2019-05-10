import os
import time
import signal
import shutil
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib import pylab as plt
from tqdm import tqdm_notebook as tqdm

from spearmint.ExperimentGrid import GridMap
from spearmint.chooser import GPEIOptChooser as module
from sklearn.model_selection import cross_val_score

grid_seed = 1 # the number of initial points to skip
grid_size = 1000

def set_timeout(func):
    def handle(signum, frame): 
        raise RuntimeError

    def to_do(*args, **kwargs):
        try:
            signal.signal(signal.SIGALRM, handle)  
            signal.alarm(args[0].time_out)   
            r = func(*args, **kwargs)
            signal.alarm(0)   
            return r
        except RuntimeError as e:
            print("Time out!")
    return to_do
    
class GPEISklearn():
    """ 
    Sklearn Hyperparameter optimization interface based on Gaussian Process - Expected Improvement (Bayesian Optimization). 

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
    
    :type time_out: float, optional, default = 10
    :param time_out: The time out threshold (in seconds) for generating the next run. 
    
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
    >>> from seqmm.pybayopt import GPEISklearn
    >>> from sklearn.model_selection import KFold
    >>> iris = datasets.load_iris()
    >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
    >>> estimator = svm.SVC()
    >>> cv = KFold(n_splits=5, random_state=0, shuffle=True)
    >>> clf = GPEISklearn(estimator, cv, ParaSpace, max_runs = 100, refit = True, verbose = True)
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

    def __init__(self, estimator, cv, para_space, max_runs = 100, time_out = 10,
                 scoring=None, refit=False, rand_seed = 0, verbose = False):

        self.estimator = estimator        
        self.cv = cv
        
        self.para_space = para_space
        self.rand_seed = rand_seed
        self.max_runs = max_runs

        self.time_out = time_out
        self.refit = refit
        self.scoring = scoring
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

    def _para_mapping(self):
        """
        This function configures different hyperparameter types of spearmint. 
        
        """

        self.variables = {}
        for item, values in self.para_space.items():
            if (values['Type']=="continuous"):
                self.variables[item] =  OrderedDict({'name': item, 
                                 'type':'float',
                                 'min': values['Range'][0],
                                 'max': values['Range'][1],
                                 'size': 1})
            elif (values['Type']=="integer"):
                self.variables[item] = OrderedDict({'name': item, 
                                 'type':'int',
                                 'min': min(values['Mapping']),
                                 'max': max(values['Mapping']),
                                 'size': 1})
            elif (values['Type']=="categorical"):
                self.variables[item] = OrderedDict({'name': item, 
                                 'type':'enum',
                                 'options': values['Mapping'],
                                 'size': 1}) 
    
    @set_timeout  
    def _spmint_opt(self, params, values, variables, file_dir):
        """
        Interface for generating next run based on spearmint. 

        """
        chooser = module.init(file_dir, "mcmc_iters=10")
        vkeys = [k for k in variables]

        #vkeys.sort()
        gmap = GridMap([variables[k] for k in vkeys], grid_size)
        candidates = gmap.hypercube_grid(grid_size, grid_seed+params.shape[0])
        if (params.shape[0] > 0):
            grid = np.vstack((params, candidates))
        else:
            grid = candidates
        grid = np.asarray(grid)
        grid_idx = np.hstack((np.zeros(params.shape[0]),
                              np.ones(candidates.shape[0])))
        job_id = chooser.next(grid, np.squeeze(values), [],
                              np.nonzero(grid_idx == 1)[0],
                              np.nonzero(grid_idx == 2)[0],
                              np.nonzero(grid_idx == 0)[0])

        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
        else:
            candidate = grid[job_id,:]
        next_params = gmap.unit_to_list(candidate)
        return candidate, next_params

    def _spearmint_run(self, obj_func):
        """
        Main loop for searching the best hyperparameters. 
        
        """        
        scores = []
        param_unit = []
        file_dir = "./temp/" + str(time.time()) + str(np.random.rand(1)[0]) + "/"
        os.makedirs(file_dir)
        np.random.seed(self.rand_seed)
        for i in range(self.max_runs):
            try:
                candidate, next_params = self._spmint_opt(np.array(param_unit), -np.array(scores), self.variables, file_dir)
                scores = obj_func(next_params)
                param_unit.append(candidate)
            except:
                print("Early Stop!")
                break

        shutil.rmtree(file_dir)
            
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
            return self.logs.loc[:,"score"].values.tolist()
        
        if self.verbose:
            self.pbar = tqdm(total=self.max_runs) 

        self.iteration = 0
        self.logs = pd.DataFrame()
        search_start_time = time.time()
        self._para_mapping()
        self._spearmint_run(obj_func)
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