from .base import BaseSeqUD

class SeqUDOptimizer(BaseSeqUD):
    """
    Optimization based on Sequential Uniform Design.
    
    Parameters
    ----------
    :type obj_func: a callable function 
    :param obj_func: This is the objective function to be estimated.
    
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

    :type rand_seed: int, optional, default=0
    :param rand_seed: The random seed for optimization.
    
    :type verbose: boolean, optional, default = False
    :param verbose: It controls whether the searching history will be printed. 

    Examples
    ----------
    >>> from sklearn import svm
    >>> from pySeqUD.optSeqUD import SeqUDSklearn
    >>> def cliff(parameters):
    >>>     x1 = parameters['x1']
    >>>     x2 = parameters['x2']
    >>>     term1 = -0.5*x1**2/100
    >>>     term2 = -0.5*(x2+0.03*x1**2-3)**2
    >>>     y = np.exp(term1 + term2)
    >>>     return  y
    >>> ParaSpace =  {'x1': {'Type': 'continuous', 'Range': [-20,20], 'Wrapper': lambda x: x}, 
    >>>           'x2': {'Type': 'continuous', 'Range': [-10,5], 'Wrapper': lambda x: x}}
    >>> Level_Number = 20
    >>> clf = SeqUDOptimizer(cliff, ParaSpace, Level_Number, n_jobs = 10, verbose = True)
    >>> clf.search()

    Attributes
    ----------
    :ivar best_score_: float
        The best average cv score among the evaluated trials.  

    :ivar best_params_: dict
        Parameters that reaches `best_score_`.

    :ivar search_time_consumed_: float
        Seconds used for whole searching procedure.
    """  
    
    def __init__(self, obj_func, para_space, level_number, max_runs = 100, max_search_iter = 100, n_jobs = 10, rand_seed = 0, verbose=False):

        super(SeqUDOptimizer,self).__init__(para_space,level_number,max_runs,max_search_iter,n_jobs,rand_seed,verbose)
        self.obj_func = obj_func
        
        
    def search(self):
        """ A search wrappper function. """
        def obj_func_wrap(parameters):
            score = self.obj_func(parameters)
            return score
        
        self._run(obj_func_wrap)

