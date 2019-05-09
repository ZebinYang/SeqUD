from .base import BaseSeqUD

class SeqUDOptimizer(BaseSeqUD):
    """
    Optimization based on Sequential Uniform Design.
    
        Parameters
        ----------
        obj_func : a callable function
            This is the objective function to be estimated.
        para_space : dict or list of dictionaries
            It has three types:
            Continuous: 
                Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
                `Wrapper`, a callable function for wrapping the values.  
            Integer:
                Specify `Type` as `integer`, and include the keys of `Mapping` (a list with all the sortted integer elements).
            Categorical:
                Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories).
        level_number: int
            The positive integer which represent the number of levels in generating uniform design. 
        max_runs: int
            The maximum number of trials to be evaluated. When this values is reached, 
            then the algorithm will stop. 
        max_search_iter: int 
            The maximum number of iterations used to generate uniform design or augmented uniform design.
        scoring : string, callable, list/tuple, dict or None, optional, optional, default = None
            A sklearn type scoring function. 
            If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.
        n_jobs: int or None, optional, optional, default=None
            Number of jobs to run in parallel.
            If -1 all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging. See the package `joblib` for details.
        rand_seed: int, optional, default=0
            The random seed for optimization.
        verbose : boolean, optional, default = False
            It controls whether the searching history will be printed. 


        Examples
        --------
        >>> from sklearn import svm
        >>> from pySeqUD.skSeqUD import SeqUDSklearn
        >>> def cliff(parameters):
        >>>     x1 = parameters['x1']
        >>>     x2 = parameters['x2']
        >>>     term1 = -0.5*x1**2/100
        >>>     term2 = -0.5*(x2+0.03*x1**2-3)**2
        >>>     y = np.exp(term1 + term2)
        >>>     return  y
        >>> ParaSpace =  {'x1': {'Type': 'continuous', 'Range': [-20,20], 'wrapper': lambda x: x}, 
        >>>           'x2': {'Type': 'continuous', 'Range': [-10,5], 'wrapper': lambda x: x}}
        >>> Level_Number = 20
        >>> clf = SeqUDOptimizer(cliff, ParaSpace, Level_Number, n_jobs = 10, verbose = True)
        >>> clf.search()
        
        Attributes
        ----------
        best_score_ : float
            The best average cv score among the evaluated trials.  

        best_params_ : dict
            Parameters that reaches `best_score_`.

        search_time_consumed_: float
            Seconds used for whole searching procedure.
    """  
    
    def __init__(self, obj_func, para_space, level_number, max_runs = 100, max_search_iter = 100, n_jobs = 10, rand_seed = 0, verbose=False):

        super(SeqUDOptimizer,self).__init__(para_space,level_number,max_runs,max_search_iter,n_jobs,rand_seed,verbose)
        self.obj_func = obj_func
        
    def search(self):
        """ A search wrappper function. """
        def obj_func_wrapper(parameters):
            score = self.obj_func(parameters)
            return score
        self._run_search(obj_func_wrapper)

