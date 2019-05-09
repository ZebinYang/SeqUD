Usage
================================

SeqUDOptimizer
------------------------------------------------
	.. py:class:: SeqUDOptimizer(obj_func,para_space,level_number,max_runs=100,max_search_iter=100,
								 n_jobs=10,rand_seed=0,verbose=False)
	
	
		Optimization based on Sequential Uniform Design.

   
		:type obj_func: a callable function.
		:param obj_func: this is the objective function to be estimated.
		
		:type para_space: dict or list of dictionaries
		:param para_space: it has three types:
		
				* Continuous: 
					Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
					`Wrapper`, a callable function for wrapping the values.  
				* Integer:
					Specify `Type` as `integer`, and include the keys of `Mapping` (a list with all the sortted integer elements).
				* Ctegorical:
					Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories).
		
		:type level_number: int
		:param level_number: the positive integer which represent the number of levels in generating uniform design. 
		
		:type max_runs: int
		:param max_runs: the maximum number of trials to be evaluated. When this values is reached, then the algorithm will stop. 
		
		:type max_search_iter: int 
		:param max_search_iter: the maximum number of iterations used to generate uniform design or augmented uniform design.
		
		:type scoring: string, callable, list/tuple, dict or None, optional, default: None
		:param scoring:	a sklearn type scoring function. If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.
		
		:type n_jobs: int or None, optional, default=None
		:param n_jobs: Number of jobs to run in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. See the package `joblib` for details.
		
		:type rand_seed: int, optional, default = 0
		:param rand_seed: The random seed for optimization.

		:type verbose: boolean, optional, default = False
		:param verbose: it controls whether the searching history will be printed. 

		:ivar best_score_: float, the best average cv score among the evaluated trials. 

		:ivar best_params_: float, parameters that reaches `best_score_`.
		
		:ivar search_time_consumed_: float, seconds used for whole searching procedure.

SeqUDSklearn
------------------------------------------------
	.. py:class:: SeqUDSklearn(estimator,cv,para_space,level_number,max_runs=100,max_search_iter=100,
							   scoring=None,n_jobs=None,refit=False,rand_seed=0,verbose=False)
		
		
		Hyperparameter optimization based on Sequential Uniform Design and Sklearn interface. 
   
   
	   :type estimator: estimator object
	   :param estimator: this is assumed to implement the scikit-learn estimator interface.
	   
	   :type cv: an sklearn object
	   :param cv: cross-validation method, `StratifiedKFold` and KFold` is used.
	   
	   :type para_space: dict or list of dictionaries
	   :param para_space: it has three types.
	   
			   * Continuous: 
				   Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
				   `Wrapper`, a callable function for wrapping the values.  
			   * Integer:
				   Specify `Type` as `integer`, and include the keys of `Mapping` (a list with all the sortted integer elements).
			   * Categorical:
				   Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories).
			
	   :type level_number: int
	   :param level_number: the positive integer which represent the number of levels in generating uniform design. 
	   
	   :type max_runs: int
	   :param max_runs: the maximum number of trials to be evaluated. When this values is reached, then the algorithm will stop. 
	   
	   :type max_search_iter: int 
	   :param max_search_iter: the maximum number of iterations used to generate uniform design or augmented uniform design.
	   
	   :type scoring: string, callable, list/tuple, dict or None, optional, default = None
	   :param scoring: a sklearn type scoring function. If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.
	   
	   :type n_jobs: int or None, optional, default=None
	   :param n_jobs: number of jobs to run in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. See the package `joblib` for details.
	   
	   :type  refit: boolean, or string, optional, default=True
	   :param refit: it controls whether to refit an estimator using the best found parameters on the whole dataset.
	
	   :type rand_seed: int, optional, default = 0
	   :param rand_seed: The random seed for optimization.
	
	   :type verbose: boolean, optional, default = False
	   :param verbose: it controls whether the searching history will be printed. 

	   :ivar best_score_: float, the best average cv score among the evaluated trials.

	   :ivar best_params_: dict, parameters that reaches `best_score_`.
	   
	   :ivar best_estimator_: Sklearn estimator object, the estimator refitted based on the `best_params_`. Not available if `refit=False`.
	   
	   :ivar search_time_consumed_: float, seconds used for whole searching procedure.
	   
	   :ivar refit_time_: float, seconds used for refitting the best model on the whole dataset. Not available if `refit=False`.
