SeqUDSklearn
===============
Here we give several examples to show the usage of this package.


Example 1: SVM for Classification
------------------------------------------------
Hyperparameter optimization based on Sequential Uniform Design and Sklearn SVM interface::

      import numpy as np
		from sklearn import svm
		from SeqMM.pySeqUD import SeqUDSklearn
		from sklearn.model_selection import KFold
		from sklearn.metrics import make_scorer, accuracy_score
		iris = datasets.load_iris()
		ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'wrapper': np.exp2}, 
			   'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'wrapper': np.exp2}}
		Level_Number = 20
		estimator = svm.SVC()
		score_metric = make_scorer(accuracy_score, True)
		cv = KFold(n_splits=5, random_state=1, shuffle=True)
		clf = SeqUDSklearn(estimator, cv, ParaSpace, Level_Number, n_jobs = 10, refit = True, verbose = True)
		clf.fit(iris.data, iris.target)
		print(clf.best_params_)
		print(clf.best_score_)
		print(clf.logs)
		
Example 2: Xgboost for Regression
------------------------------------------------
Hyperparameter optimization based on Sequential Uniform Design and Sklearn Xgboost interface::

      import numpy as np
		import xgboost as xgb
		from SeqMM.pySeqUD import SeqUDSklearn
		from sklearn.model_selection import KFold
		from sklearn.metrics import make_scorer, mean_squared_error
		boston = datasets.load_boston()
		ParaSpace = {'booster':          {'Type': 'categorical', 'Mapping': ['gbtree', 'gblinear']},
			   'max_depth':        {'Type': 'integer',     'Mapping': np.linspace(2,10,9)}, 
			   'n_estimators':     {'Type': 'integer',     'Mapping': np.linspace(100,500,401)},
			   'min_child_weight': {'Type': 'integer',     'Mapping': np.linspace(1,100,100)},
			   'subsample':        {'Type': 'continuous',  'Range': [0, 1],  'Wrapper': lambda x:x},
			   'colsample_bytree': {'Type': 'continuous',  'Range': [0, 1],  'Wrapper': lambda x:x},
			   'learning_rate':    {'Type': 'continuous',  'Range': [-5, 1], 'Wrapper': np.exp2},
			   'gamma':            {'Type': 'continuous',  'Range': [-5, 1], 'Wrapper': np.exp2},
			   'reg_lambda':       {'Type': 'continuous',  'Range': [-5, 1], 'Wrapper': np.exp2},
			   'reg_alpha':         {'Type': 'continuous',  'Range': [-5, 1], 'Wrapper': np.exp2}}

		Level_Number = 20
		estimator = xgb.XGBRegressor()
		score_metric = make_scorer(mean_squared_error, False)
		cv = KFold(n_splits=5, random_state=0, shuffle=True)
		clf = SeqUDSklearn(estimator, cv, ParaSpace, Level_Number, scoring = score_metric, n_jobs = 10, refit = True, verbose = True)
		clf.fit(boston['data'], boston['target'])
		print(clf.best_params_)
		print(clf.best_score_)
		print(clf.logs)
		
Example 3: Kmeans for Unsupervised Clustering
------------------------------------------------
Hyperparameter optimization based on Sequential Uniform Design and Sklearn Kmeans interface::

      import numpy as np
		from scipy.special import exp10
		from SeqMM.pySeqUD import SeqUDSklearn
		from sklearn.model_selection import KFold
		from sklearn.metrics import make_scorer, mean_squared_error
		from sklearn.cluster import silhouette_score
		iris = datasets.load_iris()
		ParaSpace = {'n_clusters':  {'Type': 'integer',    'Mapping': np.linspace(2,9,8)}, 
			   'tol':         {'Type': 'continuous', 'Range': [-6, -3], 'Wrapper': exp10}}
		Level_Number = 20
		estimator = KMeans()
		score_metric = make_scorer(silhouette_score, True)
		cv = KFold(n_splits=5, random_state=0, shuffle=True)
		clf = SeqUDSklearn(estimator, cv, ParaSpace, Level_Number, scoring = score_metric, n_jobs = 1, refit = True, verbose = True)
		clf.fit(iris.data, iris.target.reshape([-1,1]))
		print(clf.best_params_)
		print(clf.best_score_)
		print(clf.logs)