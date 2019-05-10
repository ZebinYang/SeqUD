GPEISklearn
===============
Here we give several examples to show the usage of this package.


Example 1: SVM for Classification
------------------------------------------------
Hyperparameter optimization based on GP-EI and Sklearn SVM interface::
        
        import numpy as np
        from sklearn import svm
        from sklearn import datasets
        from matplotlib import pylab as plt
        from sklearn.model_selection import KFold 
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, accuracy_score
        from seqmm.pybayopt import GPEISklearn

        sx = MinMaxScaler()
        dt = datasets.load_breast_cancer()
        x = sx.fit_transform(dt.data)
        y = dt.target

        ParaSpace = {'C':     {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
                     'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

        estimator = svm.SVC()
        score_metric = make_scorer(accuracy_score, True)
        cv = KFold(n_splits=5, random_state=0, shuffle=True)

        clf = GPEISklearn(estimator, cv, ParaSpace, max_runs = 100, time_out = 10, refit = True, verbose = True)
        clf.fit(x, y)
        clf.plot_scores()
        
        
Example 2: Xgboost for Regression
------------------------------------------------
Hyperparameter optimization based on GP-EI and Sklearn Xgboost interface::
        
        import numpy as np
        import xgboost as xgb
        from sklearn import datasets
        from sklearn.model_selection import KFold 
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import make_scorer, mean_squared_error
        from seqmm.pybayopt import GPEISklearn

        dt = datasets.load_diabetes()
        sx = MinMaxScaler()
        sy = MinMaxScaler()
        x = sx.fit_transform(dt.data)
        y = sy.fit_transform(dt.target.reshape([-1,1]))

        ParaSpace = {'booster':          {'Type': 'categorical', 'Mapping': ['gbtree', 'gblinear']},
                     'max_depth':        {'Type': 'integer',     'Mapping': np.linspace(2,10,9)}, 
                     'n_estimators':     {'Type': 'integer',     'Mapping': np.linspace(100,500,401)},
                     'min_child_weight': {'Type': 'integer',     'Mapping': np.linspace(1,100,100)},
                     'subsample':        {'Type': 'continuous',  'Range': [0, 1],  'Wrapper': lambda x:x},
                     'colsample_bytree': {'Type': 'continuous',  'Range': [0, 1],  'Wrapper': lambda x:x},
                     'learning_rate':    {'Type': 'continuous',  'Range': [-5, 0], 'Wrapper': lambda x: 10**x},
                     'gamma':            {'Type': 'continuous',  'Range': [-5, 0], 'Wrapper': lambda x: 10**x},
                     'reg_lambda':       {'Type': 'continuous',  'Range': [-5, 0], 'Wrapper': lambda x: 10**x},
                     'reg_alpha':         {'Type': 'continuous',  'Range': [-5, 0], 'Wrapper': lambda x: 10**x}}

        dt = datasets.load_diabetes()
        sx = MinMaxScaler()
        x = sx.fit_transform(dt.data)
        y = dt.target

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

        estimator = xgb.XGBRegressor()
        score_metric = make_scorer(mean_squared_error, False)
        cv = KFold(n_splits=5, random_state=0, shuffle=True)

        clf = GPEISklearn(estimator, cv, ParaSpace, max_runs = 100, scoring = score_metric, time_out = 30, refit = True, verbose = True)
        clf.fit(x, y)
        clf.plot_scores()


Example 3: Kmeans for Unsupervised Clustering
------------------------------------------------
Hyperparameter optimization based on GP-EI and Sklearn Kmeans interface::

        import numpy as np
        from sklearn import datasets
        from sklearn.cluster import KMeans
        from sklearn.model_selection import KFold 
        from sklearn.preprocessing import MinMaxScaler
        from seqmm.pybayopt import GPEISklearn

        sx = MinMaxScaler()
        dt = datasets.load_iris()
        x = sx.fit_transform(dt.data)
        y = dt.target.reshape([-1,1])

        ParaSpace = {'n_clusters':  {'Type': 'integer',    'Mapping': np.linspace(2,9,8)}, 
                     'tol':         {'Type': 'continuous', 'Range': [-6, -3], 'Wrapper': lambda x: 10**x}}

        estimator = KMeans()
        cv = KFold(n_splits=5, random_state=0, shuffle=True)

        clf = GPEISklearn(estimator, cv, ParaSpace, max_runs = 100, refit = True, verbose = True)
        clf.fit(x, y)
        clf.plot_scores()