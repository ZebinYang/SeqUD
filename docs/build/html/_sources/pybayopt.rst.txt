Bayesian Optimization
==========================================


Introduction 
---------------------------------------

In classical Bayesian optimization, trials points are sequentially sampled one-point-at-a-time through
maximizing the expected improvement. Let's see a univariate example. 

.. image:: ./images/Demo_BO_eng.png
    :width: 80%
    :align: center
    
The following three Bayesian optimization methods are most popular in AutoML area.

- GP-EI ([Snoek2012]_): Gaussian process and expected improvement

- SMAC ([Hutter2011]_): It use random forest as surrogote model and EI and acquisition function.

- TPE ([Bergstra2011]_): Tree-structured Parzen Estimator. It use non parametric method to model :math:`p(x|y)` and :math:`p(y)` (the prior is not of interest actually) instead of :math:`p(y|x)`.


Their corresponding python implementations can be found here. 

- Spearmint (GP-EI) : https://github.com/JasperSnoek/spearmint

- Hyperopt (TPE): https://github.com/hyperopt/hyperopt

- SMAC: https://github.com/automl/SMAC3


Potential limitations of Bayesian optimization:

- The meta-modeling and acquisition function optimization are difficult for high-dimentional problems.

- Lack uniformality considerations: algorithm can be trapped into local areas if without a good initialization.

- Bayesian optimization are designed to select trials one-by-one, which is unnatural to perform parallelization.



Code Examples 
---------------------------------------

GP-EI Xgboost::

        import numpy as np
        import xgboost as xgb
        from sklearn import datasets
        from sklearn.model_selection import KFold 
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import make_scorer, mean_squared_error
        from seqmm import SMACOPT
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
        estimator = xgb.XGBRegressor()
        score_metric = make_scorer(mean_squared_error, False)
        cv = KFold(n_splits=5, random_state=0, shuffle=True)

        clf = SMACOPT(ParaSpace, max_runs = 100, estimator = estimator, cv = cv, refit = True, scoring = score_metric, verbose = True)
        clf.fit(x, y)
        clf.plot_scores()        
        

SMAC::
  
        import numpy as np
        from sklearn import svm
        from sklearn import datasets
        from matplotlib import pylab as plt
        from sklearn.model_selection import KFold 
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, accuracy_score
        from seqmm import GPEIOPT

        sx = MinMaxScaler()
        dt = datasets.load_breast_cancer()
        x = sx.fit_transform(dt.data)
        y = dt.target

        ParaSpace = {'C':     {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
                     'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

        estimator = svm.SVC()
        score_metric = make_scorer(accuracy_score, True)
        cv = KFold(n_splits=5, random_state=0, shuffle=True)

        clf = GPEIOPT(ParaSpace, max_runs = 100, estimator = estimator, cv = cv, refit = True, scoring = score_metric, verbose = True)
        clf.fit(x, y)
        clf.plot_scores()
        
        
        
TPE::

        import numpy as np
        from sklearn import svm
        from sklearn import datasets
        from matplotlib import pylab as plt
        from sklearn.model_selection import KFold 
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, accuracy_score
        from seqmm import GPEIOPT

        sx = MinMaxScaler()
        dt = datasets.load_breast_cancer()
        x = sx.fit_transform(dt.data)
        y = dt.target

        ParaSpace = {'C':     {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
                     'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

        estimator = svm.SVC()
        score_metric = make_scorer(accuracy_score, True)
        cv = KFold(n_splits=5, random_state=0, shuffle=True)

        clf = GPEIOPT(ParaSpace, max_runs = 100, estimator = estimator, cv = cv, refit = True, scoring = score_metric, verbose = True)
        clf.fit(x, y)
        clf.plot_scores()
        
        
Reference list 
--------------------

.. [Snoek2012] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. In Advances in Neural Information Processing Systems, pages 2951–2959, 2012.

.. [Hutter2011] Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. Sequential model-based optimization for general algorithm configuration. In International Conference on Learning and Intelligent Optimization, pages 507–523. Springer, 2011.

.. [Bergstra2011] James S Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyper-parameter optimization. In Advances in Neural Information Processing Systems, pages 2546–2554, 2011.