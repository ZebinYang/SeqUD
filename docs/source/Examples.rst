More Examples
===============
Here we give several examples to show the usage of this package.


SeqUD for function optimization
_________________________________

- Example 1: Cliff-Shaped function

.. code-block::

        import numpy as np 
        from matplotlib import pylab as plt
        from seqmm import SeqUD

        def cliff(parameters):
            x1 = parameters['x1']
            x2 = parameters['x2']
            term1 = -0.5*x1**2/100
            term2 = -0.5*(x2+0.03*x1**2-3)**2
            y = np.exp(term1 + term2)
            return  y

        Level_Number = 20
        ParaSpace = {'x1': {'Type': 'continuous', 'Range': [-20,20], 'Wrapper': lambda x: x}, 
                     'x2': {'Type': 'continuous', 'Range': [-10,5], 'Wrapper': lambda x: x}}
        clf = SeqUD(ParaSpace, max_runs = 100, verbose = True)
        clf.fmin(cliff)


Working with Scikit-learn Pipeline
___________________________________

.. code-block::

        import numpy as np
        from sklearn import svm
        from matplotlib import pylab as plt 
        from sklearn.model_selection import KFold 
        from sklearn.datasets import samples_generator
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        from sklearn.pipeline import Pipeline

        from seqmm import SeqUD

        X, y = samples_generator.make_classification(
            n_informative=5, n_redundant=0, random_state=42)

        anova_filter = SelectKBest(f_regression, k=5)
        clf = svm.SVC(kernel='linear')
        anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])

        anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
        ParaSpace = {'anova__k':      {'Type': 'integer',        'Mapping':  np.linspace(2,10,9)},
                     'svc__C':        {'Type': 'continuous',     'Range': [-6, 16], 'Wrapper': np.exp2}
                    }

        cv = KFold(n_splits=5, random_state=0, shuffle=True)
        clf = SeqUD(ParaSpace, estimator = anova_svm, cv = cv, verbose = True)
        clf.fit(X, y)

Methods Comparison 
_________________________

.. code-block::

        import numpy as np
        import xgboost as xgb
        from sklearn import datasets
        from matplotlib import pylab as plt 
        from sklearn.model_selection import KFold 
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import make_scorer, mean_squared_error
        from seqmm import SeqUD, SeqRand, GPEIOPT, SMACOPT, TPEOPT,\
                    GridSearch, RandSearch, LHSSearch, SobolSearch, UDSearch

        dt = datasets.load_diabetes()
        sx = MinMaxScaler()
        sy = MinMaxScaler()
        x = sx.fit_transform(dt.data)
        y = sy.fit_transform(dt.target.reshape([-1,1]))

        ParaSpace = {'booster':          {'Type': 'categorical', 'Mapping': ['gbtree', 'gblinear']},
                     'max_depth':        {'Type': 'integer',     'Mapping': np.linspace(2,8,7)}, 
                     'n_estimators':     {'Type': 'integer',     'Mapping': np.linspace(100,300,201)},
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

        seq_model_zoo = {"SeqUD": SeqUD, 
                 "SeqRand": SeqRand,
                  "GPEI": GPEIOPT, 
                  "SMAC": SMACOPT, 
                  "TPE": TPEOPT}
        for item, model in seq_model_zoo.items():
            print("Fitting model: %s"% item)
            clf = model(ParaSpace, max_runs = 100, estimator = estimator, cv = cv, scoring =  score_metric, refit = True, verbose = False)
            clf.fit(x, y)
            print("The best score is %0.5f: ,time cost: %0.2f:"% (clf.best_score_,clf.search_time_consumed_))
            plt.plot(clf.logs["score"].cummax()) 
        plt.legend(seq_model_zoo.keys())