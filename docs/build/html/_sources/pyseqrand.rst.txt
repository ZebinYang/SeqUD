SeqRandSklearn
===============
Here we give several examples to show the usage of this package.

Example 1: SVM for Classification
------------------------------------------------
Hyperparameter optimization based on Sequential Uniform Design and Sklearn SVM interface::

        import numpy as np
        from sklearn import svm
        from sklearn import datasets
        from matplotlib import pylab as plt
        from sklearn.model_selection import KFold 
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, accuracy_score
        from seqmm import SeqRandSklearn

        sx = MinMaxScaler()
        dt = datasets.load_breast_cancer()
        x = sx.fit_transform(dt.data)
        y = dt.target

        ParaSpace = {'C':     {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
                     'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

        Level_Number = 20
        estimator = svm.SVC()
        score_metric = make_scorer(accuracy_score, True)
        cv = KFold(n_splits=5, random_state=0, shuffle=True)

        clf = SeqRandSklearn(estimator, cv, ParaSpace, scoring = score_metric, n_jobs = 2, refit = True, verbose = True)
        clf.fit(x, y)
        clf.plot_scores()
