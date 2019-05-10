Batch Designs (One-shot)
==============================
In contrast to sequential methods, batch designs can generate all the trials before conducting any experiments.
Here, we introduce five simple strategies.

Grid Search 
------------------------------------------------
Hyperparameter optimization based on GP-EI and Sklearn SVM interface::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm.pybatdoe import GridSklearn

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = GridSklearn(estimator, cv, ParaSpace, max_runs = 100, n_jobs = 10, 
                        refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()

Random Search
------------------------------------------------
Hyperparameter optimization based on GP-EI and Sklearn Xgboost interface::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm.pybatdoe import RandSklearn

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = RandSklearn(estimator, cv, ParaSpace, max_runs = 100, n_jobs = 10, 
                        refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()

Latin Hypercube Sampling
------------------------------------------------
Hyperparameter optimization based on GP-EI and Sklearn Kmeans interface::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm.pybatdoe import LHSSklearn

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = LHSSklearn(estimator, cv, ParaSpace, max_runs = 100, n_jobs = 10, refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()
        
        
Sobol Sequence
------------------------------------------------
Hyperparameter optimization based on GP-EI and Sklearn Kmeans interface::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm.pybatdoe import LHSSklearn

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = LHSSklearn(estimator, cv, ParaSpace, max_runs = 100, n_jobs = 10, refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()        

Uniform Design
------------------------------------------------
Hyperparameter optimization based on GP-EI and Sklearn Kmeans interface::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm.pybatdoe import UDSklearn

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = UDSklearn(estimator, cv, ParaSpace, level_number = 20, max_runs = 100, max_search_iter = 30, n_jobs = 10, 
                        refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()