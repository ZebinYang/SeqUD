One-shot Batch Designs
==============================

Batch designs can generate all the experimental trials before conducting any experiments.
Here, we introduce five simple strategies.

- Grid Search: exhaustive search over grid combinations

- Random Search:  more flexible than grid experiments, when not all hyperparameters are equally important. Furthermore, new trials can be added without adjustment and the experiments can also be stopped any time.  ([Bergstra2012]_)

- Latin Hypercube Sampling: 

- Sobol Sequence: 

- Uniform designs: exhaustive search over grid combinations (Huang et al., 2007) 


|pic1| |pic2| |pic3| |pic4|

.. |pic1| image::  ./images/Demo_Grid.png 
   :width: 45%

.. |pic2| image::  ./images/Demo_Random.png 
   :width: 45%


.. |pic3| image::  ./images/Demo_LHS.png
   :width: 45%

.. |pic4| image::  ./images/Demo_Sobol.png
   :width: 45%


Advantage and Disadvantage of One Shot Batch Designs.

- Easy to be paralleled, trials can be generated without too much burden. 

- However, the information of existing experiments is not utilized, which is not very efficient. 



Grid Search 
------------------------------------------------
Grid Search and Sklearn SVM interface::

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
::
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
::
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
::
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

        
Reference list 
_______________

.. [Bergstra2012] James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb):281–305, 2012.
.. [Zhang2019] Hyperparameter Tuning Methods in Automated Machine Learning. (In Chinese) Submitted.