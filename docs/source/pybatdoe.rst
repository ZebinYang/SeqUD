One-shot Batch Designs
==============================


Introduction 
------------------
Batch designs can generate all the experimental trials before conducting any experiments.
Here, we introduce five simple strategies.

- Grid Search: exhaustive search over grid combinations

- Random Search:  more flexible than grid experiments, when not all hyperparameters are equally important. Furthermore, new trials can be added without adjustment and the experiments can also be stopped any time.  ([Bergstra2012]_)

- Latin Hypercube Sampling: near-random sample (LHS; [McKay1978]_)

- Sobol Sequence: quasi-random low-discrepancy sequence ([Sobol967]_)

The figures (derived from [Zhang2019]_) present a demo of the four mentioned sampling approach, including Grid search (top left), Random search (top right), Latin hypercube (bottom left) and Sobol Sequence (bottom right).

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

- To select an appropriate number of design points is always difficult, with potential over-sampling and under-sampling problems.

Code Examples 
--------------

Grid Search::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm import GridSearch

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = GridSearch(ParaSpace, max_runs = 100, n_jobs = 10, 
                    estimator = estimator, cv = cv, refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()

Random Search::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm.pybatdoe import RandSearch

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = RandSearch(ParaSpace, max_runs = 100, n_jobs = 10, 
                    estimator = estimator, cv = cv, refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()

Latin Hypercube Sampling::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm import LHSSearch

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = LHSSearch(ParaSpace, max_runs = 100, n_jobs = 10, 
                    estimator = estimator, cv = cv, refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()
        
        
Sobol Sequence::

        import numpy as np 
        from sklearn import svm
        from sklearn import datasets
        from sklearn.model_selection import KFold
        from seqmm import SobolSearch

        iris = datasets.load_iris()
        ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
        estimator = svm.SVC()
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        clf = SobolSearch(ParaSpace, max_runs = 100, n_jobs = 10, 
                    estimator = estimator, cv = cv, refit = True, verbose = True)
        clf.fit(iris.data, iris.target)
        clf.plot_scores()        

        
Reference list 
_______________


.. [Sobol967] Sobol,I.M. (1967), "Distribution of points in a cube and approximate evaluation of integrals". Zh. Vych. Mat. Mat. Fiz. 7: 784–802 (in Russian); U.S.S.R Comput. Maths. Math. Phys. 7: 86–112 (in English)

.. [McKay1978] McKay, M.D., Beckman, R.J. and Conover, W.J., 1979. Comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics, 21(2), pp.239-245.

.. [Bergstra2012] James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb):281–305, 2012.

.. [Zhang2019] Hyperparameter Tuning Methods in Automated Machine Learning. (In Chinese) Submitted.