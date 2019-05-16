.. test documentation master file, created by
   sphinx-quickstart on Fri Dec 10 09:13:46 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sequential Meta Machine Learning
==========================================

SeqMML is an open-source python package developed for automated machine learning (AutoML) with focus on hyperparameter optimization. Unlike traditional Batch or Bayesian optimization methods, we propose to use the sequential uniform designs for the following advantages:

- Representative sampling of hyperparameter space: uniformly distributed trials tend to have a better exploration of the hyperparameter space and avoids being trapped into local optima. 

- Free surrogate modeling: it is free from the meta-model estimation and acquisition optimization procedures, where Bayesian optimization may suffer especially for high-dimensional problems.

- Parallel computing: at each stage, the expensive model evaluation procedure could be conducted in parallel


**What is AutoML?** 

- AutoML is to perform Automated Machine Learning model/algorithm selection and hyperparameter tuning.

- It also targets progressive automation of data preprocessing, feature extraction/transformation, postprocessing and interpretation.

- Hyperparameter optimization, a.k.a. (hyper) paramater tuning, plays a central role in AutoML pipelines. 

.. image:: ./images/AutoML_v1.png

**However, it is not easy.** 

- Hyperparameters can be continuous, integer-valued or categorical, e.g. regularization parameters, kernel bandwidths, learning rate, tree depth, batch size, number of layers, type of activation.

-  AutoML is of combinatorial nature, therefore a challenging problem with curse of dimensionality.

- Robustness and reproducibility of optimal configuration depend not only on the specific algorithm, but also on the specific dataset.


Contents:
-----------

.. toctree::
   :maxdepth: 2

   installation.rst
   pysequd.rst
   pybatdoe.rst
   pybayopt.rst
   examples.rst
   modules.rst
   resources.rst