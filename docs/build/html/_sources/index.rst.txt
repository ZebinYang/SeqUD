.. test documentation master file, created by
   sphinx-quickstart on Fri Dec 10 09:13:46 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SeqMML's documentation!
==========================================

This is an open-source python package developed for automated machine learning, in particular for hyperparameter optimization problems. We propose a sequential uniform design method, as an alternative to the well known Bayesian optimization approach.


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
-------------
.. toctree::
   :maxdepth: 2

   installation.rst
   pybatdoe.rst
   pybayopt.rst
   pysequd.rst
   examples.rst
   apis.rst