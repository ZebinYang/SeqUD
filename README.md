# SeqMM

[![Build Status](https://travis-ci.com/ZebinYang/SeqMM.svg?branch=master)](https://travis-ci.org/joerick/cibuildwheel)

# Installation

- Enviroment: Linux + Python 3

```sheel
pip install git+https://github.com/ZebinYang/SeqMM.git
```

# Example
```python
from sklearn import svm
from pySeqUD.search import SeqUDSklearn
from sklearn.model_selection import KFold
iris = datasets.load_iris()
ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'wrapper': np.exp2}, 
       'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'wrapper': np.exp2}}
Level_Number = 20
estimator = svm.SVC()
cv = KFold(n_splits=5, random_state=1, shuffle=True)
clf = SeqUDSklearn(estimator, cv, ParaSpace, Level_Number, n_jobs = 10, refit = True, verbose = True)
```

# Benchmark Methods:

Spearmint: https://github.com/JasperSnoek/spearmint

Hyperopt: https://github.com/hyperopt/hyperopt

SMAC: https://github.com/automl/SMAC3


# Reference:
Zebin Yang, Aijun Zhang and Ji Zhu. (2019) Hyperparameter Optimization via Sequential Uniform Designs. Submitted. 
