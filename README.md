# SeqMM

[![Build Status](https://travis-ci.com/ZebinYang/SeqMM.svg?branch=master)](https://travis-ci.com/ZebinYang/SeqMM.svg?branch=master)

# Installation

- Enviroment: Linux + Python 3

```sheel
pip install git+https://github.com/ZebinYang/SeqMM.git
```

# Example
```python
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score
from seqmm.pysequd import SeqUDSklearn

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

clf = SeqUDSklearn(estimator, cv, ParaSpace, Level_Number, scoring = score_metric, n_jobs = 2, refit = True, verbose = True)
clf.fit(x, y)
clf.plot_scores()
```

# Benchmark Methods:

Spearmint: https://github.com/JasperSnoek/spearmint

Hyperopt: https://github.com/hyperopt/hyperopt

SMAC: https://github.com/automl/SMAC3


# Reference:
Zebin Yang, Aijun Zhang and Ji Zhu. (2019) Hyperparameter Optimization via Sequential Uniform Designs. Submitted. 
