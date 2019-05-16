# SeqMM

[![Build Status](https://travis-ci.com/ZebinYang/seqmml.svg?branch=master)](https://travis-ci.com/ZebinYang/seqmml.svg?branch=master)

 ![recipe1](https://github.com/linwh8/ModernWebPrograming/raw/master/My_image/recipe.png)
 ![index](https://github.com/ZebinYang/seqmml/blob/master/docs/source/images/octopus_demo.png)

# Installation

- Enviroment: Linux + Python 3

```sheel
pip install git+https://github.com/ZebinYang/seqmml.git
```

# Examples

- Function optimization

```python 
import numpy as np 
from matplotlib import pylab as plt
from seqmml import SeqUD

def octopus(parameters):
    x1 = parameters['x1']
    x2 = parameters['x2']
    y = 2*np.cos(10*x1)*np.sin(10*x2)+np.sin(10*x1*x2)
    return  y

def plot_trajectory(xlim, ylim, func, clf, title):
    grid_num = 25
    xlist = np.linspace(xlim[0], xlim[1], grid_num)
    ylist = np.linspace(ylim[0], ylim[1], grid_num)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros((grid_num,grid_num))
    for i, x1 in enumerate(xlist):
        for j, x2 in enumerate(ylist):
            Z[j,i] = func({"x1": x1, "x2": x2})

    cp = plt.contourf(X, Y, Z)
    plt.scatter(clf.logs.loc[:,['x1']], 
                clf.logs.loc[:,['x2']], color = "red")
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    plt.colorbar(cp)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)

Level_Number = 20
ParaSpace = {'x1': {'Type': 'continuous', 'Range': [0,1], 'Wrapper': lambda x: x}, 
             'x2': {'Type': 'continuous', 'Range': [0,1], 'Wrapper': lambda x: x}}

clf = SeqUD(ParaSpace, max_runs = 100, rand_seed = 1, verbose = True)
clf.fmin(octopus)

plot_trajectory([0,1], [0,1], octopus, clf, "SeqUD")
```
 ![index](https://github.com/ZebinYang/seqmml/tree/master/docs/source/images/octopus_demo.png)

- Tuning sklearn hyperparameters
```python
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from seqmml import SeqUD

def plot_trajectory(Z, clf, title):
    levels = [0.2, 0.4, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    cp = plt.contourf(X, Y, Z, levels)
    plt.colorbar(cp)
    plt.xlabel('Log2_C')
    plt.ylabel('Log2_gamma')
    plt.scatter(np.log2(clf.logs.loc[:,['C']]), 
                np.log2(clf.logs.loc[:,['gamma']]), color = "red")
    plt.title(title)

sx = MinMaxScaler()
dt = datasets.load_breast_cancer()
x = sx.fit_transform(dt.data)
y = dt.target

ParaSpace = {'C':     {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
             'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

estimator = svm.SVC()
score_metric = make_scorer(accuracy_score, True)
cv = KFold(n_splits=5, random_state=0, shuffle=True)

clf = SeqUD(ParaSpace, level_number = 20, n_jobs = 10, estimator = estimator, cv = cv, scoring = score_metric, refit = True, verbose = True)
clf.fit(x, y)

grid_num = 25
xlist = np.linspace(-6, 16, grid_num)
ylist = np.linspace(-16, 6, grid_num)
X, Y = np.meshgrid(xlist, ylist)
Z = np.zeros((grid_num,grid_num))
for i, C in enumerate(xlist):
    for j, gamma in enumerate(ylist):
        estimator = svm.SVC(C=2**C,gamma = 2**gamma)
        out = cross_val_score(estimator, x, y, cv = cv, scoring = score_metric)
        Z[j,i] = np.mean(out)
plt.figure(figsize = (6, 4.5))
plot_trajectory(Z, clf, "SeqUD")
```

[![demo svm results](https://github.com/ZebinYang/seqmml/tree/master/docs/source/images/svm_demo.png)](https://github.com/ZebinYang/seqmml/tree/master/docs/source/images/svm_demo.png)


More examples can be referred to the documentation (URL)


# Benchmark Methods:

Spearmint: https://github.com/JasperSnoek/spearmint

Hyperopt: https://github.com/hyperopt/hyperopt

SMAC: https://github.com/automl/SMAC3


# Reference:
Zebin Yang, Aijun Zhang and Ji Zhu. (2019) Hyperparameter Optimization via Sequential Uniform Designs. Submitted. 
