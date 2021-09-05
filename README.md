# Sequential Uniform Design

# Installation

- Enviroment: 
    - Python 3
    - C++ compiler
    - swig 3
    
Assume you have figured out the above environment, the most convenient way for installation is via the pip command. 
```sheel
pip install git+https://github.com/ZebinYang/SeqUD.git
```

More details can be found in [documentation](https://zebinyang.github.io/SeqUD/build/html/index.html).

# Examples

### Function optimization

The following codes can perform function maximization. The configuration is quite simple: define the function, parameter space, and then call the fmaxfunction in the SeqUD module. 

```python 
import numpy as np 
from matplotlib import pylab as plt
from sequd import SeqUD

def octopus(parameters):
    x1 = parameters['x1']
    x2 = parameters['x2']
    y = 2 * np.cos(10*  x1) * np.sin(10 * x2) + np.sin(10 * x1 * x2)
    return  y

ParaSpace = {'x1': {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x}, 
             'x2': {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x}}

clf = SeqUD(ParaSpace, max_runs=100, random_state=1, verbose=True)
clf.fmax(octopus)
```

Let's visualize the trials points.
```python
def plot_trajectory(xlim, ylim, func, clf, title):
    grid_num = 25
    xlist = np.linspace(xlim[0], xlim[1], grid_num)
    ylist = np.linspace(ylim[0], ylim[1], grid_num)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros((grid_num,grid_num))
    for i, x1 in enumerate(xlist):
        for j, x2 in enumerate(ylist):
            Z[j, i] = func({"x1": x1, "x2": x2})

    cp = plt.contourf(X, Y, Z)
    plt.scatter(clf.logs.loc[:, ['x1']], 
                clf.logs.loc[:, ['x2']], color="red")
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.colorbar(cp)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)

plot_trajectory([0, 1], [0, 1], octopus, clf, "SeqUD")
```
 ![octopus_demo](https://github.com/ZebinYang/seqmml/blob/master/docs/source/images/octopus_demo.png)


### Tuning sklearn hyperparameters

To optimize the hyperparameters in sklearn is similar to that of function optimization. 
```python
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import KFold 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from sequd import SeqUD

sx = MinMaxScaler()
dt = datasets.load_breast_cancer()
x = sx.fit_transform(dt.data)
y = dt.target

ParaSpace = {'C':     {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
             'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

estimator = svm.SVC()
score_metric = make_scorer(accuracy_score, True)
cv = KFold(n_splits=5, random_state=0, shuffle=True)

clf = SeqUD(ParaSpace, n_runs_per_stage=20, n_jobs=1, estimator=estimator, cv=cv, scoring=score_metric, refit=True, verbose=True)
clf.fit(x, y)
```

```python

def plot_trajectory(Z, clf, title):
    levels = [0.2, 0.4, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    cp = plt.contourf(X, Y, Z, levels)
    plt.colorbar(cp)
    plt.xlabel('Log2_C')
    plt.ylabel('Log2_gamma')
    plt.scatter(np.log2(clf.logs.loc[:, ['C']]), 
                np.log2(clf.logs.loc[:, ['gamma']]), color="red")
    plt.title(title)

grid_num = 25
xlist = np.linspace(-6, 16, grid_num)
ylist = np.linspace(-16, 6, grid_num)
X, Y = np.meshgrid(xlist, ylist)
Z = np.zeros((grid_num,grid_num))
for i, C in enumerate(xlist):
    for j, gamma in enumerate(ylist):
        estimator = svm.SVC(C=2 ** C, gamma=2 ** gamma)
        out = cross_val_score(estimator, x, y, cv=cv, scoring=score_metric)
        Z[j, i] = np.mean(out)
plt.figure(figsize = (6, 4.5))
plot_trajectory(Z, clf, "SeqUD")
```

 ![svm_demo](https://github.com/ZebinYang/sequd/blob/master/docs/source/images/svm_demo.png).

More examples can be referred to the [documentation](https://zebinyang.github.io/sequd/build/html/examples.html)


# Benchmark Methods:

Spearmint: https://github.com/JasperSnoek/spearmint

Hyperopt: https://github.com/hyperopt/hyperopt

SMAC: https://github.com/automl/SMAC3

# Contact:
If you find any bugs or have any suggestions, please contact us via email: yangzb2010@hku.hk or ajzhang@hku.hk.

# Citations:
@article{yang2021hyperparameter,
	author  = {Zebin Yang and Aijun Zhang},
	title   = {Hyperparameter Optimization via Sequential Uniform Designs},
	journal = {Journal of Machine Learning Research},
	year    = {2021},
	volume  = {22},
	number  = {149},
	pages   = {1-47},
	url     = {http://jmlr.org/papers/v22/20-058.html}
}
