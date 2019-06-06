import os
import time
import shutil
import numpy as np
import pandas as pd

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from .bayopt_base import BayoptBase


class SMACOPT(BayoptBase):
    """
    Interface of SMAC (Bayesian Optimization).

    Parameters
    ----------
    :type  para_space: dict or list of dictionaries
    :param para_space: It has three types:

        Continuous:
            Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
            `Wrapper`, a callable function for wrapping the values.
        Integer:
            Specify `Type` as `integer`, and include the keys of `Mapping` (a list with all the sortted integer elements).
        Categorical:
            Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories).

    :type max_runs: int, optional, default = 100
    :param max_runs: The maximum number of trials to be evaluated. When this values is reached,
        then the algorithm will stop.

    :type  estimator: estimator object
    :param estimator: This is assumed to implement the scikit-learn estimator interface.

    :type  cv: cross-validation method, an sklearn object.
    :param cv: e.g., `StratifiedKFold` and KFold` is used.

    :type scoring: string, callable, list/tuple, dict or None, optional, default = None
    :param scoring: A sklearn type scoring function.
        If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.

    :type refit: boolean, or string, optional, default = True
    :param refit: It controls whether to refit an estimator using the best found parameters on the whole dataset.

    :type rand_seed: int, optional, default = 0
    :param rand_seed: The random seed for optimization.

    :type verbose: boolean, optional, default = False
    :param verbose: It controls whether the searching history will be printed.


    Examples
    ----------
    >>> import numpy as np
    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> from seqmml import SMACOPT
    >>> from sklearn.model_selection import KFold
    >>> iris = datasets.load_iris()
    >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2},
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
    >>> estimator = svm.SVC()
    >>> cv = KFold(n_splits=5, random_state=0, shuffle=True)
    >>> clf = SMACOPT(ParaSpace, max_runs = 100,
                estimator = estimator, cv = cv, scoring = None, refit = None, rand_seed = 0, verbose = False)
    >>> clf.fit(iris.data, iris.target)

    Attributes
    ----------
    :vartype best_score\_: float
    :ivar best_score\_: The best average cv score among the evaluated trials.

    :vartype best_params\_: dict
    :ivar best_params\_: Parameters that reaches `best_score_`.

    :vartype best_estimator\_: sklearn estimator
    :ivar best_estimator\_: The estimator refitted based on the `best_params_`.
        Not available if estimator = None or `refit=False`.

    :vartype search_time_consumed\_: float
    :ivar search_time_consumed\_: Seconds used for whole searching procedure.

    :vartype refit_time\_: float
    :ivar refit_time\_: Seconds used for refitting the best model on the whole dataset.
        Not available if estimator = None or `refit=False`.
    """

    def __init__(self, para_space, max_runs=100, estimator=None, cv=None,
                 scoring=None, refit=False, rand_seed=0, verbose=False):

        super(SMACOPT, self).__init__(para_space, max_runs, verbose)

        self.cv = cv
        self.refit = refit
        self.scoring = scoring
        self.estimator = estimator
        self.rand_seed = rand_seed
        self.method = "SMAC"

        self.cs = ConfigurationSpace()
        for item, values in self.para_space.items():
            if values['Type'] == "continuous":
                para = UniformFloatHyperparameter(item, values['Range'][0], values['Range'][1])
            elif values['Type'] == "integer":
                para = UniformIntegerHyperparameter(item, min(values['Mapping']), max(values['Mapping']))
            elif values['Type'] == "categorical":
                para = CategoricalHyperparameter(item, values['Mapping'])
            self.cs.add_hyperparameter(para)

    def obj_func(self, cfg):

        cfg = {k: cfg[k] for k in cfg}
        next_params = pd.DataFrame(cfg, columns=self.para_names, index=[0])
        parameters = {}
        for item, values in self.para_space.items():
            if (values['Type'] == "continuous"):
                parameters[item] = values['Wrapper'](float(next_params[item].iloc[0]))
            elif (values['Type'] == "integer"):
                parameters[item] = int(next_params[item].iloc[0])
            elif (values['Type'] == "categorical"):
                parameters[item] = next_params[item].iloc[0]

        score = self.wrapper_func(parameters)
        logs_aug = parameters
        logs_aug.update({"score": score})
        logs_aug = pd.DataFrame(logs_aug, index=[self.iteration])
        self.logs = pd.concat([self.logs, logs_aug]).reset_index(drop=True)

        if self.verbose:
            self.pbar.update(1)
            self.iteration += 1
            self.pbar.set_description("Iteration %d:" % self.iteration)
            self.pbar.set_postfix_str("Current Best Score = %.5f" % (self.logs.loc[:, "score"].max()))
        return -score

    def _run(self, wrapper_func):
        """
        Main loop for searching the best hyperparameters.

        """
        self.wrapper_func = wrapper_func
        file_dir = "./temp" + "/" + str(time.time()) + str(np.random.rand(1)[0]) + "/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                             "runcount-limit": self.max_runs,  # maximum function evaluations
                             "cs": self.cs,               # configuration space
                             "deterministic": "true",
                             "output_dir": file_dir,
                             "abort_on_first_run_crash": False})
        self.smac = SMAC(scenario=scenario, rng=np.random.seed(self.rand_seed), tae_runner=self.obj_func)
        self.smac.solver.intensifier.tae_runner.use_pynisher = False  # turn off the limit for resources
        self.smac.optimize()
        shutil.rmtree(file_dir)
