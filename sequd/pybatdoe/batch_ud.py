import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm_notebook as tqdm

import pyunidoe as pydoe
from .batch_base import BatchBase


class UDSearch(BatchBase):
    """
    Implementation of Uniform Design.

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

    :type max_runs: int, optional, default=100
    :param max_runs: The maximum number of trials to be evaluated. When this values is reached,
        then the algorithm will stop.

    :type max_search_iter: int, optional, default=100
    :param max_search_iter: The maximum number of iterations used to generate uniform design or augmented uniform design.

    :type  estimator: estimator object
    :param estimator: This is assumed to implement the scikit-learn estimator interface.

    :type  cv: cross-validation method, an sklearn object.
    :param cv: e.g., `StratifiedKFold` and KFold` is used.

    :type scoring: string, callable, list/tuple, dict or None, optional, default=None
    :param scoring: A sklearn type scoring function.
        If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.

    :type refit: boolean, or string, optional, default=True
    :param refit: It controls whether to refit an estimator using the best found parameters on the whole dataset.

    :type n_jobs: int or None, optional, optional, default=None
    :param n_jobs: Number of jobs to run in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging. See the package `joblib` for details.

    :type random_state: int, optional, default=0
    :param random_state: The random seed for optimization.

    :type verbose: boolean, optional, default=False
    :param verbose: It controls whether the searching history will be printed.

    Examples
    ----------
    >>> import numpy as np
    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> from sequd import UDSearch
    >>> from sklearn.model_selection import KFold
    >>> iris = datasets.load_iris()
    >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2},
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
    >>> estimator = svm.SVC()
    >>> cv = KFold(n_splits=5, random_state=0, shuffle=True)
    >>> clf = UDSearch(ParaSpace, max_runs=100, max_search_iter=100, estimator=estimator, cv=cv,
                 scoring=None, n_jobs=None, refit=False, random_state=0, verbose=False)
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
        Not available if estimator=None or `refit=False`.
    """

    def __init__(self, para_space, max_runs=100, max_search_iter=100, estimator=None, cv=None,
                 scoring=None, refit=True, n_jobs=None, random_state=0, verbose=False):

        super(UDSearch, self).__init__(para_space, max_runs, n_jobs, verbose)

        self.cv = cv
        self.refit = refit
        self.scoring = scoring
        self.estimator = estimator
        self.random_state = random_state
        self.max_search_iter = max_search_iter
        self.method = "UD Search"

    def _generate_init_design(self):
        """
        This function generates the initial uniform design.

        Returns
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point,
                and columns are used to represent variables.
        """

        ud_space = np.repeat(np.linspace(1 / (2 * self.max_runs), 1 - 1 / (2 * self.max_runs), self.max_runs).reshape([-1, 1]),
                             self.extend_factor_number, axis=1)

        base_ud = pydoe.design_query(n=self.max_runs, s=self.extend_factor_number,
                                     q=self.max_runs, crit="CD2", show_crit=False)
        if base_ud is None:
            base_ud = pydoe.gen_ud_ms(n=self.max_runs, s=self.extend_factor_number, q=self.max_runs, crit="CD2",
                                      maxiter=self.max_search_iter, random_state=self.random_state, nshoot=5)

        if (not isinstance(base_ud, np.ndarray)):
            raise ValueError('Uniform design is not correctly constructed!')

        para_set_ud = np.zeros((self.max_runs, self.extend_factor_number))
        for i in range(self.factor_number):
            loc_min = np.sum(self.variable_number[:(i + 1)])
            loc_max = np.sum(self.variable_number[:(i + 2)])
            for k in range(int(loc_min), int(loc_max)):
                para_set_ud[:, k] = ud_space[base_ud[:, k] - 1, k]
        para_set_ud = pd.DataFrame(para_set_ud, columns=self.para_ud_names)
        return para_set_ud

    def _run(self, obj_func):
        """
        Main loop for searching the best hyperparameters.

        """
        para_set_ud = self._generate_init_design()
        para_set = self._para_mapping(para_set_ud)
        para_set_ud.columns = self.para_ud_names
        candidate_params = [{para_set.columns[j]: para_set.iloc[i, j]
                             for j in range(para_set.shape[1])}
                            for i in range(para_set.shape[0])]
        if self.verbose:
            if self.n_jobs > 1:
                out = Parallel(n_jobs=self.n_jobs)(delayed(obj_func)(parameters) for parameters in tqdm(candidate_params))
            else:
                out = []
                for parameters in tqdm(candidate_params):
                    out.append(obj_func(parameters))
                out = np.array(out)

        else:
            if self.n_jobs > 1:
                out = Parallel(n_jobs=self.n_jobs)(delayed(obj_func)(parameters) for parameters in candidate_params)
            else:
                out = []
                for parameters in candidate_params:
                    out.append(obj_func(parameters))
                out = np.array(out)

        self.logs = para_set_ud.to_dict()
        self.logs.update(para_set)
        self.logs.update(pd.DataFrame(out, columns=["score"]))
        self.logs = pd.DataFrame(self.logs).reset_index(drop=True)
        if self.verbose:
            print("Search completed (%d/%d) with best score: %.5f."
                  % (self.logs.shape[0], self.max_runs, self.logs["score"].max()))
