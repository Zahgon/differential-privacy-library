# MIT License
#
# Copyright (C) IBM Corporation 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
# New BSD License
#
# Copyright (c) 2007–2019 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#      disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#      following disclaimer in the documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of its contributors may be used to endorse or
#      promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
Linear Regression with differential privacy
"""
import warnings

import numpy as np
import sklearn.linear_model as sk_lr
from scipy.optimize import minimize
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES

# TODO: remove when sklearn 1.6 a min req
try:
    from sklearn.utils.validation import validate_data
except ImportError:
    from sklearn.base import BaseEstimator
    validate_data = BaseEstimator._validate_data

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Laplace, LaplaceFolded
from diffprivlib.tools import mean
from diffprivlib.utils import warn_unused_args, PrivacyLeakWarning, check_random_state
from diffprivlib.validation import check_bounds, clip_to_bounds, DiffprivlibMixin


# noinspection PyPep8Naming
def _preprocess_data(X, y, fit_intercept, epsilon=1.0, bounds_X=None, bounds_y=None, copy=True, check_input=True,
                     random_state=None, **unused_args):
    pass


def _construct_regression_obj(X, y, bounds_X, bounds_y, epsilon, alpha, random_state):
    def get_max_sensitivity(y_lower, y_upper, x_lower, x_upper):
        pass
    def inner_obj(omega):
        pass
    def obj(idx):
        pass
    pass


# noinspection PyPep8Naming,PyAttributeOutsideInit
class LinearRegression(sk_lr.LinearRegression, DiffprivlibMixin):
    r"""
    Ordinary least squares Linear Regression with differential privacy.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp) to minimize the residual sum of squares
    between the observed targets in the dataset, and the targets predicted by the linear approximation.  Differential
    privacy is guaranteed with respect to the training sample.

    Differential privacy is achieved by adding noise to the coefficients of the objective function, taking inspiration
    from [ZZX12]_.

    Parameters
    ----------
    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds_X :  tuple
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    bounds_y : tuple
        Same as `bounds_X`, but for the training label set `y`.

    fit_intercept : bool, default: True
        Whether to calculate the intercept for this model.  If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default: True
        If True, X will be copied; else, it may be overwritten.

    random_state : int or RandomState, optional
        Controls the randomness of the model.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.  If multiple targets are passed during the fit (y 2D),
        this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of
        length n_features.

    intercept_ : float or array of shape of (n_targets,)
        Independent term in the linear model.  Set to 0.0 if `fit_intercept = False`.

    References
    ----------
    .. [ZZX12] Zhang, Jun, Zhenjie Zhang, Xiaokui Xiao, Yin Yang, and Marianne Winslett. "Functional mechanism:
        regression analysis under differential privacy." arXiv preprint arXiv:1208.0219 (2012).

    """

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        sk_lr.LinearRegression, "fit_intercept", "copy_X")

    def __init__(self, *, epsilon=1.0, bounds_X=None, bounds_y=None, fit_intercept=True, copy_X=True, random_state=None,
                 accountant=None, **unused_args):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=None)

        self.epsilon = epsilon
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y
        self.random_state = random_state
        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data

        y : array_like, shape (n_samples, n_targets)
            Target values.  Will be cast to X's dtype if necessary

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : returns an instance of self.

        """
        pass

    _preprocess_data = staticmethod(_preprocess_data)
