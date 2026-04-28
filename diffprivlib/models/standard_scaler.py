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
Standard Scaler with differential privacy
"""
import warnings

import numpy as np
import sklearn.preprocessing as sk_pp
from sklearn.preprocessing._data import _handle_zeros_in_scale

# TODO: remove when sklearn 1.6 a min req
try:
    from sklearn.utils.validation import validate_data
except ImportError:
    from sklearn.base import BaseEstimator
    validate_data = BaseEstimator._validate_data

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning, check_random_state
from diffprivlib.tools import nanvar, nanmean
from diffprivlib.validation import DiffprivlibMixin


def _incremental_mean_and_var(X, epsilon, bounds, last_mean, last_variance, last_sample_count, random_state=None):
    # Initialising new accountant, as budget is tracked in main class. Subject to review in line with GH issue #21
    pass


# noinspection PyPep8Naming,PyAttributeOutsideInit
class StandardScaler(sk_pp.StandardScaler, DiffprivlibMixin):
    """Standardize features by removing the mean and scaling to unit variance, calculated with differential privacy
    guarantees.  Differential privacy is guaranteed on the learned scaler with respect to the training sample; the
    transformed output will certainly not satisfy differential privacy.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the (differentially private) mean of the training samples or zero if `with_mean=False`, and `s` is the
    (differentially private) standard deviation of the training samples or one if `with_std=False`.

    Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in
    the training set.  Mean and standard deviation are then stored to be used on later data using the `transform`
    method.

    For further information, users are referred to :class:`sklearn.preprocessing.StandardScaler`.

    Parameters
    ----------
    epsilon : float, default: 1.0
        The privacy budget to be allocated to learning the mean and variance of the training sample.  If
        `with_std=True`,  the privacy budget is split evenly between mean and variance (the mean must be calculated even
        when `with_mean=False`, as it is used in the calculation of the variance.

    bounds : tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    copy : boolean, default: True
        If False, try to avoid a copy and do inplace scaling instead.  This is not guaranteed to always work inplace;
        e.g. if the data is not a NumPy array, a copy may still be returned.

    with_mean : boolean, True by default
        If True, center the data before scaling.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently, unit standard deviation).

    random_state : int or RandomState, optional
        Controls the randomness of the model.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    scale_ : ndarray or None, shape (n_features,)
        Per feature relative scaling of the data.  This is calculated using `np.sqrt(var_)`.  Equal to ``None`` when
        ``with_std=False``.

    mean_ : ndarray or None, shape (n_features,)
        The mean value for each feature in the training set.  Equal to ``None`` when ``with_mean=False``.

    var_ : ndarray or None, shape (n_features,)
        The variance for each feature in the training set.  Used to compute `scale_`.  Equal to ``None`` when
        ``with_std=False``.

    n_samples_seen_ : int or array, shape (n_features,)
        The number of samples processed by the estimator for each feature.  If there are not missing samples, the
        ``n_samples_seen`` will be an integer, otherwise it will be an array.
        Will be reset on new calls to fit, but increments across ``partial_fit`` calls.

    See also
    --------
    :class:`sklearn.preprocessing.StandardScaler`
        Vanilla scikit-learn version, without differential privacy.

    :class:`.PCA`
        Further removes the linear correlation across features with 'whiten=True'.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in transform.

    """  # noqa
    def __init__(self, *, epsilon=1.0, bounds=None, copy=True, with_mean=True, with_std=True, random_state=None,
                 accountant=None):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.epsilon = epsilon
        self.bounds = bounds
        self.random_state = random_state
        self.accountant = BudgetAccountant.load_default(accountant)

    def partial_fit(self, X, y=None, sample_weight=None):
        """Online computation of mean and std with differential privacy on X for later scaling.  All of X is processed
        as a single batch.  This is intended for cases when `fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b in Chan, Tony F., Gene H. Golub, and
        Randall J. LeVeque. "Algorithms for computing the sample variance: Analysis and recommendations." The American
        Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation used for later scaling along the features axis.

        y
            Ignored

        sample_weight
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        """
        pass
