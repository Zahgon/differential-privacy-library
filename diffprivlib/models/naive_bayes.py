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
"""
Gaussian Naive Bayes classifier satisfying differential privacy
"""
import warnings

import numpy as np
import sklearn.naive_bayes as sk_nb
from sklearn.utils.multiclass import _check_partial_fit_first_call

# TODO: remove when sklearn 1.6 a min req
try:
    from sklearn.utils.validation import validate_data
except ImportError:
    from sklearn.base import BaseEstimator
    validate_data = BaseEstimator._validate_data

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import LaplaceBoundedDomain, GeometricTruncated, LaplaceTruncated
from diffprivlib.utils import PrivacyLeakWarning, warn_unused_args, check_random_state
from diffprivlib.validation import DiffprivlibMixin


class GaussianNB(sk_nb.GaussianNB, DiffprivlibMixin):
    r"""Gaussian Naive Bayes (GaussianNB) with differential privacy

    Inherits the :class:`sklearn.naive_bayes.GaussianNB` class from Scikit Learn and adds noise to satisfy differential
    privacy to the learned means and variances.  Adapted from the work presented in [VSB13]_.

    Parameters
    ----------
    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon` for the model.

    bounds :  tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes.  If specified the priors are not adjusted according to the data.

    var_smoothing : float, default: 1e-9
        Portion of the largest variance of all features that is added to variances for calculation stability.

    random_state : int or RandomState, optional
        Controls the randomness of the model.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    class_prior_ : array, shape (n_classes,)
        probability of each class.

    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.

    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class

    var_ : array, shape (n_classes, n_features)
        variance of each feature per class

    epsilon_ : float
        absolute additive value to variances (unrelated to ``epsilon`` parameter for differential privacy)

    References
    ----------
    .. [VSB13] Vaidya, Jaideep, Basit Shafiq, Anirban Basu, and Yuan Hong. "Differentially private naive bayes
        classification." In 2013 IEEE/WIC/ACM International Joint Conferences on Web Intelligence (WI) and Intelligent
        Agent Technologies (IAT), vol. 1, pp. 571-576. IEEE, 2013.

    """

    def __init__(self, *, epsilon=1.0, bounds=None, priors=None, var_smoothing=1e-9, random_state=None,
                 accountant=None):
        super().__init__(priors=priors, var_smoothing=var_smoothing)

        self.epsilon = epsilon
        self.bounds = bounds
        self.random_state = random_state
        self.accountant = BudgetAccountant.load_default(accountant)

    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        pass

    def _update_mean_variance(self, n_past, mu, var, X, random_state, sample_weight=None, n_noisy=None):
        """Compute online update of Gaussian mean and variance.

        Given starting sample count, mean, and variance, a new set of points X return the updated mean and variance.
        (NB - each dimension (column) in X is treated as independent -- you get variance, not covariance).

        Can take scalar mean and variance, or vector mean and variance to simultaneously update a number of
        independent Gaussians.

        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance.  If sample weights were given, this should contain
            the sum of sample weights represented in old mean and variance.

        mu : array-like, shape (number of Gaussians,)
            Means for Gaussians in original set.

        var : array-like, shape (number of Gaussians,)
            Variances for Gaussians in original set.

        random_state : RandomState
            Controls the randomness of the model.  To obtain a deterministic behaviour during randomisation,
            ``random_state`` has to be fixed to an integer.

        sample_weight : ignored
            Ignored in diffprivlib.

        n_noisy : int, optional
            Noisy count of the given class, satisfying differential privacy.

        Returns
        -------
        total_mu : array-like, shape (number of Gaussians,)
            Updated mean for each Gaussian over the combined set.

        total_var : array-like, shape (number of Gaussians,)
            Updated variance for each Gaussian over the combined set.
        """
        pass

    def _noisy_class_counts(self, y, random_state):
        pass
