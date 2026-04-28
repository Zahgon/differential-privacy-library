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
K-means clustering algorithm satisfying differential privacy.
"""
import warnings

import numpy as np
import sklearn.cluster as sk_cluster

# TODO: remove when sklearn 1.6 a min req
try:
    from sklearn.utils.validation import validate_data
except ImportError:
    from sklearn.base import BaseEstimator
    validate_data = BaseEstimator._validate_data

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import LaplaceBoundedDomain, GeometricFolded
from diffprivlib.utils import PrivacyLeakWarning, check_random_state
from diffprivlib.validation import DiffprivlibMixin


class KMeans(sk_cluster.KMeans, DiffprivlibMixin):
    r"""K-Means clustering with differential privacy.

    Implements the DPLloyd approach presented in [SCL16]_, leveraging the :class:`sklearn.cluster.KMeans` class for full
    integration with Scikit Learn.

    Parameters
    ----------
    n_clusters : int, default: 8
        The number of clusters to form as well as the number of centroids to generate.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds :  tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    random_state : int or RandomState, optional
        Controls the randomness of the model.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.  If the algorithm stops before fully converging, these will not be consistent
        with ``labels_``.

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [SCL16] Su, Dong, Jianneng Cao, Ninghui Li, Elisa Bertino, and Hongxia Jin. "Differentially private k-means
        clustering." In Proceedings of the sixth ACM conference on data and application security and privacy, pp. 26-37.
        ACM, 2016.

    """

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        sk_cluster.KMeans, "n_clusters", "random_state")

    def __init__(self, n_clusters=8, *, epsilon=1.0, bounds=None, random_state=None, accountant=None, **unused_args):
        super().__init__(n_clusters=n_clusters, random_state=random_state)

        self.epsilon = epsilon
        self.bounds = bounds
        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

        self.cluster_centers_ = None
        self.bounds_processed = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self._n_threads = 1

    def fit(self, X, y=None, sample_weight=None):
        """Computes k-means clustering with differential privacy.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            not used, present here for API consistency by convention.

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : class

        """
        pass

    def _init_centers(self, dims, random_state):
        pass

    def _distances_labels(self, X, centers):
        pass

    def _update_centers(self, X, centers, labels, dims, total_iters, random_state):
        """Updates the centers of the KMeans algorithm for the current iteration, while satisfying differential
        privacy.

        Differential privacy is satisfied by adding (integer-valued, using :class:`.GeometricFolded`) random noise to
        the count of nearest neighbours to the previous cluster centers, and adding (real-valued, using
        :class:`.LaplaceBoundedDomain`) random noise to the sum of values per dimension.

        """
        pass

    def _split_epsilon(self, dims, total_iters, rho=0.225):
        """Split epsilon between sum perturbation and count perturbation, as proposed by Su et al.

        Parameters
        ----------
        dims : int
            Number of dimensions to split `epsilon` across.

        total_iters : int
            Total number of iterations to split `epsilon` across.

        rho : float, default: 0.225
            Coordinate normalisation factor.

        Returns
        -------
        epsilon_0 : float
            The epsilon value for satisfying differential privacy on the count of a cluster.

        epsilon_i : float
            The epsilon value for satisfying differential privacy on each dimension of the center of a cluster.

        """
        pass

    def _calc_iters(self, n_dims, n_samples, rho=0.225):
        """Calculate the number of iterations to allow for the KMeans algorithm."""
        pass
