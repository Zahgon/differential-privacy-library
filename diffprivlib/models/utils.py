# MIT License
#
# Copyright (C) IBM Corporation 2020
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
Utilities for use in machine learning models
"""
import warnings
from numbers import Integral

import numpy as np
from scipy.linalg import null_space

from diffprivlib.mechanisms import LaplaceBoundedDomain, Bingham
from diffprivlib.utils import PrivacyLeakWarning, check_random_state


def covariance_eig(array, epsilon=1.0, norm=None, dims=None, eigvals_only=False, random_state=None):
    r"""
    Return the eigenvalues and eigenvectors of the covariance matrix of `array`, satisfying differential privacy.

    Paper link: http://papers.nips.cc/paper/9567-differentially-private-covariance-estimation.pdf

    Parameters
    ----------
    array : array-like, shape (n_samples, n_features)
        Matrix for which the covariance matrix is sought.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    norm : float, optional
        The max l2 norm of any row of the input array.  This defines the spread of data that will be protected by
        differential privacy.

        If not specified, the max norm is taken from the data, but will result in a :class:`.PrivacyLeakWarning`, as it
        reveals information about the data.  To preserve differential privacy fully, `norm` should be selected
        independently of the data, i.e. with domain knowledge.

    dims : int, optional
        Number of eigenvectors to return.  If `None`, return all eigenvectors.

    eigvals_only : bool, default: False
        Only return the eigenvalue estimates.  If True, all the privacy budget is spent on estimating the eigenvalues.

    random_state : int or RandomState, optional
        Controls the randomness of the model.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    Returns
    -------
    w : (n_features) array
        The eigenvalues, each repeated according to its multiplicity.

    v : (n_features, dims) array
        The normalized (unit "length") eigenvectors, such that the column ``v[:,i]`` is the eigenvector corresponding to
        the eigenvalue ``w[i]``.

    """
    pass
