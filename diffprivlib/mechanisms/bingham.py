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
The Bingham mechanism in differential privacy, for estimating the first eigenvector of a covariance matrix.
"""
import secrets
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.utils import copy_docstring


class Bingham(DPMechanism):
    r"""
    The Bingham mechanism in differential privacy.

    Used to estimate the first eigenvector (associated with the largest eigenvalue) of a covariance matrix.

    Paper link: http://eprints.whiterose.ac.uk/123206/7/simbingham8.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, sensitivity=1.0, random_state=None):
        super().__init__(epsilon=epsilon, delta=0, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)

        if isinstance(self._rng, secrets.SystemRandom):
            self._rng = np.random.default_rng()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        pass

    def _check_all(self, value):
        pass

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        pass

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        pass

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : numpy array
            The data to be randomised.

        Returns
        -------
        numpy array
            The randomised eigenvector.

        """
        pass
