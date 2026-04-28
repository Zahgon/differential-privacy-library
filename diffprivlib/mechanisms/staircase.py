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
The staircase mechanism in differential privacy.
"""
import secrets
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.laplace import Laplace
from diffprivlib.utils import copy_docstring


class Staircase(Laplace):
    r"""
    The staircase mechanism in differential privacy.

    The staircase mechanism is an optimisation of the classical Laplace Mechanism (:class:`.Laplace`), described as a
    "geometric mixture of uniform random variables".
    Paper link: https://arxiv.org/pdf/1212.1186.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    gamma : float, default: 1 / (1 + exp(epsilon/2))
        Value of the tuning parameter gamma for the mechanism.  Must be in [0, 1].

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, sensitivity, gamma=None, random_state=None):
        super().__init__(epsilon=epsilon, delta=0, sensitivity=sensitivity, random_state=random_state)
        self.gamma = self._check_gamma(gamma, epsilon=self.epsilon)

        if isinstance(self._rng, secrets.SystemRandom):
            self._rng = np.random.default_rng()

    @classmethod
    def _check_gamma(cls, gamma, epsilon=None):
        pass

    @copy_docstring(Laplace._check_all)
    def _check_all(self, value):
        pass

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        pass

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        pass

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        pass
