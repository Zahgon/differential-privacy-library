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
The vector mechanism in differential privacy, for producing perturbed objectives
"""
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.utils import copy_docstring


class Vector(DPMechanism):
    r"""
    The vector mechanism in differential privacy.

    The vector mechanism is used when perturbing convex objective functions.
    Full paper: http://www.jmlr.org/papers/volume12/chaudhuri11a/chaudhuri11a.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    function_sensitivity : float
        The function sensitivity of the mechanism.  Must be in [0, ∞).

    data_sensitivity : float, default: 1.0
        The data sensitivity of the mechanism.  Must be in [0, ∞).

    dimension : int
        Function input dimension.  This dimension relates to the size of the input vector of the function being
        considered by the mechanism.  This corresponds to the size of the random vector produced by the mechanism. Must
        be in [1, ∞).

    alpha : float, default: 0.01
        Regularisation parameter.  Must be in (0, ∞).

    n : int, default: 1
        Size of the training dataset, required to calibrate the influence of the random vector in the objective.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, function_sensitivity, data_sensitivity=1.0, dimension, alpha=0.01, n=1,
                 random_state=None):
        super().__init__(epsilon=epsilon, delta=0.0, random_state=random_state)
        self.function_sensitivity, self.data_sensitivity = self._check_sensitivity(function_sensitivity,
                                                                                   data_sensitivity)
        self.dimension = self._check_dimension(dimension)
        self.alpha = self._check_alpha(alpha)
        self.n = int(n)

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    @classmethod
    def _check_alpha(cls, alpha):
        pass

    @classmethod
    def _check_dimension(cls, vector_dim):
        pass

    @classmethod
    def _check_sensitivity(cls, function_sensitivity, data_sensitivity):
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

        If `value` is a method of two outputs, they are taken as `f` and `fprime` (i.e., its gradient), and both are
        perturbed accordingly.

        Parameters
        ----------
        value : method
            The function to be randomised.

        Returns
        -------
        method
            The randomised method.

        """
        def output_func(*args):
            pass
        pass
