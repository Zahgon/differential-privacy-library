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
The classic geometric mechanism for differential privacy, and its derivatives.
"""
from numbers import Integral

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from diffprivlib.utils import copy_docstring


class Geometric(DPMechanism):
    r"""
    The classic geometric mechanism for differential privacy, as first proposed by Ghosh, Roughgarden and Sundararajan.
    Extended to allow for non-unity sensitivity.

    Paper link: https://arxiv.org/pdf/0811.2841.pdf

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
    def __init__(self, *, epsilon, sensitivity=1, random_state=None):
        super().__init__(epsilon=epsilon, delta=0.0, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = - self.epsilon / self.sensitivity if self.sensitivity > 0 else - float("inf")

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        pass

    def _check_all(self, value):
        pass

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
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
        value : int
            The value to be randomised.

        Returns
        -------
        int
            The randomised value.

        """
        pass


class GeometricTruncated(Geometric, TruncationAndFoldingMixin):
    r"""
    The truncated geometric mechanism, where values that fall outside a pre-described range are mapped back to the
    closest point within the range.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : int
        The lower bound of the mechanism.

    upper : int
        The upper bound of the mechanism.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, sensitivity=1, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @classmethod
    def _check_bounds(cls, lower, upper):
        pass

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        pass

    @copy_docstring(DPMechanism.bias)
    def variance(self, value):
        pass

    def _check_all(self, value):
        pass

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        pass


class GeometricFolded(Geometric, TruncationAndFoldingMixin):
    r"""
    The folded geometric mechanism, where values outside a pre-described range are folded back toward the domain around
    the closest point within the domain.
    Half-integer bounds are permitted.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : int or float
        The lower bound of the mechanism.  Must be integer or half-integer -valued.

    upper : int or float
        The upper bound of the mechanism.  Must be integer or half-integer -valued.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, sensitivity=1, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @classmethod
    def _check_bounds(cls, lower, upper):
        pass

    def _fold(self, value):
        pass

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        pass

    @copy_docstring(DPMechanism.bias)
    def variance(self, value):
        pass

    def _check_all(self, value):
        pass

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        pass
