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
The uniform mechanism in differential privacy.
"""
from numbers import Real

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.mechanisms.laplace import Laplace
from diffprivlib.utils import copy_docstring


class Uniform(DPMechanism):
    r"""
    The Uniform mechanism in differential privacy.

    This emerges as a special case of the :class:`.LaplaceBoundedNoise` mechanism when epsilon = 0.
    Paper link: https://arxiv.org/pdf/1810.00877.pdf

    Parameters
    ----------
    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 0.5].

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, delta, sensitivity, random_state=None):
        super().__init__(epsilon=0.0, delta=delta, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        pass

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        pass

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        pass

    def _check_all(self, value):
        pass

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        pass
