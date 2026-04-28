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
The classic Gaussian mechanism in differential privacy, and its derivatives.
"""
from math import erf
from numbers import Real, Integral

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism, bernoulli_neg_exp
from diffprivlib.mechanisms.geometric import Geometric
from diffprivlib.mechanisms.laplace import Laplace
from diffprivlib.utils import copy_docstring


class Gaussian(DPMechanism):
    r"""The Gaussian mechanism in differential privacy.

    First proposed by Dwork and Roth in "The algorithmic foundations of differential privacy" [DR14]_.  Samples from the
    Gaussian distribution are generated using two samples from `random.normalvariate` as detailed in [HB21b]_, to
    prevent against reconstruction attacks due to limited floating point precision.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, 1].  For ``epsilon > 1``, use
        :class:`.GaussianAnalytic`.

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    References
    ----------
    .. [DR14] Dwork, Cynthia, and Aaron Roth. "The algorithmic foundations of differential privacy." Found. Trends
        Theor. Comput. Sci. 9, no. 3-4 (2014): 211-407.

    .. [HB21b] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy." arXiv preprint
        arXiv:2107.10138 (2021).

    """
    def __init__(self, *, epsilon, delta, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        pass

    def _check_all(self, value):
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


class GaussianAnalytic(Gaussian):
    r"""The analytic Gaussian mechanism in differential privacy.

    As first proposed by Balle and Wang in "Improving the Gaussian Mechanism for Differential Privacy: Analytical
    Calibration and Optimal Denoising".

    Paper link: https://arxiv.org/pdf/1805.06530.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, delta, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        self._scale = self._find_scale()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    def _check_all(self, value):
        pass

    def _find_scale(self):
        def b_minus(val):
            pass
        def b_plus(val):
            pass
        def objective(sigma, epsilon_, delta_, sensitivity_):
            pass
        def phi(val):
            pass
        pass


class GaussianDiscrete(DPMechanism):
    r"""The Discrete Gaussian mechanism in differential privacy.

    As proposed by Canonne, Kamath and Steinke, re-purposed for approximate :math:`(\epsilon,\delta)`-differential
    privacy.

    Paper link: https://arxiv.org/pdf/2004.00010.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].

    sensitivity : int, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, delta, sensitivity=1, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = self._find_scale()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        pass

    def _check_all(self, value):
        pass

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        pass

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        pass

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        pass

    def _find_scale(self):
        """Determine the scale of the mechanism's distribution given epsilon and delta.
        """
        pass
