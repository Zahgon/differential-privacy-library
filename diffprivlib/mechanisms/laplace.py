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
The classic Laplace mechanism in differential privacy, and its derivatives.
"""
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from diffprivlib.utils import copy_docstring


class Laplace(DPMechanism):
    r"""
    The classical Laplace mechanism in differential privacy.

    First proposed by Dwork, McSherry, Nissim and Smith [DMNS16]_, with support for (relaxed)
    :math:`(\epsilon,\delta)`-differential privacy [HLM15]_.

    Samples from the Laplace distribution are generated using 4 uniform variates, as detailed in [HB21]_, to prevent
    against reconstruction attacks due to limited floating point precision.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    References
    ----------
    .. [DMNS16] Dwork, Cynthia, Frank McSherry, Kobbi Nissim, and Adam Smith. "Calibrating noise to sensitivity in
        private data analysis." Journal of Privacy and Confidentiality 7, no. 3 (2016): 17-51.

    .. [HLM15] Holohan, Naoise, Douglas J. Leith, and Oliver Mason. "Differential privacy in metric spaces: Numerical,
        categorical and functional data under the one roof." Information Sciences 305 (2015): 256-268.

    .. [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy." arXiv preprint
        arXiv:2107.10138 (2021).

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = None

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        pass

    def _check_all(self, value):
        pass

    def bias(self, value):
        """Returns the bias of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the bias of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The bias of the mechanism at `value`.

        """
        pass

    def variance(self, value):
        """Returns the variance of the mechanism at a given `value`.

        Parameters
        ----------
        value : float
            The value at which the variance of the mechanism is sought.

        Returns
        -------
        bias : float
            The variance of the mechanism at `value`.

        """
        pass

    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        pass

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : float
            The value to be randomised.

        Returns
        -------
        float
            The randomised value.

        """
        pass


class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):
    r"""
    The truncated Laplace mechanism, where values outside a pre-described domain are mapped to the closest point
    within the domain.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

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


class LaplaceFolded(Laplace, TruncationAndFoldingMixin):
    r"""
    The folded Laplace mechanism, where values outside a pre-described domain are folded around the domain until they
    fall within.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        pass

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        pass

    def _check_all(self, value):
        pass

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        pass


class LaplaceBoundedDomain(LaplaceTruncated):
    r"""
    The bounded Laplace mechanism on a bounded domain.  The mechanism draws values directly from the domain using
    rejection sampling, without any post-processing [HABM20]_.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    References
    ----------
    .. [HABM20] Holohan, Naoise, Spiros Antonatos, Stefano Braghin, and Pól Mac Aonghusa. "The Bounded Laplace Mechanism
        in Differential Privacy." Journal of Privacy and Confidentiality 10, no. 1 (2020).

    """
    def _find_scale(self):
        def _delta_c(shape):
            pass
        def _f(shape):
            pass
        pass

    def effective_epsilon(self):
        r"""Gets the effective epsilon of the mechanism, only for strict :math:`\epsilon`-differential privacy.  Returns
        ``None`` if :math:`\delta` is non-zero.

        Returns
        -------
        float
            The effective :math:`\epsilon` parameter of the mechanism.  Returns ``None`` if `delta` is non-zero.

        """
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


class LaplaceBoundedNoise(Laplace):
    r"""
    The Laplace mechanism with bounded noise, only applicable for approximate differential privacy (delta > 0)
    [GDGK18]_.

    Epsilon must be strictly positive, `epsilon` > 0. `delta` must be strictly in the interval (0, 0.5).
     - For zero `epsilon`, use :class:`.Uniform`.
     - For zero `delta`, use :class:`.Laplace`.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 0.5).

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    References
    ----------
    .. [GDGK18] Geng, Quan, Wei Ding, Ruiqi Guo, and Sanjiv Kumar. "Truncated Laplacian Mechanism for Approximate
        Differential Privacy." arXiv preprint arXiv:1810.00877v1 (2018).

    """
    def __init__(self, *, epsilon, delta, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        self._noise_bound = None

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        pass

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        pass

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        pass
