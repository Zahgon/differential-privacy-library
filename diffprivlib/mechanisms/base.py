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
Base classes for differential privacy mechanisms.
"""
import abc
from copy import copy
import inspect
from numbers import Real

from diffprivlib.utils import check_random_state


class DPMachine(abc.ABC):
    """
    Parent class for :class:`.DPMechanism` and :class:`.DPTransformer`, providing and specifying basic functionality.

    """
    @abc.abstractmethod
    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : int or float or str or method
            The value to be randomised.

        Returns
        -------
        int or float or str or method
            The randomised value, same type as `value`.

        """

    def copy(self):
        """Produces a copy of the class.

        Returns
        -------
        self : class
            Returns the copy.

        """
        pass


class DPMechanism(DPMachine, abc.ABC):
    r"""Abstract base class for all mechanisms.  Instantiated from :class:`.DPMachine`.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, delta, random_state=None):
        self.epsilon, self.delta = self._check_epsilon_delta(epsilon, delta)
        self.random_state = random_state

        self._rng = check_random_state(random_state, True)

    def __repr__(self):
        attrs = inspect.getfullargspec(self.__class__).kwonlyargs
        attr_output = []

        for attr in attrs:
            attr_output.append(attr + "=" + repr(self.__getattribute__(attr)))

        return str(self.__module__) + "." + str(self.__class__.__name__) + "(" + ", ".join(attr_output) + ")"

    @abc.abstractmethod
    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : int or float or str or method
            The value to be randomised.

        Returns
        -------
        int or float or str or method
            The randomised value, same type as `value`.

        """

    def bias(self, value):
        """Returns the bias of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the bias of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The bias of the mechanism at `value` if defined, `None` otherwise.

        """
        pass

    def variance(self, value):
        """Returns the variance of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the variance of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The variance of the mechanism at `value` if defined, `None` otherwise.

        """
        pass

    def mse(self, value):
        """Returns the mean squared error (MSE) of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the MSE of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The MSE of the mechanism at `value` if defined, `None` otherwise.

        """
        pass

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    def _check_all(self, value):
        pass


class TruncationAndFoldingMixin:  # pylint: disable=too-few-public-methods
    """Mixin for truncating or folding the outputs of a mechanism.  Must be instantiated with a :class:`.DPMechanism`.

    Parameters
    ----------
    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    """
    def __init__(self, *, lower, upper):
        if not isinstance(self, DPMechanism):
            raise TypeError("TruncationAndFoldingMachine must be implemented alongside a :class:`.DPMechanism`")

        self.lower, self.upper = self._check_bounds(lower, upper)

    @classmethod
    def _check_bounds(cls, lower, upper):
        """Performs a check on the bounds provided for the mechanism."""
        pass

    def _check_all(self, value):
        """Checks that all parameters of the mechanism have been initialised correctly"""
        pass

    def _truncate(self, value):
        pass

    def _fold(self, value):
        pass


def bernoulli_neg_exp(gamma, random_state=None):
    """Sample from Bernoulli(exp(-gamma)).

    Adapted from "The Discrete Gaussian for Differential Privacy", Canonne, Kamath, Steinke, 2020.
    https://arxiv.org/pdf/2004.00010v2.pdf

    Parameters
    ----------
    gamma : float
        Parameter to sample from Bernoulli(exp(-gamma)).  Must be non-negative.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    Returns
    -------
    One sample from the Bernoulli(exp(-gamma)) distribution.

    """
    pass
