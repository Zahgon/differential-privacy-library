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
Implementation of the standard exponential mechanism, and its derivative, the hierarchical mechanism.
"""
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism, bernoulli_neg_exp
from diffprivlib.mechanisms.binary import Binary
from diffprivlib.utils import copy_docstring


class Exponential(DPMechanism):
    r"""
    The exponential mechanism for achieving differential privacy on candidate selection, as first proposed by McSherry
    and Talwar.

    The exponential mechanism achieves differential privacy by randomly choosing a candidate subject to candidate
    utility scores, with greater probability given to higher-utility candidates.

    Paper link: https://www.cs.drexel.edu/~greenie/privacy/mdviadp.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float
        The sensitivity in utility values to a change in a datapoint in the underlying dataset.

    utility : list
        A list of non-negative utility values for each candidate.

    monotonic : bool, default: False
        Specifies if the utility function is monotonic, i.e. that adding an individual to the underlying dataset can
        only increase the values in `utility`.

    candidates : list, optional
        An optional list of candidate labels.  If omitted, the zero-indexed list [0, 1, ..., n] is used.

    measure : list, optional
        An optional list of measures for each candidate.  If omitted, a uniform measure is used.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, sensitivity, utility, monotonic=False, candidates=None, measure=None,
                 random_state=None):
        super().__init__(epsilon=epsilon, delta=0.0, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self.utility, self.candidates, self.measure = self._check_utility_candidates_measure(utility, candidates,
                                                                                             measure)
        self.monotonic = bool(monotonic)
        self._probabilities = self._find_probabilities(self.epsilon, self.sensitivity, self.utility, self.monotonic,
                                                       self.measure)

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        pass

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        pass

    @classmethod
    def _check_utility_candidates_measure(cls, utility, candidates, measure):
        pass

    @classmethod
    def _find_probabilities(cls, epsilon, sensitivity, utility, monotonic, measure):
        pass

    def _check_all(self, value):
        pass

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        pass

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        pass

    def randomise(self, value=None):
        """Select a candidate with differential privacy.

        Parameters
        ----------
        value : None
            Ignored.

        Returns
        -------
        int or other
            The randomised candidate.

        """
        pass


class PermuteAndFlip(Exponential):
    r"""
    The permute and flip mechanism for achieving differential privacy on candidate selection, as first proposed by
    McKenna and Sheldon.

    The permute and flip mechanism is an alternative to the exponential mechanism, and achieves differential privacy by
    randomly choosing a candidate subject to candidate utility scores, with greater probability given to higher-utility
    candidates.

    Paper link: https://arxiv.org/pdf/2010.12603.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float
        The sensitivity in utility values to a change in a datapoint in the underlying dataset.

    utility : list
        A list of non-negative utility values for each candidate.

    monotonic : bool, default: False
        Specifies if the utility function is monotonic, i.e. that adding an individual to the underlying dataset can
        only increase the values in `utility`.

    candidates : list, optional
        An optional list of candidate labels.  If omitted, the zero-indexed list [0, 1, ..., n] is used.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, sensitivity, utility, monotonic=False, candidates=None, random_state=None):
        super().__init__(epsilon=epsilon, sensitivity=sensitivity, utility=utility, monotonic=monotonic,
                         candidates=candidates, measure=None, random_state=random_state)

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        pass

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        pass

    @classmethod
    def _find_probabilities(cls, epsilon, sensitivity, utility, monotonic, measure):
        pass

    def randomise(self, value=None):
        """Select a candidate with differential privacy.

        Parameters
        ----------
        value : None
            Ignored.

        Returns
        -------
        int or other
            The randomised candidate.

        """
        pass


class ExponentialCategorical(DPMechanism):
    r"""
    The exponential mechanism for achieving differential privacy on categorical inputs, as first proposed by McSherry
    and Talwar.

    The exponential mechanism achieves differential privacy by randomly choosing an output value for a given input
    value, with greater probability given to values 'closer' to the input, as measured by a given utility function.

    Paper link: https://www.cs.drexel.edu/~greenie/privacy/mdviadp.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    utility_list : list of tuples
        The utility list of the mechanism.  Must be specified as a list of tuples, of the form ("value1", "value2",
        utility), where each `value` is a string and `utility` is a strictly positive float.  A `utility` must be
        specified for every pair of values given in the `utility_list`.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, utility_list, random_state=None):
        super().__init__(epsilon=epsilon, delta=0.0, random_state=random_state)

        self._balanced_tree = False
        self._utility_values, self._sensitivity, self._domain_values = self._build_utility(utility_list)
        self._check_utility_full(self._domain_values)
        self._normalising_constant = self._build_normalising_constant()

    def _build_utility(self, utility_list):
        pass

    def _check_utility_full(self, domain_values):
        pass

    @property
    def utility_list(self):
        """Gets the utility list of the mechanism, in the same form as accepted by `.set_utility_list`.

        Returns
        -------
        utility_list : list of tuples (str, str, float), or None
            Returns a list of tuples of the form ("value1", "value2", utility), or `None` if the utility has not yet
            been set.

        """
        pass

    def _build_normalising_constant(self, re_eval=False):
        pass

    def _get_utility(self, value1, value2):
        pass

    def _get_prob(self, value1, value2):
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

    @copy_docstring(Binary.randomise)
    def randomise(self, value):
        pass


class ExponentialHierarchical(ExponentialCategorical):
    r"""
    Adaptation of the exponential mechanism to hierarchical data.  Simplifies the process of specifying utility values,
    as the values can be inferred from the hierarchy.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    hierarchy : nested list of str
        The hierarchy as specified as a nested list of string.  Each string must be a leaf node, and each leaf node
        must lie at the same depth in the hierarchy.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    Examples
    --------
    Example hierarchies:

    >>> flat_hierarchy = ["A", "B", "C", "D", "E"]
    >>> nested_hierarchy = [["A"], ["B"], ["C"], ["D", "E"]]

    """
    def __init__(self, *, epsilon, hierarchy, random_state=None):
        self.hierarchy = hierarchy
        utility_list = self._build_utility_list(self._build_hierarchy(hierarchy))
        super().__init__(epsilon=epsilon, utility_list=utility_list, random_state=random_state)
        self._list_hierarchy = None

    def _build_hierarchy(self, nested_list, parent_node=None):
        pass

    @staticmethod
    def _check_hierarchy_height(hierarchy):
        pass

    @staticmethod
    def _build_utility_list(hierarchy):
        pass

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        pass

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        pass
