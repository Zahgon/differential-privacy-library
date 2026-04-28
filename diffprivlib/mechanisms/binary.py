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
The binary mechanism for differential privacy.

"""
import numpy as np

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.utils import copy_docstring


class Binary(DPMechanism):
    r"""The classic binary mechanism in differential privacy.

    Given a binary input value, the mechanism randomly decides to flip to the other binary value or not, in order to
    satisfy differential privacy.

    Paper link: https://arxiv.org/pdf/1612.05568.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    value0 : str
        0th binary label.

    value1 : str
        1st binary label.  Cannot be the same as ``value0``.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    Notes
    -----
    * The binary attributes, known as `labels`, must be specified as strings.  If non-string labels are required (e.g.
      integer-valued labels), a :class:`.DPTransformer` can be used (e.g. :class:`.IntToString`).

    """
    def __init__(self, *, epsilon, value0, value1, random_state=None):
        super().__init__(epsilon=epsilon, delta=0.0, random_state=random_state)
        self.value0, self.value1 = self._check_labels(value0, value1)

    @classmethod
    def _check_labels(cls, value0, value1):
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
        value : str
            The value to be randomised.

        Returns
        -------
        str
            The randomised value.

        """
        pass
