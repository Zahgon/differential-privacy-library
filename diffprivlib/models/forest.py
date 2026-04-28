# MIT License
#
# Copyright (C) IBM Corporation 2021
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
Random Forest Classifier with Differential Privacy
"""
from collections import namedtuple
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.tree._tree import Tree, DOUBLE, DTYPE, NODE_DTYPE  # pylint: disable=no-name-in-module
from sklearn.ensemble._forest import RandomForestClassifier as skRandomForestClassifier, _parallel_build_trees
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier

# TODO: remove when sklearn 1.6 a min req
try:
    from sklearn.utils.validation import validate_data
except ImportError:
    from sklearn.base import BaseEstimator
    validate_data = BaseEstimator._validate_data

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning, check_random_state
from diffprivlib.mechanisms import PermuteAndFlip
from diffprivlib.validation import DiffprivlibMixin

MAX_INT = np.iinfo(np.int32).max


class RandomForestClassifier(skRandomForestClassifier, DiffprivlibMixin):  # pylint: disable=too-many-ancestors
    r"""Random Forest Classifier with differential privacy.

    This class implements Differentially Private Random Decision Forests using [1].
    :math:`\epsilon`-Differential privacy is achieved by constructing decision trees via random splitting criterion and
    applying the :class:`.PermuteAndFlip` Mechanism to determine a noisy label.

    Parameters
    ----------
    n_estimators : int, default: 10
        The number of trees in the forest.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds :  tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    classes : array-like of shape (n_classes,)
        Array of classes to be trained on.  If not provided, the classes will be read from the data when ``.fit()`` is
        first called, resulting in a :class:`.PrivacyLeakWarning`.

    n_jobs : int, default: 1
        Number of CPU cores used when parallelising over classes. ``-1`` means using all processors.

    verbose : int, default: 0
        Set to any positive number for verbosity.

    random_state : int or RandomState, optional
        Controls both the randomness of the shuffling of the samples used when building trees (if ``shuffle=True``) and
        training of the differentially-private :class:`.DecisionTreeClassifier` to construct the forest.  To obtain a
        deterministic behaviour during randomisation, ``random_state`` has to be fixed to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    max_depth : int, default: 5
        The maximum depth of the tree.  The depth translates to an exponential increase in memory usage.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble,
        otherwise, just fit a whole new forest.

    shuffle : bool, default=False
        When set to ``True``, shuffles the datapoints to be trained on trees at random.  In diffprivlib, each datapoint
        is used to train exactly one tree. When set to ``False``, datapoints are chosen in-order to their tree in
        sequence.

    Attributes
    ----------
    estimator_ : DecisionTreeClassifier
        The child estimator template used to create the collection of fitted sub-estimators.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X` has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from diffprivlib.models import RandomForestClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    References
    ----------
    [1] Sam Fletcher, Md Zahidul Islam. "Differentially Private Random Decision Forests using Smooth Sensitivity"
    https://arxiv.org/abs/1606.03572

    """

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        skRandomForestClassifier, "n_estimators", "n_jobs", "verbose", "random_state", "warm_start")

    def __init__(self, n_estimators=10, *, epsilon=1.0, bounds=None, classes=None, n_jobs=1, verbose=0, accountant=None,
                 random_state=None, max_depth=5, warm_start=False, shuffle=False, **unused_args):
        super().__init__(
            n_estimators=n_estimators,
            criterion=None,
            bootstrap=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)
        self.epsilon = epsilon
        self.bounds = bounds
        self.classes = classes
        self.max_depth = max_depth
        self.shuffle = shuffle
        self.accountant = BudgetAccountant.load_default(accountant)
        self.estimator = DecisionTreeClassifier()
        self.estimator_params = ("max_depth", "epsilon", "bounds", "classes")

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to ``dtype=np.float32``.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass


class DecisionTreeClassifier(skDecisionTreeClassifier, DiffprivlibMixin):
    r"""Decision Tree Classifier with differential privacy.

    This class implements the base differentially private decision tree classifier
    for the Random Forest classifier algorithm. Not meant to be used separately.

    Parameters
    ----------
    max_depth : int, default: 5
        The maximum depth of the tree.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    classes : array-like of shape (n_classes,), optional
        Array of class labels. If not provided, the classes will be read from the data when ``.fit()`` is first called,
        resulting in a :class:`.PrivacyLeakWarning`.

    random_state : int or RandomState, optional
        Controls the randomness of the estimator.  At each split, the feature to split on is chosen randomly, as is the
        threshold at which to split.  The classification label at each leaf is then randomised, subject to differential
        privacy constraints. To obtain a deterministic behaviour during randomisation, ``random_state`` has to be fixed
        to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    n_features_in_: int
        The number of features when fit is performed.

    n_classes_: int
        The number of classes.

    classes_: array of shape (n_classes, )
        The class labels.

    """

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        skDecisionTreeClassifier, "max_depth", "random_state")

    def __init__(self, max_depth=5, *, epsilon=1, bounds=None, classes=None, random_state=None, accountant=None,
                 criterion=None, **unused_args):
        super().__init__(
            criterion=None,
            splitter=None,
            max_depth=max_depth,
            min_samples_split=None,
            min_samples_leaf=None,
            min_weight_fraction_leaf=None,
            max_features=None,
            random_state=random_state,
            max_leaf_nodes=None,
            min_impurity_decrease=None
        )
        self.epsilon = epsilon
        self.bounds = bounds
        self.classes = classes
        self.accountant = BudgetAccountant.load_default(accountant)

        if criterion is not None:
            unused_args['criterion'] = criterion

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a differentially-private decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to ``dtype=np.float32``.

        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        check_input : bool, default=True
            Allow to bypass several input checking. Don't use this parameter unless you know what you do.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        pass

    def _fit(self, X, y, sample_weight=None, check_input=True, missing_values_in_feature_mask=None):
        pass

    @property
    def n_features_(self):
        pass

    def _more_tags(self):
        pass


class _FittingTree(DiffprivlibMixin):
    r"""Array-based representation of a binary decision tree, trained with differential privacy.

    This tree mimics the architecture of the corresponding Tree from sklearn.tree.tree_, but without many methods given
    in Tree. The purpose of _FittingTree is to fit the parameters of the model, and have those parameters passed to
    Tree (using _FittingTree.__getstate__() and Tree.__setstate__()), to be used for prediction.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.

    n_features : int
        The number of features of the training dataset.

    classes : array-like of shape (n_classes,)
        The classes of the training dataset.

    epsilon : float
        Privacy parameter :math:`\epsilon`.

    bounds : tuple
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data.

    random_state : RandomState
        Controls the randomness of the building and training process: the feature to split at each node, the threshold
        to split at and the randomisation of the label at each leaf.

    """
    _TREE_LEAF = -1
    _TREE_UNDEFINED = -2
    StackNode = namedtuple("StackNode", ["parent", "is_left", "depth", "bounds"])

    def __init__(self, max_depth, n_features, classes, epsilon, bounds, random_state):
        self.node_count = 0
        self.nodes = []
        self.max_depth = max_depth
        self.n_features = n_features
        self.classes = classes
        self.epsilon = epsilon
        self.bounds = bounds
        self.random_state = random_state

    def __getstate__(self):
        """Get state of _FittingTree to feed into __setstate__ of sklearn.Tree"""
        d = {"max_depth": self.max_depth,
             "node_count": self.node_count,
             "nodes": np.array([tuple(node) for node in self.nodes], dtype=NODE_DTYPE),
             "values": self.values_}
        return d

    def build(self):
        """Build the decision tree using random feature selection and random thresholding."""
        pass

    def fit(self, X, y):
        """Fit the tree to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        """
        pass

    def apply(self, X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        pass


class _Node:
    """Base storage structure for the nodes in a _FittingTree object."""
    def __init__(self, node_id, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.left_child = -1
        self.right_child = -1
        self.node_id = node_id

    def __iter__(self):
        """Defines parameters needed to populate NODE_DTYPE for Tree.__setstate__ using tuple(_Node)."""
        yield self.left_child
        yield self.right_child
        yield self.feature
        yield self.threshold
        yield 0.0  # Impurity
        yield 0  # n_node_samples
        yield 0.0  # weighted_n_node_samples

        # remove branch when scikit-learn v1.3 is min requirement
        if len(NODE_DTYPE) > 7:
            yield False
