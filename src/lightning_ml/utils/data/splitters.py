"""Data split utilities that extend scikit-learn splitters with a **validation** subset.

This module defines :class:`TrainTestValSplitter`, a lightweight wrapper around any
scikit-learn :class:`sklearn.model_selection.BaseCrossValidator` that leaves the
training indices untouched but subdivides the held-out indices into *validation*
and *test* parts.
"""

from sklearn.model_selection._split import *
