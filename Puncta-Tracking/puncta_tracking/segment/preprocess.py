"""
Functions for preprocessing movies before calling ilastik
"""

# import numpy as np
from numpy.typing import NDArray


def bkg_subtract_images(movie: NDArray) -> NDArray:
    """
    Currently not implemented.

    Performs background subtraction for given movie. Movie should have
    structure [...]YX -- that is, the last two axes should be Y and X
    respectively. Background subtraction is done independently on each
    'frame'.
    """
    return movie


def bleach_correction(movie: NDArray, method: str = "ratio") -> NDArray:
    """
    Currently not implemented.

    Performs bleach correction for a given movie. Movie is assumed to have
    the axes ordering as T[...]YX -- that is, the first axis should be time,
    and the last two axes should be Y and X respectively.

    Follows ImageJ Bleach correction methods:
        'ratio' : Simple Ratio
        'exp'   : Exponential fitting (with offset)
        'hist'  : Histogram matching
    """
    return movie
