"""
Functions for reading and writing movies
"""

from pathlib import Path
from typing import Callable
import os
import subprocess
import functools

import nd2
import tifffile as tff
import numpy as np
from numpy.typing import NDArray


def imread_nd2(movie_path: Path) -> tuple[NDArray, str]:
    """
    Reads nd2 movie, returns as numpy array along with
    axes order.

    Uses nd2 package
    """
    with nd2.ND2File(movie_path) as myfile:
        movie = myfile.asarray()
        axes = "".join(myfile.sizes.keys())
    return movie, axes


def imread_tff(movie_path: Path) -> tuple[NDArray, str]:
    """
    Reads tiff movie, returns as numpy array along with
    axes order.

    Uses tifffile package
    Multi-series and multi-level images are not supported.
    For a mulit-series tiff file, the first series and level
    are returned
    """
    with tff.TiffFile(movie_path) as myfile:
        movie = myfile.asarray()
        axes = myfile.series[0].get_axes()
    return movie, axes


def read_movie(movie_path: Path | str) -> tuple[NDArray, str]:
    """
    Reads nd2 or tif/tiff movie as a numpy array
    """
    movie_path = Path(movie_path).resolve()
    match movie_path.suffix:
        case ".nd2":
            movie, axes = imread_nd2(movie_path)
        case ".tif" | ".tiff":
            movie, axes = imread_tff(movie_path)
        case _:
            raise ValueError(f"Unknown movie type {movie_path.suffix}")
    return movie, axes


def write_imagej_movie(movie_path: Path | str, movie: NDArray, axes_order: str) -> None:
    """
    Writes imagej hyperstack to given location. Only usable for
    axes orders which are a subset of TZCYX.
    """
    movie_path = Path(movie_path).resolve()
    # To write an imagej file, the axes orders must follow the TZCYX order.
    # Arrange axes of movie in that order, based on given axes_order
    new_axes_order, old_axes_indices = rearrange_axes(axes_order, "TZCYX")
    movie = np.moveaxis(movie, old_axes_indices, range(len(axes_order)))
    # Write imagej hyperstack using tifffile
    tff.imwrite(movie_path, movie, imagej=True, metadata={"axes": new_axes_order})


def rearrange_axes(
    input_axes_order: str, master_axes_order: str
) -> tuple[str, list[int]]:
    """
    Rearranges given input axes order, assumed to be a subset of master axes order,
    such that the axes follow the same order as that given in the master axes order.
    For example, if input axes order is CTYX and master axes order is TZCYX, then the
    axes order is rearranged to TCYX.

    Raises a ValueError if the input_axes_order is not a subset of master_axes_order.

    Returns the rearranged axes order, and the list of indices which rearrange
    input_axes_order into the rearranged axes order. For the above example, the full
    output is then 'TCYX',[1,0,2,3] . Note that 'CTYX'[1] = 'T', 'CTYX'[0] = 'C',
    'CTYX'[2] = 'Y' and 'CTYX'[3] = 'X'.
    """
    # Check subset
    input_set = set(input_axes_order)
    if input_set.intersection(master_axes_order) != input_set:
        raise ValueError(
            f"Input axes order {input_axes_order} is not a subset of {master_axes_order}"
        )
    # Sort input based on master
    input_dict = {ax: ind for ind, ax in enumerate(input_axes_order)}
    output_axes_order = "".join(sorted(input_dict, key=master_axes_order.index))
    sort_indices = [input_dict[ax] for ax in output_axes_order]
    return output_axes_order, sort_indices


def _call_nt_vs_posix(func: Callable) -> Callable:
    """
    Decorator to change quotes between windows and posix shells. Input function
    must return a list of arguments to be passed to subprocess written in the
    posix style -- that is, by using single quotes to enclose strings and double
    quotes only within single quoted strings. The returned function joins the
    argument list, checks if os is 'nt' or 'posix', swaps ' and " if os.name is
    'nt' and then calls subprocess.run with given argument result. Captured text
    output from subprocess.run call is returned by the output function.
    """

    def swap_chars(str_in, char1, char2):
        str_out = ""
        for char in str_in:
            if char == char1:
                str_out += char2
            elif char == char2:
                str_out += char1
            else:
                str_out += char
        return str_out

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process_args = func(*args, **kwargs)
        process_args = " ".join(process_args)
        if os.name == "nt":
            process_args = swap_chars(process_args, "'", '"')
        result = subprocess.run(
            process_args, check=True, capture_output=True, text=True, shell=True
        )
        return result

    return wrapper
