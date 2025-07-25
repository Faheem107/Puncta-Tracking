"""
Functions for working with ilastik:
    1. Calling ilastik with (preprocessed) images
    2. Reading segmentations and preparing masks
"""

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray, DTypeLike

from ..config import __external_paths__
from ..io import read_movie, write_imagej_movie, _call_nt_vs_posix

@_call_nt_vs_posix
def _call_ilastik(project_path: Path, movie_path: Path, segment_path: Path):
    """
    Calls ilastik on given movie. Project file is assumed to be for pixel classification.
    Stores segmentation result as a multipage tiff in segment_path
    """
    args = [
        f"'{__external_paths__['ilastik-exe']}'",
        "--headless",
        f"--project='{project_path}'",
        "--output_format='multipage tiff'",
        f"--output_filename_format='{segment_path}'",
        "--export_source='Simple Segmentation'",
        f"'{movie_path}'",
    ]
    return args


def ilastik_segmenter(
    movie: NDArray,
    movie_axes_order: str,
    segment_label: int,
    results_dir: Path | str,
    project_file: Optional[Path | str] = None,
    movie_dtype: DTypeLike = np.dtype("uint16"),
) -> Optional[tuple[NDArray,str]]:
    """
    Uses ilastik project file to segment the given movie.

    Results are stored in given results directory under the names:
        'ilastik_input.tiff'        : input movie
        'ilastik_segmentation.tiff' : output segmentations
        'ilastik_masks.tiff'        : output masks

    The input is converted to given dtype specified by movie_dtype
    (defaults to uint16).

    Only axes order that are a subset of TZCYX are accepted.

    If project file is None, only the input movie tiff is generated.
    This may be useful to generate the preprocessed movie to be
    used for training. None is returned in this case. Unless a 
    project_file is specified, default is to assume no project file.

    If project file is not None, returns the masks as a numpy array as 
    the first argument, and the axes order of the masks as the second
    argument.
    """
    # Save file names
    input_name = "ilastik_input.tiff"
    seg_name = "ilastik_segmentation.tiff"
    mask_name = "ilastik_mask.tiff"
    ilastik_out = "ilastik_stdout.log"
    ilastik_err = "ilastik_stderr.log"
    # Save input movie with given data type
    results_dir = Path(results_dir).resolve()
    write_imagej_movie(
        results_dir / input_name, movie.astype(movie_dtype), axes_order=movie_axes_order
    )
    # Call ilastik
    if project_file is None:
        print(
            f"No project file given. Generated input movie at {results_dir/input_name} ."
        )
        return None
    res = _call_ilastik(
        Path(project_file), results_dir / input_name, results_dir / seg_name
    )
    # Save logs -- pylint complains here since the decorator for _call_ilastik takes
    # care of calling subprocess and getting the output result, not _call_ilastik itself
    # Hence to pylint it looks like _call_ilastik is outputting a list.
    with open(results_dir / ilastik_out, "w", encoding="utf-8") as f:
        f.write(res.stdout)  # pylint: disable=no-member
    with open(results_dir / ilastik_err, "w", encoding="utf-8") as f:
        f.write(res.stderr)  # pylint: disable=no-member
    # Generate masks - they are generated as a binary image, with 1 where the given label
    # is and 0 everywhere else
    mask, _ = read_movie(results_dir / seg_name)
    mask = (mask == segment_label).astype(np.float64)
    # Store and return masks
    # For mask, if input movie is 3D, the axes order is TZYX. If it is 2D, axes order is TYX.
    mask_axes_order = "TZYX" if "Z" in movie_axes_order else "TYX"
    write_imagej_movie(
        results_dir / mask_name, mask.astype(movie_dtype), mask_axes_order
    )
    return mask, mask_axes_order
