"""
Functions for working with trackmate:
    1. Multiply masks with input movies
    2. Call trackmate for tracking on these movies
    3. Interpret trackmate xml file
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray, DTypeLike
import pandas as pd

from ..config import __external_paths__
from ..io import _call_nt_vs_posix, write_imagej_movie
from .xml import parse_trackmate_xml, extract_spots, extract_edges, extract_tracks

@_call_nt_vs_posix
def _call_trackmate(movie_path: Path, trackmate_xml_path: Path):
    """
    Function to call trackmate on given movie. Stores the resulting
    xml file in trackmate_xml_path.
    """
    # Common arguments
    args = [
        f"'{__external_paths__['fiji-exe']}'",
        "--ij2",
        "--headless",
        "--console",
        "--run",
        f"'{__external_paths__['trackmate-jython']}'",
    ]
    # Script arguments
    script_params = ["inFile", "xmlFile"]
    script_paths = [movie_path, trackmate_xml_path]
    script_args_list = [
        f'{param}="{path}"' for param, path in zip(script_params, script_paths)
    ]
    script_args = ",".join(script_args_list)
    script_args = f"'{script_args}'"
    # Run FIJI
    args.append(script_args)
    return args


def trackmate_tracker(
    movie: NDArray,
    movie_axes_order: str,
    results_dir: Path | str,
    movie_dtype: DTypeLike = np.dtype("uint16"),
    remove_undetected_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calls trackmate to track given movie.

    Results are stored in given results directory under the names:
        'trackmate_input.tiff'      : input movie
        'trackmate_output.xml'      : output tracking result (xml)

    The input is converted to given dtype specified by movie_dtype
    (defaults to uint16).

    If remove_undetected_features is True, any (spot,edge,track)
    feature not observed in atleast one (spot,edge,track) is removed
    from the resulting dataframes.

    Returns spots, edges, tracks as dataframes
    """
    movie_name = "trackmate_input.tiff"
    xml_name = "trackmate_output.xml"
    # Save input movie
    results_dir = Path(results_dir).resolve()
    write_imagej_movie(
        results_dir / movie_name, movie.astype(movie_dtype), movie_axes_order
    )
    # Call trackmate
    res = _call_trackmate(results_dir / movie_name, results_dir / xml_name)
    with open(results_dir / "trackmate_stdout.log", "w", encoding="utf-8") as f:
        f.write(res.stdout)  # pylint: disable=no-member
    with open(results_dir / "trackmate_stderr.log", "w", encoding="utf-8") as f:
        f.write(res.stderr)  # pylint: disable=no-member
    # Get spots, edges and tracks tables
    xml_tree = parse_trackmate_xml(results_dir / xml_name)
    spots = extract_spots(xml_tree, remove_undetected_features)
    edges = extract_edges(xml_tree, remove_undetected_features)
    tracks = extract_tracks(xml_tree, remove_undetected_features)
    return spots, edges, tracks
