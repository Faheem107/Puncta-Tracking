"""
Functions to extract region properties:
    1. Use regionprops from skimage to extract properties
    2. Update spots table from trackmate using these regionprops
"""

from typing import Optional, Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from skimage.measure import label  # pylint: disable = no-name-in-module
from skimage.measure import regionprops_table  # pylint: disable = no-name-in-module
import pandas as pd

from ..io import rearrange_axes


def create_masked_movie(movie: NDArray, mask: NDArray, dims: int = 2) -> NDArray:
    """
    Multiplies given mask with given movie. mask is assumed to have axis
    order as TYX if dims=2 and TZYX if dims=3. movie is assumed to have
    an axis order as T[...]YX if dims=2 and T[...]ZYX if dims=3.
    mask is multiplied to all axes within [...] in movie using numpy
    broadcasting rules. For example:
        If movie has axis order TCYX, then the multiplication occurs as
        movie*mask[:,None,...] -- thus the mask is applied to all channels.
    The masked movie is returned
    """
    if dims not in [2, 3]:
        raise ValueError(f"Unknown number of dimensions {dims}")
    spatial_axes = dims
    time_axis = 1
    one_dim_axes = len(movie.shape) - spatial_axes - time_axis
    return movie * mask[(slice(None), *[None for _ in range(one_dim_axes)], Ellipsis)]


def get_regionprops(
    movie: NDArray,
    movie_axes_order: str,
    masks: NDArray,
    masks_axes_order: str,
    properties: tuple[str, ...] | list[str] = ("label", "centroid", "bbox"),
    extra_properties: Optional[Iterable[Callable]] = None,
    spacing: Optional[dict[str, float]] = None,
    connectivity: Optional[int] = 1,
) -> pd.DataFrame:
    """
    Calculates given properties for each connected component in masks and for
    each frame in masks and movie. Both movie and mask axes orders are assumed to
    be a subset of TZCYX. Furthermore, mask and movie axes order may differ only in
    whether C is present in movie or not. For example, TCYX and TYX are valid axes
    orders for movie and masks, but TYX is not a valid mask axes order for a movie
    with axes order TZCYX. Connectivity is set to the highest spatial order (that is,
    connectivity is 3 for ZYX and 2 for YX).

    properties and extra_properties are passed directly to regionprops_table. Label
    should always be added to properties to ensure the spots can be tracked. centroid
    and bbox are needed for matching with spots from trackmate. By default, properties
    only has label, centroid and bbox. 

    spacing -- a dictionary that specifies the spacing along each of ZYX axes. None by
    default, implying a value of {'Z': 1, 'Y': 1, 'X': 1}.

    Returns a pandas dataframe with each 'spot' with a frame index attached.
    """
    # Check if C is the only extra axes in movie
    check_one_channel_axis = set(movie_axes_order).difference(masks_axes_order)
    if check_one_channel_axis != {"C"} and check_one_channel_axis != set():
        raise ValueError("Cannot have more than one channel axes in movie.")
    # Rearrange movie and masks to have the order required by regionprops. regionprops
    # requires that the Channel axis, if present, is the last axis. For proper looping
    # over time, we also require that the Time axis, if present, is the first axis.
    master_axes_order = "TZYXC"  # pushes C to the end if present, and T to the front.
    new_movie_axes_order, sort_indices = rearrange_axes(
        movie_axes_order, master_axes_order
    )
    movie = np.moveaxis(movie, sort_indices, range(len(movie_axes_order)))
    _, sort_indices = rearrange_axes(masks_axes_order, master_axes_order)
    masks = np.moveaxis(masks, sort_indices, range(len(masks_axes_order)))
    # Reorder spacing accordingly
    if spacing is not None:
        new_spacing = [
            spacing[ax] for ax in new_movie_axes_order if ax in spacing.keys()
        ]
    else:
        new_spacing = None
    # regionprops_table and label do not distinguish between time and spatial axes. Thus,
    # we cannot give a T[Z]YX array as mask and T[Z]YX[C] array as movie directly -- label
    # will assume a connectivity of 3 for TYX and 4 for TZYX. Which means, regionprops_table
    # will generate regions that connect pixels in different frames to make its connected
    # components. To avoid this, we need to call regionprops_table for each frame separately.
    # Check if T is present -- otherwise the usual regionprops suffices.
    if "T" in movie_axes_order:
        all_props = []
        for ind_frame, (mask, frame) in enumerate(zip(masks, movie)):
            label_mask = label(mask, connectivity=connectivity)
            props = regionprops_table(
                label_mask,
                frame,
                properties=properties,
                extra_properties=extra_properties,
                spacing=new_spacing,
            )
            # Attach frame number to props
            props["frame"] = ind_frame
            all_props.append(pd.DataFrame(props))
        # Concatenate all dataframes. Each 'spot' is then recognized by its label and frame
        return pd.concat(all_props, ignore_index=True)
    # No 'T' -- just send to regionprops
    props = regionprops_table(
        label(masks),
        movie,
        properties=properties,
        extra_properties=extra_properties,
        spacing=new_spacing,
    )
    return pd.DataFrame(props)
