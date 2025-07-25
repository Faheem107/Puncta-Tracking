"""
Matching between trackmate and regionprops output tables
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def match_spots_regionprops_dist(
    spots: pd.DataFrame, props: pd.DataFrame, dims: int = 2, keep_duplicates: bool = False
) -> pd.DataFrame:
    """
    Matches spots detected in trackmate with regions in props. A spot is matched to the region
    with the closest centroid to the spot's location, along with the same frame index. A spot
    center cannot lie outside the bounding box of the region.

    Assumes the following columns in spots and props:
        * spots -> FRAME, POSITION_X, POSITION_Y, [POSITION_Z]
        * props -> frame, centroid-(0,1,[2]), bbox-(0,1,2,3,[4,5])
    Trackmate only supports 2D and 3D spots. dims sets the dimensions to use -- 2 for 2D spots
    and 3 for 3D spots.

    If keep_duplicates is True, then the duplicate columns from spots and props are kept -- that 
    is, both FRAME from spots and frame from props will be in the returned dataframe (and so on for
    other columns). Defaults to False.

    Returns a pandas dataframe valid as a spots table.
    """
    # Numpy axes order (row major)
    match dims:
        case 2:  # 2D
            spot_coord_names = ["POSITION_Y", "POSITION_X"]
            region_centroid_names = ["centroid-0", "centroid-1"]
            region_bbox_min_names = ["bbox-0", "bbox-1"]
            region_bbox_max_names = ["bbox-2", "bbox-3"]
        case 3:  # 3D
            spot_coord_names = ["POSITION_Z", "POSITION_Y", "POSITION_X"]
            region_centroid_names = ["centroid-0", "centroid-1", "centroid-2"]
            region_bbox_min_names = ["bbox-0", "bbox-1", "bbox-2"]
            region_bbox_max_names = ["bbox-3", "bbox-4", "bbox-5"]
        case _:
            raise ValueError(f"dims should be either 2 or 3, not {dims}")

    # Copy-on-Write enabled for pandas
    with pd.option_context("mode.copy_on_write", True):
        # Store index of closest region to a spot in spots
        spots["join_key_closest_region_index"] = np.nan

        # Loop over shared frames -- better use of memory for slower timing
        frames = np.intersect1d(
            np.unique(spots["FRAME"]), np.unique(props["frame"]), assume_unique=True
        )
        for frame in frames:
            # Distance between regionprops and trackmate centers
            spots_centers = np.array(
                spots.loc[spots["FRAME"] == frame, spot_coord_names], dtype=np.float64
            )
            regions_centers = np.array(
                props.loc[props["frame"] == frame, region_centroid_names],
                dtype=np.float64,
            )
            spots_regions_distances = cdist(spots_centers, regions_centers, "euclidean")
            # Check if spots are within bounding box of regions
            spots_in_region_bbox = np.full(spots_regions_distances.shape, True)
            regions_min_bbox = np.array(
                props.loc[props["frame"] == frame, region_bbox_min_names],
                dtype=np.float64,
            )
            regions_max_bbox = np.array(
                props.loc[props["frame"] == frame, region_bbox_max_names],
                dtype=np.float64,
            )
            for ind_dim in range(dims):
                # Converting 1D arrays to 2D with singleton axes for broadcasting purposes
                # Allows comparing each min or max with each spot coordinates
                check_min = (
                    regions_min_bbox[:, ind_dim][None, :]
                    <= spots_centers[:, ind_dim][:, None]
                )
                check_max = (
                    regions_max_bbox[:, ind_dim][None, :]
                    >= spots_centers[:, ind_dim][:, None]
                )
                check_all = np.logical_and(check_min, check_max)
                spots_in_region_bbox = np.logical_and(spots_in_region_bbox, check_all)
            spots_regions_distances[~spots_in_region_bbox] = np.inf
            # Assign spots to regions with closest distances
            # Spots with no regions close to them should be assigned to nan
            closest_region_index = [
                (
                    props.loc[props["frame"] == frame].index[np.argmin(srd)]
                    if not np.min(srd) == np.inf
                    else np.nan
                )
                for srd in spots_regions_distances
            ]
            spots.loc[spots["FRAME"] == frame, "join_key_closest_region_index"] = (
                closest_region_index
            )

        # Separate spots with no matches in regionprops
        if not np.all(np.isfinite(spots["join_key_closest_region_index"])):
            spots_matching = spots.groupby(
                np.isfinite(spots["join_key_closest_region_index"]),
                sort=False,
                dropna=False,
                as_index=False,
            )
            spots_matched, spots_not_matched = [
                spots_matching.get_group(val) for val in [True, False]
            ]
        else:
            spots_matched = spots
            spots_not_matched = pd.DataFrame({})

        # Join spots with props, retaining all spots but removing any props that
        # do not match with any spots
        try:
            new_spots = spots_matched.join(
                props, on="join_key_closest_region_index", how="left", validate="1:1"
            )
        except pd.errors.MergeError as merge_error:
            raise ValueError(
                "Too many spots match a region, check if your regions/spots overlap?"
            ) from merge_error

        # Add in unmatched spots. All regionprops values should be filled by NaN
        new_spots = pd.concat([new_spots, spots_not_matched], copy=False, axis=0)

        # Remove duplicate columns -- take the value from spots column
        if not keep_duplicates:
            new_spots = new_spots.drop(
                columns=["frame", *region_centroid_names, "join_key_closest_region_index"]
            )

        return new_spots


def match_spots_regionprops_label(
    spots: pd.DataFrame, props: pd.DataFrame, spots_label_col: str
) -> pd.DataFrame:
    """
    Matches spots detected in trackmate with regions in props. A spot is matched to region based
    on the region's label and corresponding intensity in the label channel for trackmate.

    For this function, add an additional 'channel' in the movie to be tracked by trackmate. This
    additional channel (called here the label channel) should be generated by the label function
    from skimage. The function then matches the corresponding spots with their regions by
    comparing the regions' label for each frame to the 'intensity' in this label channel.

    Assumes the following columns in spots and props:
        * spots -> FRAME, MEAN_INTENSITY_CH<n>
        * props -> frame, label
    where n is the number of the label channel in the movie sent to trackmate. n is set using the
    spots_label_channel argument of this function, and defaults to 1 (trackmate channel numbering
    is 1-indexed). Ensure that mean intensity is always calculated in the label channel.

    Returns a pandas dataframe
    """

    def pairing_cantor(a, b):
        """Cantor pairing function for hashing two integers together"""
        return 0.5 * (a + b) * (a + b + 1) + b

    # Copy-on-Write enabled for pandas
    with pd.option_context("mode.copy_on_write", True):
        # Generate comparing index using label and frame
        spots["hash_index"] = pairing_cantor(
            spots["FRAME"], spots[spots_label_col]
        )
        props["hash_index"] = pairing_cantor(props["frame"], props["label"])

        # Merge based on hash_index
        new_spots = spots.merge(props, how="left", on="hash_index", validate="1:1")
        # Remove duplicate columns -- take the value from spots column
        new_spots = new_spots.drop(
            columns=[
                "frame",
                spots_label_col,
                "hash_index",
            ]
        )

        return new_spots
