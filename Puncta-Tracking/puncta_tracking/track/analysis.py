"""
Analysis functions to use with Network digraph objects, which represent tracks.

Currently only has merges/splits and MSD calculator defined.
"""

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from networkx import DiGraph


def get_merges_splits(
    tracks: Iterable[DiGraph], attribute: str, selector: str, accumulate_fun=None
) -> tuple[NDArray, NDArray]:
    """
    Gets properties for merges/splilts for given tracks. selector selects the mode.

    If selector is "Merge", merge events are investigated.
    If selector is "Split", split events are investigated.

    accumulate_fun is a function that takes as input an iterable and returns a single value.
    It determines how the properties of all predecessors (in a merge event) or successors
    (in a split event) are combined. By default, accumulate_fun is None.

    Returns 2 arrays, events_num and events_att, of size Nx3, where N is the number of
    events.
    For events_num, the first column is the number of predecessors, the second column is
    always 1 (only one spot at any event), and the last column is the number of successor
    of the given event.
    For events_att, the first column is the 'accumulated' value of given attribute for the
    predecessors, the second column is the value of the given attribute for the spot where
    the event occured, and the last column is the 'accumulated' value of given attribute for
    the successors.
    """
    if selector.lower() not in ["merge", "split"]:
        raise ValueError(
            f"Unknown selector value {selector}. Choose either Merge or Split."
        )
    merge_selector = selector.lower() == "merge"
    if accumulate_fun is None:
        accumulate_fun = sum
    events_num = []
    events_att = []
    for track in tracks:
        iter_nodes = track.in_degree() if merge_selector else track.out_degree()
        for key, val in iter_nodes:
            if val > 1:
                pred_num = sum(1 for _ in track.predecessors(key))
                pred_att = accumulate_fun(
                    [track.nodes[pred][attribute] for pred in track.predecessors(key)]
                )
                succ_num = sum(1 for _ in track.successors(key))
                succ_att = accumulate_fun(
                    [track.nodes[succ][attribute] for succ in track.successors(key)]
                )
                events_num.append([pred_num, 1, succ_num])
                events_att.append([pred_att, track.nodes[key][attribute], succ_att])
    events_num = np.array(events_num)
    events_att = np.array(events_att)
    return events_att, events_num


def get_msd_vs_time(tracks: Iterable[DiGraph], t_eval: NDArray):
    """
    Calculates MSD vs time for a given set of tracks. t_eval is assumed to be a vector and in 
    increasing order
    """
    # Ignore all tracks with merges or splits
    tracks = [
        track
        for track in tracks
        if (track.graph["NUMBER_MERGES"] == 0 and track.graph["NUMBER_SPLITS"] == 0)
    ]
    # For each track, calculate square displacement versus time
    coordinate_features = ["POSITION_X", "POSITION_Y", "POSITION_Z"]
    interp_sq_displacement = np.empty((t_eval.shape[0], len(tracks)))
    for ind_track, track in enumerate(tracks):
        # Order nodes by time
        nodes_time_sorted = sorted(
            track.nodes, key=lambda node: track.nodes[node]["POSITION_T"]
        )
        # Get square displacement and time
        time = []
        square_displacement = []
        for node in nodes_time_sorted:
            time.append(track.nodes[node]["POSITION_T"])
            r = [
                track.nodes[node][coordinate_feature]
                - track.nodes[nodes_time_sorted[0]][coordinate_feature]
                for coordinate_feature in coordinate_features
            ]
            square_displacement.append((np.linalg.norm(r)) ** 2)
        # Interpolate and calculate square displacement on t_eval
        interp_sq_displacement[:, ind_track] = np.interp(
            t_eval, time, square_displacement, left=np.nan, right=np.nan
        )
    # Return mean of square displacement
    return np.nanmean(interp_sq_displacement, axis=1)
