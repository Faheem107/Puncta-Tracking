"""
Functions to convert potentially merging/splitting tracks
into single tracks for phase portrait calculation.
"""

from typing import Callable, Optional, Any
from dataclasses import dataclass
from collections.abc import Iterator

import networkx as nx
from networkx import weakly_connected  # pylint:disable=import-error, no-name-in-module
import pandas as pd


def is_valid_track(track: nx.DiGraph) -> bool:
    """
    Checks if given input is a track.

    A graph is a track if it is has
    * directed edges: Each directed edge represents a change that happened in a time step.
    * no directed cycles: As time shouldn't backwards, track cannot have a directed cycles.
    * weakly connected: A single track consists of the same set of particles tracked over
    time, and thus will be a weakly connected graph. Disconnected directed graphs may
    indicate a collection of tracks.
    """
    check_dag = nx.is_directed_acyclic_graph(track)
    check_conn = weakly_connected.is_weakly_connected(track)
    return check_dag & check_conn


def is_simple_track(track: nx.DiGraph) -> bool:
    """
    Checks if a valid track is simple.

    A track is simple if has no merges or splits. That is:
    * in-degree = 1 for all nodes except one, which has in-degree = 0
    * out-degree = 1 for all nodes except one, which has out-degree = 0

    Note: this function only tests for node degrees, it does not check if a track is valid.
    Use is_valid_track to check the same.
    Example of a graph which is "simple" but not a valid track might be a graph with two
    disconnected parts: a cycle and a simple track.
    """

    # Helper function to check degree relations
    def check_deg(degree_iterator):
        """
        degree_iterator[i] = (node_id,degree)
        Returns True if all nodes in degree_iterator have degree 1 except one node, which
        has degree 0. Returns False otherwise.
        """
        num_deg_0 = 0  # Count how many 0-deg
        for _, deg in degree_iterator:
            if deg == 0:
                # Update count
                num_deg_0 += 1
                # If more than 1 0-deg node, return False
                if num_deg_0 > 1:
                    return False
            elif deg != 1:
                # If degree not 0 or 1, return False
                return False
        return True

    return check_deg(track.in_degree()) & check_deg(track.out_degree())


def is_track(track: nx.DiGraph, check_simple: bool = False) -> bool:
    """
    Checks if given input is a track, and optionally if it simple.
    If check_simple is True, also checks if the track is simple.

    A graph is a track if it is has
    * directed edges: Each directed edge represents a change that happened in a time step.
    * no directed cycles: As time shouldn't backwards, track cannot have a directed cycles.
    * weakly connected: A single track consists of the same set of particles tracked over
    time, and thus will be a weakly connected graph. Disconnected directed graphs may
    indicate a collection of tracks.

    A track is simple if has no merges or splits. That is:
    * in-degree = 1 for all nodes except one, which has in-degree = 0
    * out-degree = 1 for all nodes except one, which has out-degree = 0
    """
    if check_simple:
        return is_valid_track(track) & is_simple_track(track)
    return is_valid_track(track)


def _cut_edge(graph: nx.DiGraph, edge, node_dup: tuple[bool, bool] = (False, False)):
    """
    Cuts given edge in directed graph. Edge must exist in graph, otherwise result is undefined.

    node_dup defines if a node is duplicated. node_dup is 2-element tuple -- first element defines
    if parent is duplicated, and second if child is duplicated.
        node_dup[i] = True -- node is duplicated
        node_dup[i] = False -- node is not duplicated
    Defaults to (False,False) -- meaning both nodes are not duplicated.

    edge == (parent_node_id, child_node_id)

    Node duplication creates a new node with a different id but the same attributes, and creates
    an edge between this new node and the older child if node_dup is True or parent if node_dup
    is False -- effectively creating a duplicate edge matching the one which was cut. Edge
    attributes are also duplicated, if defined any.

    Raises a ValueError if the given edge does not exist in graph.
    """
    edge_attributes = graph.get_edge_data(*edge)
    if edge_attributes is None:
        raise ValueError(f"Edge {edge} does not exist in given graph: {graph.graph}")
    graph.remove_edge(*edge)
    for ind, nd in enumerate(node_dup):
        if nd:
            old_node = graph.nodes(data=True)[edge[ind]]
            new_node = max(graph.nodes) + 1
            graph.add_node(new_node, **old_node)
            if ind == 0:
                graph.add_edge(new_node, edge[1], **edge_attributes)
            else:
                graph.add_edge(edge[0], new_node, **edge_attributes)


def make_track_simple(
    track: nx.DiGraph,
    graph_cut: Callable,
    modify_track: bool = False,
) -> list[nx.DiGraph]:
    """
    Converts a given track (DiGraph) into multiple simple tracks by cutting merges and splits
    according to given cutting functions. Returns a list of simple track (DiGraph, see is_track
    for definition of simple).

    Merges and splits in given track are defined as:
        Merge: Node with in-degree > 1 == more than one parent
        Split: Node with out-degree > 1 == more than one children
    Note that a node may both be a merge and a split.

    graph_cut should be a two argument functions that return boolean value. make_track_simple will
    call it with the following signature:
        For split node: graph_cut(split_node_attributes,child_node_attributes)
        For merge node: graph_cut(parent_node_attributes,merge_node_attributes)
    Thus, the arguments are always provided in time-order. Note that only attributes are passed to
    graph_cut.

    Note that only one strategy for cutting an edge is supported. Thus, graph_cut should not try to
    change its output based on whether it is operating on a split or a merge.

    For each merge/split, make_track_simple calls graph_cut for each parent/children
    of the merge/split node. For each such call, if graph_cut is True, then a copy of the
    merge/split node is made and attached to the parent/child; otherwise if graph_cut
    is False, no such copy is made.

    If modify_track is True, track is directly modified -- making it into a disconnected graph, each
    being a simple track. If modify_track is False, track is not modified.
    """
    # Check if track is actually a track
    if not is_track(track, check_simple=False):
        raise ValueError("Track is not valid")
    # Make a copy of track
    if not modify_track:
        track = track.to_directed()
    # Find merges and splits
    # A node can both be a merge and a split node
    graph_cuts = []
    for edge in track.edges:
        edge_data = [track.nodes(data=True)[node] for node in edge]
        node_dup = graph_cut(*edge_data)
        # Cut edge only if parent is split or child is merge
        parent_split = track.out_degree(edge[0]) > 1
        child_merge = track.in_degree(edge[1]) > 1
        if parent_split and child_merge:
            node_dup = (node_dup, node_dup)
        elif parent_split:
            node_dup = (node_dup, False)
        elif child_merge:
            node_dup = (False, node_dup)
        else:
            # Do not cut edge
            continue
        graph_cuts.append({"edge": edge, "node_dup": node_dup})
    # Perform cuts
    for gc in graph_cuts:
        _cut_edge(track, **gc)
    # Remove all zero degree nodes
    remove_nodes = [node for node in track.nodes if track.degree(node) == 0]  # type: ignore
    track.remove_nodes_from(remove_nodes)
    # track should be a disconnected collection of simple tracks now. Return
    # connected components
    return [
        track.subgraph(conn_nodes)
        for conn_nodes in nx.weakly_connected_components(track)
    ]


def track_to_dataframe(track: nx.DiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts track into pandas dataframes. Two tables can be generated from a given track:
    spots table and edges table, in that order.
    """
    spots_table = pd.DataFrame(node for _, node in track.nodes(data=True))
    edges_table = pd.DataFrame(edge for _, _, edge in track.edges(data=True))
    return spots_table, edges_table


@dataclass(eq=True, frozen=True)
class SimpleTrack:
    """
    Helper class for TrackCollection. Stores a simple track (DiGraph) along with
    identifying information: id of the track and id of the original track from which
    the simple track is generated, both as assigned in the TrackCollection object.
    """

    track: nx.DiGraph
    id: int
    org_id: int

    def __post_init__(self):
        if not is_track(self.track, check_simple=True):
            raise ValueError("Not a Simple Track")

    @property
    def spots(self):
        """
        Returns spot/node attributes in track object as pandas dataframe
        """
        return track_to_dataframe(self.track)[0]

    @property
    def edges(self):
        """
        Returns edge attributes in track object as pandas dataframe
        """
        return track_to_dataframe(self.track)[1]


class TrackCollection:
    """
    Collection of general tracks (DiGraphs) processed into simple tracks.

    TrackCollection maintains two lists -- a list of the original tracks, accessible
    as org_tracks attribute and a list of the processed simple tracks, accessible as
    tracks attributes. org_tracks is a list of DiGraphs, tracks is a list of SimpleTrack
    objects (containing the DiGraph object with identifying information, see SimpleTrack
    documentation).

    See __init__ doc for details.
    """

    def __init__(
        self,
        tracks: list[nx.DiGraph],
        graph_cut: Optional[Callable[[Any,Any],Any]] = None,
    ):
        """
        Constructs a TrackCollection object using given tracks (as a list of DiGraph objects).
        An optional function graph_cut may also be passed to determine how complex tracks are
        simplified into simple tracks. If graph_cut is not given or is None, a default graph_cut
        which always returns False (that is, cuts all edges to merges or splits, see
        make_track_simple) is used.

        See add_track doc for details.
        """
        # Initialize _org_tracks and _tracks here
        self._org_tracks = []  # _org_tracks Stores the original digraph objects
        self._tracks = []  # _tracks stores simple tracks computed from original tracks

        # Add all tracks

        # Default graph cut -- removes all links between merges and splits
        def default_cut(x, y):  # pylint: disable=unused-argument
            return False

        if graph_cut is None:
            graph_cut = default_cut

        for track in tracks:
            self.add_track(track, graph_cut)

    def add_track(self, track: nx.DiGraph, graph_cut: Callable):
        """
        Adds given (possibly complex) track to TrackCollection, which is simplified
        using given function graph_cut. Complex track is simplified by passing it
        with the graph_cut function to make_track_simple.

        Raises ValueError if given DiGraph object is not a track. Input DiGraph is
        always copied to org_tracks. If input DiGraph is a simple track, it is also
        added as a SimpleTrack to tracks. If input DiGraph is a complex track, it
        is cut into simple tracks by passing it to make_track_simple along with input
        function graph_cut (signature: make_track_simple(track,graph_cut,modify_track=False)).
        Each simple track returned by make_track_simple is added to tracks as a
        SimpleTrack object.

        For each SimpleTrack object, id == index of that object in tracks and
        org_id == index of the DiGraph (original track) from which the SimpleTrack was made
        in org_tracks.
        """
        # Check if track is valid track
        if not is_valid_track(track):
            raise ValueError("Invalid Track")
        track = track.to_directed()
        self._org_tracks.append(track)
        # Check if simple
        if is_simple_track(track):
            self._tracks.append(
                SimpleTrack(track, len(self._tracks), len(self._org_tracks) - 1)
            )
        else:
            # Complex track, make simple tracks
            simple_tracks = make_track_simple(track, graph_cut, modify_track=False)
            for simple_track in simple_tracks:
                self._tracks.append(
                    SimpleTrack(
                        simple_track, len(self._tracks), len(self._org_tracks) - 1
                    )
                )

    @property
    def tracks(self) -> list[SimpleTrack]:
        """
        Returns list of simplified tracks (SimpleTrack) in TrackCollection.
        """
        return self._tracks

    @property
    def org_tracks(self) -> list[nx.DiGraph]:
        """
        Returns list of original tracks (DiGraph) in TrackCollection.
        """
        return self._org_tracks

    def filter_tracks(
        self,
        filter_func: Callable[[SimpleTrack], bool],
    ) -> Iterator[SimpleTrack]:
        """
        Returns an iterator on tracks in collection, filtered by given function filter_func.
        filter_func should take as input either a DiGraph object (input_type == "DiGraph",
        default) or SimpleTrack object (input_type == "SimpleTrack"); and return a boolean
        output.

        filter_func is called on each (simple) track in the collection. For each track, if
        the function returns true, it is included in the iterator. If the function returns
        False, it is excluded from the iterator.
        """

        def iter_tracks():
            for track in self.tracks:
                if filter_func(track):
                    yield track

        return iter_tracks()
