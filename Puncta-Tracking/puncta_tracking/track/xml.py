"""
XML reading for trackmate and converting to Networkx Digraph
"""

from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from lxml import etree
import pandas as pd
from networkx import DiGraph


def parse_trackmate_xml(path_xml_file: Path | str):
    """
    Returns a parsed version of the xml file. Adds track id to edges
    """
    # Parse xml -- pylance complains that parse is not available in
    # etree, even though it is. Hence type: ignore.
    xml_tree = etree.parse(path_xml_file) # type: ignore
    # Add track id to edges
    for edge in xml_tree.iter("Edge"):
        edge.set("TRACK_ID", edge.getparent().get("TRACK_ID"))
    return xml_tree


def get_features(features_nodes_iter: Iterable, feature_dict: dict) -> dict:
    """
    Looks for feature tags in children of features_node, and appends their
    feature attribute (as key) and isint (as value) attributes to feature_dict.
    Returns updated feature_dict. For isint, the value for the key is True if
    isint is 'true', False otherwise.
    """
    for feature in features_nodes_iter:
        feature_dict[feature.get("feature")] = feature.get("isint").lower() == "true"
    return feature_dict


def extract_features(
    nodes_iter: Iterable, feature_dict: dict
) -> tuple[pd.DataFrame, set]:
    """
    Extracts the attributes in each node in node_iter, as dictated by
    feature_dict. For each node, the functions checks if a key of feature_dict
    is an attribute for the node and extracts its value if it is. If value
    for the key is true, the value is converted to int -- otherwise it is
    converted to float. Returns a pandas dataframe, with the column headings
    being the keys in feature_dict and each row being the values extracted for
    each node in nodes_iter. Also returns a set of the keys which were not
    detected as attributes in atleast one node, called features_not_detected.
    """
    # Loop through all nodes. Extract all features in feature list
    # Get attributes for each node -- as a list of dictionaries
    # Keep track of features not detected in atleast one node
    node_props = []
    features_not_detected = []
    for node in nodes_iter:
        props_dict = {}
        for feature, isint in feature_dict.items():
            # If a feature is not present, set to nan
            if node.get(feature) is None:
                features_not_detected.append(feature)
                props_dict[feature] = np.nan
                continue
            # If feature is present
            props_dict[feature] = node.get(feature)
            if isint:  # If feature is integer
                props_dict[feature] = int(props_dict[feature])
            else:  # Otherwise set to float
                props_dict[feature] = float(props_dict[feature])
        node_props.append(props_dict)
    # Convert and return
    node_props = pd.DataFrame.from_records(node_props)
    features_not_detected = set(features_not_detected)
    return node_props, features_not_detected


def extract_trackmate_output(
    xml_features: Iterable,
    xml_nodes: Iterable,
    feature_dict: Optional[dict] = None,
    id_function: Optional[Callable] = None,
    remove_not_detected: bool = True,
) -> pd.DataFrame:
    """
    Template function for extract_spots, extract_edges, extract_tracks
    """
    # Defaults
    if feature_dict is None:
        feature_dict = {}
    # Get feature list from xml_features
    feature_dict = get_features(xml_features, feature_dict)
    # Extract attributes as feature values from xml_nodes
    nodes, features_not_detected = extract_features(xml_nodes, feature_dict)
    # Set dataframe index
    if id_function is not None:
        nodes = id_function(nodes)
    # Remove not detected features
    if remove_not_detected:
        nodes.drop(columns=features_not_detected, inplace=True)  # type: ignore
    # Return dataframe
    return nodes


def extract_spots(xml_tree, remove_not_detected_features: bool = True) -> pd.DataFrame:
    """
    Extracts spots as a pandas dataframe from a parsed xml tree (using
    xml.etree.ElementTree). Takes as input the parsed xml_tree and a boolean
    remove_not_detected columns (default True). If remove_not_detected_columns
    is true, any features not detected in atleast one spot tag are removed
    from the returned pandas dataframe. The ID of the spots is used as the
    index in the returned dataframe.
    """
    # Get feature list for spots
    spot_features = xml_tree.find(".//SpotFeatures").iter("Feature")
    # Get all spots
    spots = xml_tree.iter("Spot")
    # ID is not reported in the feature list. Add it separately
    feature_dict = {"ID": True}  # ID is integer
    return extract_trackmate_output(
        xml_features=spot_features,
        xml_nodes=spots,
        feature_dict=feature_dict,
        id_function=lambda x: x.set_index("ID"),
        remove_not_detected=remove_not_detected_features,
    )


def extract_edges(xml_tree, remove_not_detected_features: bool = True) -> pd.DataFrame:
    """
    Extracts edges as a pandas dataframe from a parsed xml tree (using
    xml.etree.ElementTree). Takes as input the parsed xml_tree and a boolean
    remove_not_detected columns (default True). If remove_not_detected_columns
    is true, any features not detected in atleast one edge tag are removed
    from the returned pandas dataframe.
    """
    # Get feature list for spots
    edge_features = xml_tree.find(".//EdgeFeatures").iter("Feature")
    # Get all spots
    edges = xml_tree.iter("Edge")
    # TRACK_ID is added separately
    feature_dict = {"TRACK_ID": True}

    def id_function(data):
        edges_id = []
        for source, target in zip(data["SPOT_SOURCE_ID"], data["SPOT_TARGET_ID"]):
            edges_id.append(f"{source}->{target}")
        data["ID"] = edges_id
        return data.set_index("ID")

    return extract_trackmate_output(
        xml_features=edge_features,
        xml_nodes=edges,
        feature_dict=feature_dict,
        id_function=id_function,
        remove_not_detected=remove_not_detected_features,
    )


def extract_tracks(xml_tree, remove_not_detected_features: bool = True) -> pd.DataFrame:
    """
    Extracts tracks as a pandas dataframe from a parsed xml tree (using
    xml.etree.ElementTree). Takes as input the parsed xml_tree and a boolean
    remove_not_detected columns (default True). If remove_not_detected_columns
    is true, any features not detected in atleast one track tag are removed
    from the returned pandas dataframe.
    """
    # Get feature list for spots
    track_features = xml_tree.find(".//TrackFeatures").iter("Feature")
    # Get all spots
    tracks = xml_tree.iter("Track")
    return extract_trackmate_output(
        xml_features=track_features,
        xml_nodes=tracks,
        id_function=lambda x: x.set_index("TRACK_ID"),
        remove_not_detected=remove_not_detected_features,
    )


def extract_tracks_graphs(
    spots: pd.DataFrame, edges: pd.DataFrame, tracks: pd.DataFrame
) -> list[DiGraph]:
    """
    Takes as input the extracted spots, edges and tracks as pandas dataframe
    (using extract_spots,extract_edges,extract_tracks_info). Returns a list
    of DiGraph objects (from NetworkX), each representing an individual
    track with its edges and spots.
    """

    def add_edges_graph(edge, graph):
        source, target = edge["SPOT_SOURCE_ID"], edge["SPOT_TARGET_ID"]
        graph.add_node(source, **spots.loc[source].to_dict())
        graph.add_node(target, **spots.loc[target].to_dict())
        graph.add_edge(source, target, **edge.to_dict())

    track_graphs = [DiGraph() for _ in tracks.index]
    # Iterate over tracks
    for track_id, track_graph in zip(tracks.index, track_graphs):
        # Create graph and set graph attributes
        track_graph.graph = tracks.loc[track_id].to_dict()
        # Get all edges in this track
        edges_track = edges.loc[edges["TRACK_ID"] == track_id]
        # Add nodes for each edge, and then the edge itself
        edges_track.apply(lambda edge: add_edges_graph(edge, track_graph), axis=1)
    return track_graphs
