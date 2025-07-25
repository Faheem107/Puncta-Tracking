"""
Phase portrait class defintion

# TODO: Main phase portrait class should provide an easier access point
# Meaning it should only have one value/rate pair per column, should not
# modify rate df and just bin with no filters
# May not be completely required -- just simplified instances could also
# be provided

# TODO: Provide plotting support as separate function(s)

"""

from types import EllipsisType
from typing import Callable, Optional, Any
from functools import wraps, singledispatchmethod
from collections import Counter

import pandas as pd
from networkx import DiGraph

from .tracks import TrackCollection, SimpleTrack

__DEFAULTS__ = {
    "time_col": "FRAME",
    "track_name": "tracks",
    "spot_name": "spots",
    "val_name": "val",
    "rate_name": "rate",
}


class RateCalculator:
    """Factory class for calculating rates on a collection of tracks.

    A track in the context of this class is defined as a `SimpleTrack` object.
    Each track contains multiple spots and properties measured for each spot,
    stored as a pandas DataFrame in `SimpleTrack.spots` in which each spot is
    an individual row and each property measured for a spot is a column. A
    track also has an id attribute, which identifies it within its
    `TrackCollection`. All tracks must have the same set of spot properties.

    Each instance of this class is a callable object that calculates "rates"
    for each spot property over all tracks, based on user-defined functions
    that can define multiple "rates" for each spot_property. See doc for
    `__call__` for details.

    Attributes
    ----------
    track_collection: TrackCollection
        Tracks to use for rate calculation
    track_filter: Callable[[SimpleTrack],bool] | None
        User-defined filter that selects which tracks to use for rate
        calculation.
    time_col: str
        Spot property recording time information, see doc for `__call__` for
        details
    track_name: str
        Index name for track ids, see doc for `__call__` for details
    spot_name: str
        Index name for spot ids, see doc for `__call__` for details
    """

    def __init__(
        self,
        tracks: TrackCollection,
        track_filter: Optional[Callable[[SimpleTrack], bool]] = None,
        time_col: str = __DEFAULTS__["time_col"],
        track_name: str = __DEFAULTS__["track_name"],
        spot_name: str = __DEFAULTS__["spot_name"],
    ):
        """Default constructor for RateCalculator

        Parameters
        ----------
        tracks : TrackCollection
            Tracks to use in rate calculation, collected as a `TrackCollection` object
        track_filter : Callable[[SimpleTrack], bool], optional
            Callable to filter tracks during rate calculation, by default None. If None,
            no filtering is applied. If not None, is passed to `tracks.filter_tracks()`.
        time_col: str
            Spot property recording time information, see doc for `__call__` for
            details
        track_name : str, optional
            Index level name for , by default "tracks"
        spot_name : str, optional
            Index level name for spot id, by default "spots"
        """
        self.track_collection = tracks
        self.track_name = track_name
        self.spot_name = spot_name
        self.track_filter = track_filter
        self.time_col = time_col

    @staticmethod
    def wrap_filter(
        track_filter: Optional[Callable[[DiGraph], bool]] = None
    ) -> Callable[[SimpleTrack], bool] | None:
        """Converts track filter using DiGraph into one using SimpleTrack.

        A track filter is a function which takes as input a track and returns a
        boolean. This function is passed to `TrackCollection.filter_tracks`

        Parameters
        ----------
        track_filter : Callable[[DiGraph], bool], optional
            Track filter using `DiGraph` as track input, by default None

        Returns
        -------
        Callable[[SimpleTrack], bool] | None
            If track_filter is None, returns None
            If track_filter is not None, returns a wrapper around given
            track_filter which takes `SimpleTrack` as track input.
        """
        if track_filter is None:
            return None

        @wraps(track_filter)
        def wrapped(track: SimpleTrack):
            return track_filter(track.track)

        return wrapped

    @classmethod
    def from_digraphs(
        cls,
        tracks: list[DiGraph],
        graph_cut: Optional[Callable[[Any, Any], Any]] = None,
        track_filter: Optional[Callable[[DiGraph], bool]] = None,
        **kwargs,
    ):
        """Constructs RateCalculator object with its own `TrackCollection`.

        Parameters
        ----------
        tracks : list[DiGraph]
            List of tracks (simple or complex) to use for rate calculation. Used
            to construct `TrackCollection` object.
        graph_cut : Callable[[Any, Any], Any], optional
            Callable used to simplfy complex tracks, by default None. Used to
            construct `TrackCollection` object.
        track_filter : Callable[[DiGraph], bool], optional
            Callable to filter tracks during rate calculation, by default None.
            If None, no filtering is applied. If not None, is wrapped by wrap_filter
            before passing to TrackCollection.filter_tracks
        **kwargs: dict, optional
            Extra arguments to '__init__, see doc for __init__ for details.

        Notes
        -----
        This classmethod is an alternative constructor for RateCalculator, to be used
        to create a RateCalculator object directly from a list of DiGraph objects
        (understood as tracks). A new TrackCollection is created from given tracks and
        graph_cut function.

        As this constructor is meant for DiGraph objects, track_filter is assumed to
        take a DiGraph object; and is thus wrapped by wrap_filter before constructing
        RateCalculator.
        """
        return cls(
            TrackCollection(tracks, graph_cut),
            cls.wrap_filter(track_filter),
            **kwargs,
        )

    def __call__(
        self,
        rate_col: dict[Any, Callable[[pd.Series, pd.Series], list[pd.Series]]],
        only_val_cols: Optional[set[str]] = None,
        val_name: str = __DEFAULTS__["val_name"],
        rate_name: str = __DEFAULTS__["rate_name"],
    ) -> pd.DataFrame:
        """
        Calculate rate for each track in RateCalculator's track collection.

        Parameters
        ----------
        rate_col : dict[str, Callable[[pd.Series,pd.Series], list[pd.Series]]]
            User-defined functions that define how to calculate rate for each
            spots property, arranged in the format:
                {"spot_property":function_to_calculate_rate}
            For each spot_property, the function to calculate rate should take
            in as input two pandas series, the first containing values of the
            spot property of interest and second containing values of the spot
            property indicated in `self.time_col`; and return a list of pandas
            Series, each being a "rate". Calculated rates are matched to their
            corresponding spots by index.
        only_val_cols: set[str] | None
            List of spot properties for which no rate calculation is done, but
            are still included in the returned DataFrame. Any spot properties
            already in `rate_col` are not duplicated.
        val_name: str
            Column name for values of spot properties, see below. By default
            "val"
        rate_name: str
            Column name for calculated rates for each spot property, see below.
            By default, "rate".

        Returns
        -------
        pd.DataFrame
            Returned DataFrame has the following structure:
            * Columns have two levels. Outer (second) level is spot properties.
            For each spot property, the first column in the inner (first) level
            contains the values of the spot property itself (colname is
            defined by `val_name`). All other columns in a spot property
            are the calculated rates for that property, listed in the same
            order as the output of the corresponding rate calculation function
            (colname is defined as f"{`rate_name`}_{index_in_out_list}"). For
            spot properties where no rate calculation is done, only the first
            column at the inner level (defined by `val_name`) is present.
            * Index has two levels. Outer (second) level identifies the track
            in `self.TrackCollection.tracks` and the inner (first) level
            identifies the spot in the track. Index names are defined by
            `self.track_name` (outer level) and `self.spot_name` (inner level).
        """
        if only_val_cols is None:
            only_val_cols = set()

        if self.track_filter is None:
            tracks_iter = self.track_collection.tracks
        else:
            tracks_iter = self.track_collection.filter_tracks(self.track_filter)

        # Enable Copy on Write for pandas, better performance
        with pd.option_context("mode.copy_on_write", True):
            rates = []
            for track in tracks_iter:
                # Get value and rates for each spot property as a separate dataframe
                dfs = {}
                for colname, colfunc in rate_col.items():
                    val = track.spots[colname]
                    rate = colfunc(val, track.spots[self.time_col])
                    rate_names = [
                        val_name,
                        *[f"{rate_name}_{id}" for id in range(len(rate))],
                    ]
                    dfs[colname] = pd.concat([val, *rate], axis=1, keys=rate_names)
                for colname in only_val_cols:
                    if colname not in dfs:
                        val = track.spots[colname]
                        dfs[colname] = pd.concat([val], axis=1, keys=[val_name])
                # Combine into one dataframe with correct column levels
                dfs = pd.concat(dfs.values(), axis=1, keys=dfs.keys())
                # Set MultiIndex for dataframe index
                idx = dfs.index.to_frame()
                idx = idx.rename(columns={0: self.spot_name})
                idx.insert(0, self.track_name, track.id)
                dfs.index = pd.MultiIndex.from_frame(idx)
                rates.append(dfs)
            return pd.concat(rates, axis=0)


class _GroupByIndexer:

    __groupby_functions__ = [
        "agg",
        "aggregate",
        "all",
        "any",
        "count",
        "describe",
        "max",
        "min",
        "mean",
        "median",
        "min",
        "sem",
        "std",
        "skew",
        "sum",
        "var",
    ]

    def __init__(self, grouped_data, bins: dict):
        self.data = grouped_data
        self.bins = bins

    def __getitem__(self, idx):
        return _GroupByIndexer(self.data[idx], self.bins)

    @property
    def df(self):
        """Unbinned (combined) data from the bins selected"""
        return self.data.obj

    def _agg(self, agg_func):
        @wraps(agg_func)
        def wrapped(*args, **kwargs):
            agg_df = agg_func(*args, **kwargs)
            idx = pd.MultiIndex.from_tuples(agg_df.index)
            agg_df.index = idx.set_names([f"bin_{col}" for col in self.bins])
            return agg_df.to_xarray()

        return wrapped

    def __getattr__(self, name):
        if name in self.__groupby_functions__:
            return self._agg(self.data.__getattribute__(name))
        return self.data.__getattribute__(name)


class BinDataFrame:
    """
    Bins dataframes across multiple columns using pd.cut

    Binning over multiple columns implies that each bin is specified using
    n 1d bins, where n is the number of binned columns. Thus, binning over
    multiple columns is understood to imply a grid of bins, with each binned
    column acting as one dimension of this bin.

    Once constructed, a BinDataFrame object may be indexed like a numpy
    array in order to access different individual bins or collection of bins.
    For example:
        df = pd.DataFrame({'x_1':[x1 values],'x_2':[x2 values],...})
        bdf = BinDataFrame(df, bins = {'x_1':[bin intervals], ...})
        bdf[i,j,...] -> ith bin interval in x_1, jth bin interval in x_2, ...
        bdf[i1:i2,...] -> i1  to i2 bin intervals in x_1 (python slicing), ...
        bdf[...] -> All bins

    Operations defined on pd.GroupBy can be applied after indexing. These
    operations, as far as possible, will return xarray Dataset and DataArray
    objects with dimensions being the bins in the binned columns instead of
    just dataframes.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe to use for binning.
    bins_col: dict
        Which columns to bin and how to bin them. Columns to bin in the
        dataframe should be given as the keys of the dictionary. For each
        column to bin (key), its corresponding value is passed to pd.cut as
        its bins argument to calculate the bins for that column. Any column
        which is not a key in bins_col is not binned.

    Attributes
    ----------
    df: pd.DataFrame
        Original dataframe used for binning
    bins: dict
        Bins for each column. Binned columns are keys of this dictionary. For
        each binned column, the corresponding value is the bins for that column.
    ndim: int
        How many columns have been binned.
    shape: tuple[int, ...]
        Shape of the binning grid. nth dimension represents the column which is
        the nth key in bins.

    Methods
    -------
    remake_bins: (bins_col: dict, **kwargs)
        Redefines the binned grid with the one given by bins_col
    """

    # Binning variables
    _bins_idx: pd.DataFrame
    _combined: pd.Series

    def __init__(self, df: pd.DataFrame, bins_col: dict, **kwargs):
        self.df = df
        self.bins = {}
        self.remake_bins(bins_col, **kwargs)

    def remake_bins(self, bins_col: dict, **kwargs):
        """Redefines the binned grid with the one given by bins_col

        Parameters
        ----------
        bins_col : dict
            New definition for bins. Which columns to bin and how to bin them.
            Columns to bin in the dataframe (self.df) should be given as the
            keys of the dictionary. For each column to bin (key), its
            corresponding value is passed to pd.cut as its bins argument to
            calculate the bins for that column. Any column which is not a key
            in bins_col is not binned.
        """
        self.bins = {}
        self._bins_idx = {}  # type: ignore
        for bin_col, bin_calc in bins_col.items():
            if callable(bin_calc):
                bin_calc = bin_calc(self.df[bin_col])
            self._bins_idx[bin_col] = pd.cut(
                self.df[bin_col],
                bin_calc,
                labels=None,
                retbins=False,
                ordered=True,
                **kwargs,
            )
            self.bins[bin_col] = self._bins_idx[bin_col].cat.categories
        self._bins_idx = pd.DataFrame(self._bins_idx)
        self._combined = self._bins_idx.apply(tuple, axis=1)

    @property
    def ndim(self):
        """How many columns have been binned."""
        return len(self.bins)

    @property
    def shape(self):
        """Shape of the binning grid. nth dimension represents the column which is
        the nth key in bins.
        """
        return tuple(len(curr_bin) for _, curr_bin in self.bins.items())

    @singledispatchmethod
    def __getitem__(self, indices):
        raise TypeError(f"Indices must be integers or slices, not {type(indices)}")

    @__getitem__.register
    def _(self, indices: int | slice | EllipsisType):
        return self[(indices,)]  # type: ignore

    @__getitem__.register  # type: ignore
    def _(self, indices: tuple):
        if Ellipsis in indices:
            cnt = Counter(indices)
            if cnt[Ellipsis] > 1:
                raise IndexError("An index can only have a single ellipsis ('...')")
            ellipsis_index = indices.index(Ellipsis)
            indices = (
                *indices[:ellipsis_index],
                *[slice(None) for _ in range(self.ndim - len(indices) + 1)],
                *indices[ellipsis_index + 1 :],
            )
            return self[indices]  # type: ignore
        if len(indices) < self.ndim:
            extra_indices = [slice(None) for _ in range(self.ndim - len(indices))]
            indices = (*indices, *extra_indices)
            return self[indices]  # type: ignore

        return self._getitem(indices)

    def _getitem(self, indices: tuple[int | slice | tuple, ...]):
        if len(indices) != self.ndim:
            raise ValueError(
                f"Too many indices for {type(self)}: PhasePortrait is {self.ndim}-dimensional, but {len(indices)} were indexed"
            )

        indexer = True
        new_bins = {}
        for index, col in zip(indices, self.bins):
            new_bins[col] = self.bins[col][index]
            match index:
                case int():
                    indexer &= self._bins_idx[col] == self.bins[col][index]
                case slice():
                    group_indexer = False
                    for bin_col in self.bins[col][index]:
                        group_indexer |= self._bins_idx[col] == bin_col
                    indexer &= group_indexer
                case _:
                    raise TypeError(f"Invalid index type {type(index)}")

        grouped = self.df.loc[indexer, :].groupby(
            by=self._combined[indexer], sort=False, observed=True, dropna=True
        )
        return _GroupByIndexer(grouped, new_bins)

    def __getattr__(self, name):
        return self[...].__getattribute__(name)  # type: ignore
