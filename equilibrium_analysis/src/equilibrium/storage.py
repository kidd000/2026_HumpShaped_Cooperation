"""
HDF5 Storage Module for Equilibrium Data
========================================

Efficient storage and retrieval of equilibrium computation results.

HDF5 Structure:
    /metadata/
        version         - Storage format version
        created_at      - Creation timestamp
        grid_shape      - (n_x0, n_K, n_init_coop)
        max_equilibria  - Maximum equilibria per parameter set

    /parameters/
        x0_values       - (n_x0,) float64
        K_values        - (n_K,) float64
        init_coop_values - (n_init_coop,) float64

    /equilibria/
        points          - (n_x0, n_K, n_init_coop, max_eq, 4) float32
                          [p_AllC, p_AllD, p_CC, p_H]
        types           - (n_x0, n_K, n_init_coop, max_eq) int8
                          0=stable, 1=unstable, 2=saddle, -1=empty
        n_equilibria    - (n_x0, n_K, n_init_coop) int8

    /progress/
        completed_x0    - (n_x0,) bool - which x0 slices are done
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from .grid import ParameterGrid


# Equilibrium type constants
EQ_TYPE_STABLE = 0
EQ_TYPE_UNSTABLE = 1
EQ_TYPE_SADDLE = 2
EQ_TYPE_EMPTY = -1


@dataclass
class EquilibriumResult:
    """Result for a single parameter combination."""
    points: np.ndarray  # (n_eq, 4) - [p_AllC, p_AllD, p_CC, p_H]
    types: np.ndarray   # (n_eq,) - stability types
    n_equilibria: int

    @classmethod
    def empty(cls) -> 'EquilibriumResult':
        """Create empty result."""
        return cls(
            points=np.zeros((0, 4), dtype=np.float32),
            types=np.zeros(0, dtype=np.int8),
            n_equilibria=0
        )


class EquilibriumStore:
    """
    HDF5-based storage for equilibrium computation results.

    Examples
    --------
    >>> grid = ParameterGrid.default()
    >>> store = EquilibriumStore.create("equilibria.h5", grid)
    >>> # Save results for x0 slice
    >>> store.save_x0_slice(0, slice_data)
    >>> # Load results
    >>> data = store.load_x0_slice(0)
    >>> store.close()
    """

    STORAGE_VERSION = "1.0"
    DEFAULT_MAX_EQUILIBRIA = 10

    def __init__(self, filepath: str, mode: str = 'r'):
        """
        Open existing HDF5 store.

        Parameters
        ----------
        filepath : str
            Path to HDF5 file
        mode : str
            'r' for read-only, 'r+' for read-write, 'a' for append
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self._file: Optional[h5py.File] = None
        self._grid: Optional[ParameterGrid] = None

    def __enter__(self) -> 'EquilibriumStore':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open the HDF5 file."""
        if self._file is None:
            self._file = h5py.File(self.filepath, self.mode)
            self._load_grid()

    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _load_grid(self):
        """Load parameter grid from file."""
        if self._file is None:
            return

        params = self._file['parameters']
        self._grid = ParameterGrid(
            x0_values=params['x0_values'][:],
            K_values=params['K_values'][:],
            init_coop_values=params['init_coop_values'][:]
        )

    @property
    def grid(self) -> ParameterGrid:
        """Get parameter grid."""
        if self._grid is None:
            raise ValueError("Store not opened")
        return self._grid

    @property
    def max_equilibria(self) -> int:
        """Maximum number of equilibria per parameter set."""
        if self._file is None:
            raise ValueError("Store not opened")
        return int(self._file['metadata'].attrs['max_equilibria'])

    @classmethod
    def create(
        cls,
        filepath: str,
        grid: ParameterGrid,
        max_equilibria: int = DEFAULT_MAX_EQUILIBRIA,
        overwrite: bool = False
    ) -> 'EquilibriumStore':
        """
        Create new HDF5 store.

        Parameters
        ----------
        filepath : str
            Path for new HDF5 file
        grid : ParameterGrid
            Parameter grid definition
        max_equilibria : int
            Maximum equilibria to store per parameter set
        overwrite : bool
            If True, overwrite existing file

        Returns
        -------
        EquilibriumStore
            Opened store ready for writing
        """
        filepath = Path(filepath)

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File exists: {filepath}. Use overwrite=True to replace.")

        with h5py.File(filepath, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['version'] = cls.STORAGE_VERSION
            meta.attrs['created_at'] = datetime.now().isoformat()
            meta.attrs['grid_shape'] = grid.shape
            meta.attrs['max_equilibria'] = max_equilibria

            # Parameters
            params = f.create_group('parameters')
            params.create_dataset('x0_values', data=grid.x0_values)
            params.create_dataset('K_values', data=grid.K_values)
            params.create_dataset('init_coop_values', data=grid.init_coop_values)

            # Equilibria data (pre-allocate)
            n_x0, n_K, n_init_coop = grid.shape
            eq_group = f.create_group('equilibria')

            # Points: (n_x0, n_K, n_init_coop, max_eq, 4)
            eq_group.create_dataset(
                'points',
                shape=(n_x0, n_K, n_init_coop, max_equilibria, 4),
                dtype=np.float32,
                fillvalue=np.nan,
                chunks=(1, n_K, n_init_coop, max_equilibria, 4),  # Chunk by x0 slice
                compression='gzip',
                compression_opts=4
            )

            # Types: (n_x0, n_K, n_init_coop, max_eq)
            eq_group.create_dataset(
                'types',
                shape=(n_x0, n_K, n_init_coop, max_equilibria),
                dtype=np.int8,
                fillvalue=EQ_TYPE_EMPTY,
                chunks=(1, n_K, n_init_coop, max_equilibria),
                compression='gzip',
                compression_opts=4
            )

            # Number of equilibria: (n_x0, n_K, n_init_coop)
            eq_group.create_dataset(
                'n_equilibria',
                shape=(n_x0, n_K, n_init_coop),
                dtype=np.int8,
                fillvalue=0,
                chunks=(1, n_K, n_init_coop),
                compression='gzip',
                compression_opts=4
            )

            # Progress tracking
            progress = f.create_group('progress')
            progress.create_dataset(
                'completed_x0',
                shape=(n_x0,),
                dtype=bool,
                fillvalue=False
            )

        # Return opened store
        store = cls(filepath, mode='r+')
        store.open()
        return store

    def save_x0_slice(
        self,
        x0_idx: int,
        data: Dict[Tuple[int, int], EquilibriumResult]
    ):
        """
        Save equilibrium results for one x0 slice.

        Parameters
        ----------
        x0_idx : int
            Index of x0 value
        data : Dict[Tuple[int, int], EquilibriumResult]
            Dictionary mapping (K_idx, init_coop_idx) to EquilibriumResult
        """
        if self._file is None:
            raise ValueError("Store not opened")

        eq_group = self._file['equilibria']
        max_eq = self.max_equilibria

        # Prepare arrays for this slice
        n_K = self.grid.n_K
        n_init_coop = self.grid.n_init_coop

        points_slice = np.full((n_K, n_init_coop, max_eq, 4), np.nan, dtype=np.float32)
        types_slice = np.full((n_K, n_init_coop, max_eq), EQ_TYPE_EMPTY, dtype=np.int8)
        n_eq_slice = np.zeros((n_K, n_init_coop), dtype=np.int8)

        # Fill in data
        for (k_idx, ic_idx), result in data.items():
            n_eq = min(result.n_equilibria, max_eq)
            n_eq_slice[k_idx, ic_idx] = n_eq

            if n_eq > 0:
                points_slice[k_idx, ic_idx, :n_eq, :] = result.points[:n_eq]
                types_slice[k_idx, ic_idx, :n_eq] = result.types[:n_eq]

        # Write to HDF5
        eq_group['points'][x0_idx] = points_slice
        eq_group['types'][x0_idx] = types_slice
        eq_group['n_equilibria'][x0_idx] = n_eq_slice

        # Mark as completed
        self._file['progress']['completed_x0'][x0_idx] = True
        self._file.flush()

    def load_x0_slice(self, x0_idx: int) -> Dict[Tuple[int, int], EquilibriumResult]:
        """
        Load equilibrium results for one x0 slice.

        Parameters
        ----------
        x0_idx : int
            Index of x0 value

        Returns
        -------
        Dict[Tuple[int, int], EquilibriumResult]
            Dictionary mapping (K_idx, init_coop_idx) to EquilibriumResult
        """
        if self._file is None:
            raise ValueError("Store not opened")

        eq_group = self._file['equilibria']

        points_slice = eq_group['points'][x0_idx]
        types_slice = eq_group['types'][x0_idx]
        n_eq_slice = eq_group['n_equilibria'][x0_idx]

        data = {}
        for k_idx in range(self.grid.n_K):
            for ic_idx in range(self.grid.n_init_coop):
                n_eq = n_eq_slice[k_idx, ic_idx]
                if n_eq > 0:
                    data[(k_idx, ic_idx)] = EquilibriumResult(
                        points=points_slice[k_idx, ic_idx, :n_eq],
                        types=types_slice[k_idx, ic_idx, :n_eq],
                        n_equilibria=n_eq
                    )
                else:
                    data[(k_idx, ic_idx)] = EquilibriumResult.empty()

        return data


def open_store(filepath: str, mode: str = 'r') -> EquilibriumStore:
    """
    Open existing equilibrium store.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file
    mode : str
        'r' for read-only, 'r+' for read-write

    Returns
    -------
    EquilibriumStore
        Opened store
    """
    store = EquilibriumStore(filepath, mode)
    store.open()
    return store
