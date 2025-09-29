"""Utilities for loading pre-computed embeddings for analysis and visualization.

This module provides utilities for loading sharded embedding files created by
embedding extraction scripts (scripts/extract_embeddings.py, scripts/embed_annotations.py).

**NOTE**: This is for ANALYSIS embeddings (SAE normalization, concept analysis, dashboard).
For TRAINING embeddings, use `interplm/train/data_loader.py:ActivationsDataLoader` instead.

Supported Formats:
-----------------
1. Flat File Format:
   data_dir/
   ├── shard_0.pt  (torch tensor or dict with 'embeddings' key)
   ├── shard_1.pt
   └── ...

2. Nested Directory Format (with metadata):
   data_dir/
   ├── shard_0/
   │   ├── embeddings.pt  (tensor or dict with 'embeddings', 'boundaries', 'protein_ids')
   │   └── metadata.yaml (optional)
   ├── shard_1/
   └── ...

Usage:
------
**Simple loading (for single shards):**

    from interplm.data_processing.embedding_loader import load_shard_embeddings

    # Auto-detects format and loads shard 0
    embeddings = load_shard_embeddings('/path/to/embeddings', shard_idx=0)

**Batch loading (for iterating over all shards):**

    from interplm.data_processing.embedding_loader import detect_and_create_loader

    loader = detect_and_create_loader('/path/to/embeddings')

    # Iterate over all shards
    for shard_idx, embeddings in loader.iter_shards():
        process(embeddings)

    # Or load specific shard with metadata
    data = loader.load_shard_with_metadata(0)
    embeddings = data['embeddings']
    boundaries = data['boundaries']  # May be None
    protein_ids = data['protein_ids']  # May be None
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union
import torch


def load_shard_embeddings(
    embeddings_dir: Path,
    shard_idx: int,
    device: str = 'cpu',
    return_tensor_only: bool = True
) -> Union[torch.Tensor, dict]:
    """
    Load embeddings from a single shard, automatically detecting format.

    Tries both flat format (shard_X.pt) and nested format (shard_X/embeddings.pt or shard_X/activations.pt).

    Args:
        embeddings_dir: Directory containing embedding shards
        shard_idx: Shard index to load
        device: Device to load tensor to ('cpu' or 'cuda')
        return_tensor_only: If True, extract and return only the embeddings tensor.
                           If False, return the full data structure (may be dict or tensor).

    Returns:
        If return_tensor_only=True: torch.Tensor of embeddings (n_amino_acids × d_model)
        If return_tensor_only=False: Either a tensor or a dict with 'embeddings', 'boundaries', 'protein_ids' keys

    Raises:
        FileNotFoundError: If shard file not found in any format
    """
    embeddings_dir = Path(embeddings_dir)

    # Try flat format first (shard_X.pt)
    flat_path = embeddings_dir / f"shard_{shard_idx}.pt"
    if flat_path.exists():
        data = torch.load(flat_path, map_location=device, weights_only=True)
        if return_tensor_only and isinstance(data, dict) and 'embeddings' in data:
            return data['embeddings']
        return data

    # Try nested format (shard_X/embeddings.pt or shard_X/activations.pt)
    shard_dir = embeddings_dir / f"shard_{shard_idx}"
    if shard_dir.is_dir():
        # Try common filenames
        for filename in ["embeddings.pt", "activations.pt", "data.pt"]:
            nested_path = shard_dir / filename
            if nested_path.exists():
                data = torch.load(nested_path, map_location=device, weights_only=True)
                if return_tensor_only and isinstance(data, dict) and 'embeddings' in data:
                    return data['embeddings']
                return data

    # Not found in any format
    raise FileNotFoundError(
        f"Could not find shard {shard_idx} in {embeddings_dir}. "
        f"Tried: {flat_path} and {shard_dir}/{{embeddings,activations,data}}.pt"
    )


class ShardDataLoader(ABC):
    """Abstract base class for loading sharded data."""

    def __init__(self, data_dir: Path, device: str = 'cpu'):
        self.data_dir = Path(data_dir)
        self._device = device

    @abstractmethod
    def get_shard_count(self) -> int:
        """Return the total number of shards available."""
        pass

    @abstractmethod
    def get_available_shard_indices(self) -> list[int]:
        """Return list of available shard indices."""
        pass

    def get_shard_indices(self, n: Optional[int] = None) -> list[int]:
        """
        Get shard indices to process.

        Args:
            n: Number of shards to process (None for all)

        Returns:
            List of shard indices
        """
        available_indices = self.get_available_shard_indices()
        if n is None:
            return available_indices
        return available_indices[:n]

    @abstractmethod
    def load_shard(self, shard_idx: int) -> torch.Tensor:
        """Load data from a specific shard."""
        pass

    def iter_shards(self) -> Iterator[Tuple[int, torch.Tensor]]:
        """Iterate over all shards, yielding (shard_idx, data)."""
        for shard_idx in self.get_available_shard_indices():
            yield shard_idx, self.load_shard(shard_idx)

    @abstractmethod
    def load_shard_with_metadata(self, shard_idx: int) -> dict:
        """Load shard with all metadata (embeddings, boundaries, protein_ids if available)."""
        pass


class FlatFileShardLoader(ShardDataLoader):
    """Handles shard_0.pt, shard_1.pt, etc in single directory."""

    def __init__(self, data_dir: Path, shard_pattern: str = "shard_*.pt", device: str = 'cpu'):
        super().__init__(data_dir, device)
        self.shard_pattern = shard_pattern
        self._shard_files = sorted(self.data_dir.glob(self.shard_pattern))

        # Extract shard indices from filenames and sort them
        self._shard_indices = []
        for shard_file in self._shard_files:
            # Extract number from shard_X.pt format
            try:
                shard_num = int(shard_file.stem.split('_')[1])
                self._shard_indices.append(shard_num)
            except (IndexError, ValueError):
                # If we can't parse the shard number, skip this file
                continue
        self._shard_indices.sort()

    def get_shard_count(self) -> int:
        return len(self._shard_indices)

    def get_available_shard_indices(self) -> list[int]:
        return self._shard_indices.copy()

    def load_shard(self, shard_idx: int) -> torch.Tensor:
        if shard_idx not in self._shard_indices:
            raise IndexError(f"Shard index {shard_idx} not available. Available: {self._shard_indices}")

        shard_path = self.data_dir / f"shard_{shard_idx}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        data = torch.load(shard_path, map_location=self._device, weights_only=True)

        # Handle both old format (just tensor) and new format (dict with embeddings key)
        if isinstance(data, dict):
            return data.get('embeddings', data)  # Return embeddings if dict, otherwise whole dict
        return data

    def load_shard_with_metadata(self, shard_idx: int) -> dict:
        """Load shard with all metadata (embeddings, boundaries, protein_ids if available)."""
        if shard_idx not in self._shard_indices:
            raise IndexError(f"Shard index {shard_idx} not available. Available: {self._shard_indices}")

        shard_path = self.data_dir / f"shard_{shard_idx}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        data = torch.load(shard_path, map_location=self._device, weights_only=True)

        # If it's already a dict with metadata, return it
        if isinstance(data, dict) and 'boundaries' in data:
            return data

        # Otherwise, return just embeddings (old format)
        return {'embeddings': data, 'boundaries': None, 'protein_ids': None}


class NestedFolderShardLoader(ShardDataLoader):
    """Handles shard_0/activations.pt, shard_1/activations.pt, etc."""

    def __init__(self, data_dir: Path, filename: str = "activations.pt", device: str = 'cpu'):
        super().__init__(data_dir, device)
        self.filename = filename
        self._shard_dirs = sorted([d for d in self.data_dir.glob("shard_*") if d.is_dir()])

        # Extract shard indices from directory names and sort them
        self._shard_indices = []
        for shard_dir in self._shard_dirs:
            # Extract number from shard_X directory format
            try:
                shard_num = int(shard_dir.name.split('_')[1])
                self._shard_indices.append(shard_num)
            except (IndexError, ValueError):
                # If we can't parse the shard number, skip this directory
                continue
        self._shard_indices.sort()

    def get_shard_count(self) -> int:
        return len(self._shard_indices)

    def get_available_shard_indices(self) -> list[int]:
        return self._shard_indices.copy()

    def load_shard(self, shard_idx: int) -> torch.Tensor:
        if shard_idx not in self._shard_indices:
            raise IndexError(f"Shard index {shard_idx} not available. Available: {self._shard_indices}")

        shard_path = self.data_dir / f"shard_{shard_idx}" / self.filename
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        data = torch.load(shard_path, map_location=self._device, weights_only=True)

        # Handle both old format (just tensor) and new format (dict with embeddings key)
        if isinstance(data, dict):
            return data.get('embeddings', data)  # Return embeddings if dict, otherwise whole dict
        return data

    def load_shard_with_metadata(self, shard_idx: int) -> dict:
        """Load shard with all metadata (embeddings, boundaries, protein_ids if available)."""
        if shard_idx not in self._shard_indices:
            raise IndexError(f"Shard index {shard_idx} not available. Available: {self._shard_indices}")

        shard_path = self.data_dir / f"shard_{shard_idx}" / self.filename
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        data = torch.load(shard_path, map_location=self._device, weights_only=True)

        # If it's already a dict with metadata, return it
        if isinstance(data, dict) and 'boundaries' in data:
            return data

        # Otherwise, return just embeddings (old format)
        return {'embeddings': data, 'boundaries': None, 'protein_ids': None}


def detect_and_create_loader(data_dir: Path, device: str = 'cpu') -> ShardDataLoader:
    """
    Automatically detect the data structure and return appropriate loader.

    Args:
        data_dir: Directory containing the data
        device: Device to load tensors to ('cpu' or 'cuda')

    Returns:
        Appropriate ShardDataLoader instance

    Raises:
        ValueError: If data structure cannot be determined
    """
    data_dir = Path(data_dir)

    # Check for flat shard files
    flat_shards = list(data_dir.glob("shard_*.pt"))
    if flat_shards:
        print(f"Detected flat file structure with {len(flat_shards)} shards")
        return FlatFileShardLoader(data_dir, device=device)

    # Check for nested shard directories
    nested_shards = [d for d in data_dir.glob("shard_*") if d.is_dir()]
    if nested_shards:
        # Try to find common activation file
        common_files = ["activations.pt", "embeddings.pt", "data.pt"]
        for filename in common_files:
            if (nested_shards[0] / filename).exists():
                print(f"Detected nested folder structure with {len(nested_shards)} shards, using {filename}")
                return NestedFolderShardLoader(data_dir, filename, device=device)

        # If no common filename found, check what files exist
        first_shard_files = list(nested_shards[0].glob("*.pt"))
        if first_shard_files:
            filename = first_shard_files[0].name
            print(f"Detected nested folder structure with {len(nested_shards)} shards, using {filename}")
            return NestedFolderShardLoader(data_dir, filename, device=device)

    # Check for layer subdirectory (common pattern)
    layer_dirs = [d for d in data_dir.glob("layer_*") if d.is_dir()]
    if layer_dirs:
        # Try the first layer directory
        return detect_and_create_loader(layer_dirs[0], device=device)

    raise ValueError(
        f"Could not detect data structure in {data_dir}. "
        f"Expected either shard_*.pt files or shard_*/{{activations,embeddings,data}}.pt structure"
    )