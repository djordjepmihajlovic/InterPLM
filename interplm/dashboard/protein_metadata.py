"""
Protein metadata abstraction layer for UniProt data.

This module provides a unified interface for accessing protein metadata
from UniProt/UniProtKB sources.

Note:
    To add support for another protein metadata source, implement a new class that inherits from
    BaseProteinMetadata and provides the required methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd


class BaseProteinMetadata(ABC):
    """Abstract base class for protein metadata access."""

    @abstractmethod
    def get_protein_sequence(self, protein_id: str) -> str:
        """Get the protein sequence for a given protein ID."""
        pass

    @abstractmethod
    def get_protein_name(self, protein_id: str) -> str:
        """Get the protein name for a given protein ID."""
        pass

    @abstractmethod
    def get_embedding_shard_info(self, protein_id: str) -> str:
        """Get embedding shard information for a given protein ID.

        Returns:
            str: shard filename
        """
        pass

    @abstractmethod
    def get_protein_metadata(self, protein_id: str) -> Dict:
        """Get all available metadata for a given protein ID."""
        pass

    @abstractmethod
    def list_available_proteins(self) -> List[str]:
        """Get list of all available protein IDs."""
        pass

    @property
    @abstractmethod
    def metadata_type(self) -> str:
        """Return the type of metadata (e.g., 'uniprot', 'uniclust')."""
        pass


class UniProtMetadata(BaseProteinMetadata):
    """UniProt-based protein metadata (refactored from original ProteinMetadata)."""

    def __init__(
        self,
        metadata_path: Path,
        uniprot_id_col: str = "Entry",
        sequence_col: str = "Sequence",
        protein_name_col: str = "Protein names",
        embedding_shard_col: str = "shard",
    ):
        self._metadata_path = Path(metadata_path)
        self.uniprot_id_col = uniprot_id_col
        self.sequence_col = sequence_col
        self.protein_name_col = protein_name_col
        self.embedding_shard_col = embedding_shard_col
        self._data = None

    @property
    def metadata_type(self) -> str:
        return "uniprot"

    @property
    def data(self) -> pd.DataFrame:
        """Lazy load the metadata DataFrame."""
        if self._data is None:
            # Handle different file formats
            suffix = self._metadata_path.suffix
            if suffix == ".gz" and self._metadata_path.with_suffix("").suffix == ".tsv":
                df = pd.read_csv(self._metadata_path, sep="\t")
            elif suffix == ".tsv":
                df = pd.read_csv(self._metadata_path, sep="\t")
            elif suffix == ".csv":
                df = pd.read_csv(self._metadata_path)
            else:
                raise ValueError("Metadata file must end in .tsv.gz, .tsv, or .csv")

            # Convert uniprot_id_col values to uppercase for consistency
            df[self.uniprot_id_col] = df[self.uniprot_id_col].str.upper()
            self._data = df.set_index(self.uniprot_id_col)
            
        return self._data

    def get_protein_name(self, protein_id: str) -> str:
        """Get the protein name for a given UniProt ID."""
        protein_id = protein_id.upper()
        return self.data.loc[protein_id, self.protein_name_col]

    def get_protein_sequence(self, protein_id: str) -> str:
        """Get the protein sequence for a given UniProt ID."""
        protein_id = protein_id.upper()
        return self.data.loc[protein_id, self.sequence_col]

    def get_embedding_shard_info(self, protein_id: str) -> str:
        """Get the embedding shard filename for a given UniProt ID."""
        protein_id = protein_id.upper()
        return self.data.loc[protein_id, self.embedding_shard_col]

    def get_protein_metadata(self, protein_id: str) -> Dict:
        """Get all metadata for a given UniProt ID."""
        protein_id = protein_id.upper()
        return self.data.loc[protein_id].to_dict()

    def list_available_proteins(self) -> List[str]:
        """Get list of all available UniProt IDs."""
        return self.data.index.tolist()

    @classmethod
    def load_from_dict(cls, metadata_dict: Dict) -> "UniProtMetadata":
        """Load UniProt metadata from a dictionary (for backward compatibility)."""
        return cls(
            metadata_path=metadata_dict["metadata_path"],
            uniprot_id_col=metadata_dict.get("uniprot_id_col", "Entry"),
            sequence_col=metadata_dict.get("sequence_col", "Sequence"),
            protein_name_col=metadata_dict.get("protein_name_col", "Protein names"),
            embedding_shard_col=metadata_dict.get("embedding_shard_col", "shard"),
        )



def create_protein_metadata(
    metadata_type: str,
    **kwargs
) -> BaseProteinMetadata:
    """Factory function to create appropriate protein metadata instance."""

    if metadata_type.lower() in ['uniprot', 'uniprotkb']:
        return UniProtMetadata(**kwargs)
    else:
        raise ValueError(f"Unknown metadata type: {metadata_type}")