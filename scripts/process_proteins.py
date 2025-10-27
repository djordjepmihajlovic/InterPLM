#!/usr/bin/env python
"""Process a batch of protein IDs in parallel."""

import yaml
import argparse
from pathlib import Path
import heapq
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from interplm.sae.dictionary import Dictionary
from interplm.sae.inference import get_sae_feats_in_batches, split_up_feature_list
from interplm.utils import get_device
from interplm.data_processing.embedding_loader import load_shard_embeddings
from transformers import AutoTokenizer, EsmForProteinFolding
from interplm.analysis.per_protein_tracking import PerProteinActivationTracker
# Import your existing functions
# from your_module import get_protein_sequence, generate_pdb
def load_esmfold_model(device: str):
    """Load and configure ESMFold model for structure prediction."""
    print("Loading ESMFold model for structure prediction...")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1",
        clean_up_tokenization_spaces=True
    )
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model.float()
    # Always use CPU for ESMFold to avoid memory issues on limited GPU memory
    # model = model.to("cpu")
    model.eval()
    print("✓ ESMFold model loaded on device:", device)
    return tokenizer, model


def get_protein_sequence(metadata_dir: Path, num_shard: int, protein_id: str) -> Optional[str]:
    """Retrieve protein sequence from metadata."""
    import pandas as pd
    for shard in range(num_shard):
        # Try to load protein data from the shard
        protein_data_path = metadata_dir / f"shard_{shard}" / "protein_data.tsv"
        if protein_data_path.exists():
            df = pd.read_csv(protein_data_path, sep="\t")
            # Convert protein IDs to uppercase to match
            df['Entry'] = df['Entry'].str.upper()
            protein_id = protein_id.upper()
            
            if protein_id in df['Entry'].values:
                # print(shard)
                row = df[df['Entry'] == protein_id].iloc[0]
                # print(row['Sequence']) if 'Sequence' in df.columns else print("none")
                # print(row)
                return row['Sequence'] if 'Sequence' in df.columns else None
    
    return None

def generate_pdb(sequence: str, tokenizer, model, device: str) -> str:
    """Generate PDB structure from protein sequence using ESMFold."""
    with torch.no_grad():
        pdb_string = model.infer_pdb(sequence)
    return pdb_string

def proteins_to_compute(top_proteins_file: Path) -> list:
    """Load unique protein IDs from top proteins per feature YAML file."""
    with open(top_proteins_file, 'r') as f:
        per_feature_max_prots = yaml.safe_load(f)
    proteins_to_compute = set()
    for feature in per_feature_max_prots:
        for protein in per_feature_max_prots[feature]:
            proteins_to_compute.add(protein)
    return list(proteins_to_compute)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_proteins', type=Path, required=True,
                       help='YAML file feature_id: proteinids list')
    parser.add_argument('--num_shards', type=int, required=True,
                       help='number of shards used when creating protein annotations')
    parser.add_argument('--metadata_dir', type=Path, required=True,
                       help='metadata containing protein sequences')
    parser.add_argument('--output_file', type=Path, required=True,
                       help='Output YAML file for this batch')
    parser.add_argument('--start_idx', type=int, required=True,
                       help='Start index (inclusive)')
    parser.add_argument('--end_idx', type=int, required=True,
                       help='End index (exclusive)')
    args = parser.parse_args()
    
    # Load all protein IDs
    all_ids = proteins_to_compute(args.top_proteins)
    with open(proteins_to_compute(args.top_proteins)) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    
    # Get this batch's IDs
    batch_ids = all_ids[args.start_idx:args.end_idx]
    
    print(f"Processing {len(batch_ids)} proteins (indices {args.start_idx}-{args.end_idx-1})")
    print()
    device = get_device()
    esmfold_tokenizer, esmfold_model = load_esmfold_model(device)
    print()
    # Process each protein
    results = {}
    for i, protein_id in enumerate(batch_ids):
        print(f"[{i+1}/{len(batch_ids)}] Processing {protein_id}...")
        
        # Get sequence
        sequence = get_protein_sequence(args.metadata_dir, args.num_shards, protein_id)
        print(f"  Sequence length: {len(sequence)}")
        
        # Generate PDB (this is slow)
        pdb = generate_pdb(sequence, esmfold_tokenizer, esmfold_model, device)
        print(f"  ✓ PDB generated")
        
        # Store both
        results[protein_id] = {
            'sequence': sequence,
            'pdb': pdb
        }
    
    # Save batch results
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        yaml.dump(results, f)
    
    print()
    print(f"✓ Batch complete! Saved {len(results)} proteins to {args.output_file}")

if __name__ == '__main__':
    main()