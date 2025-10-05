#!/usr/bin/env python
"""Analyze SAE features: find max activating proteins and compute statistics.

This script analyzes a trained SAE by:
1. Finding the top activating proteins for each feature
2. Computing feature statistics (frequency, activation percentages)
3. Saving results for later use (e.g., in dashboard creation)

Run this after training an SAE and before creating a dashboard.
"""

from pathlib import Path
from typing import List, Optional
import yaml
import torch

from interplm.sae.inference import load_sae, load_sae_from_hf
from interplm.analysis.per_protein_tracking_struc import find_max_examples_with_structures, load_esmfold_model
from interplm.utils import get_device


def collect_feature_activations(
    sae_dir: Path,
    embeddings_dir: Path,
    metadata_dir: Path,
    output_dir: Optional[Path] = None,
    shards: Optional[List[int]] = None,
    shard_range: Optional[List[int]] = None,
    activation_threshold: float = 0.05,
    save_protein_structure: bool = True,
):
    """
    Analyze SAE features and find max activating proteins with optional structure prediction.

    Args:
        sae_dir: Directory containing trained SAE model
        embeddings_dir: Directory containing embeddings
        metadata_dir: Directory containing protein metadata
        output_dir: Output directory for results (default: same as sae_dir)
        shards: Shard indices to search
        shard_range: Shard range [start, end] (inclusive) to search
        activation_threshold: Minimum activation value to count as 'activated'
        save_protein_structure: If True, generate 3D structures for max activating proteins
    """
    # Handle shard arguments
    if shards is not None and shard_range is not None:
        raise ValueError("Cannot specify both shards and shard_range")
    elif shard_range is not None:
        shards = list(range(shard_range[0], shard_range[1] + 1))
    elif shards is None:
        shards = [0]

    # Set output directory
    if output_dir is None:
        output_dir = sae_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAE Feature Analysis")
    print("=" * 70)
    print(f"SAE directory: {sae_dir}")
    print(f"Embeddings: {embeddings_dir}")
    print(f"Metadata: {metadata_dir}")
    print(f"Output: {output_dir}")
    print(f"Analyzing shards: {shards}")
    print(f"Generate 3D structures: {save_protein_structure}")
    print()

    # Load SAE
    device = get_device()
    print(f"Loading SAE on device: {device}")
    sae = load_sae_from_hf(plm_model="esm2-8m", plm_layer=4)
    print(f"SAE loaded: {sae.__class__.__name__} with {sae.dict_size} features")
    print()

    # Load ESMFold model if needed
    esmfold_tokenizer = None
    esmfold_model = None
    if save_protein_structure:
        esmfold_tokenizer, esmfold_model = load_esmfold_model(device)
        print()

    # Find max activating proteins
    print("Finding max activating proteins for each feature...")
    per_protein_tracker = find_max_examples_with_structures(
        sae=sae,
        aa_embeds_dir=embeddings_dir,
        aa_metadata_dir=metadata_dir,
        shards_to_search=shards,
        activation_threshold=activation_threshold,
        save_protein_structure=save_protein_structure,
        esmfold_model=esmfold_model,
        esmfold_tokenizer=esmfold_tokenizer,
    )
    print("✓ Analysis complete")
    print()

    # Save results
    print("Saving results...")

    # Save standard results (as before)
    max_activations = per_protein_tracker["max_activation_per_feature"]
    torch.save(torch.tensor(max_activations), output_dir / "max_activations_per_feature.pt")
    print(f"✓ Max activations saved")

    per_feature_statistics = {
        "Per_prot_frequency_of_any_activation": per_protein_tracker["pct_proteins_with_activation"],
        "Per_prot_pct_activated_when_present": per_protein_tracker["avg_pct_activated_when_present"],
    }
    with open(output_dir / "Per_feature_statistics.yaml", "w") as f:
        yaml.dump(per_feature_statistics, f)
    print(f"✓ Feature statistics saved")

    with open(output_dir / "Per_feature_max_examples.yaml", "w") as f:
        yaml.dump(per_protein_tracker["max"], f)
    print(f"✓ Max examples saved")

    with open(output_dir / "Per_feature_quantile_examples.yaml", "w") as f:
        yaml.dump(per_protein_tracker["lower_quantile"], f)
    print(f"✓ Quantile examples saved")

    # Save structures if generated
    if save_protein_structure and "max_with_structures" in per_protein_tracker:
        output_path = output_dir / "Per_feature_max_examples_with_structures.yaml"
        with open(output_path, "w") as f:
            yaml.dump(per_protein_tracker["max_with_structures"], f)
        print(f"✓ Protein structures saved to: {output_path}")
        
        # Also save individual PDB files for easier access
        pdb_dir = output_dir / "pdb_structures"
        pdb_dir.mkdir(exist_ok=True)
        
        saved_proteins = set()
        for feature_data in per_protein_tracker["max_with_structures"].values():
            for protein_entry in feature_data:
                protein_id = protein_entry['protein_id']
                if 'pdb' in protein_entry and protein_entry['pdb'] and protein_id not in saved_proteins:
                    pdb_path = pdb_dir / f"{protein_id}.pdb"
                    with open(pdb_path, 'w') as f:
                        f.write(protein_entry['pdb'])
                    saved_proteins.add(protein_id)
        
        if saved_proteins:
            print(f"✓ Individual PDB files saved to: {pdb_dir} ({len(saved_proteins)} structures)")

    print()
    print("=" * 70)
    print("✅ Feature analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    from tap import tapify
    import pandas as pd  # Import at module level for helper functions
    tapify(collect_feature_activations)