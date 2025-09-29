#!/usr/bin/env python
"""Create a dashboard for visualizing trained SAE features.

This script creates a dashboard cache from a trained SAE model, preparing it
for interactive visualization in the InterPLM dashboard.

Run this after training an SAE (e.g., with train_basic_sae.py) and collecting
feature activations (with collect_feature_activations.py).
"""

import os
from pathlib import Path
from interplm.dashboard.dashboard_cache import DashboardCache
from interplm.dashboard.protein_metadata import UniProtMetadata
from interplm.sae.dictionary import ReLUSAE

def main():
    # Get INTERPLM_DATA from environment or use default
    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")

    # Get LAYER from environment or use default
    LAYER = os.environ.get("LAYER", "3")

    # Paths from walkthrough
    SAE_PATH = Path("models") / "walkthrough_model" / f"layer_{LAYER}" / "ae.pt"
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "analysis_embeddings" / "esm2_8m" / f"layer_{LAYER}"

    # Use the original source metadata
    # Dashboard will recompute embeddings on-the-fly when visualizing proteins
    METADATA_PATH = Path("data") / "uniprotkb" / "swissprot_dense_annot_1k_subset.tsv.gz"

    # Check if normalized version exists, use it instead
    SAE_NORMALIZED_PATH = SAE_PATH.parent / "ae_normalized.pt"
    if SAE_NORMALIZED_PATH.exists():
        SAE_PATH = SAE_NORMALIZED_PATH

    # Dashboard configuration
    DASHBOARD_NAME = "walkthrough"
    CACHE_DIR = Path(INTERPLM_DATA) / "dashboard_cache" / DASHBOARD_NAME
    LAYER_NAME = f"layer_{LAYER}"

    print("=" * 60)
    print("Creating Dashboard for Walkthrough SAE")
    print("=" * 60)
    print(f"SAE model: {SAE_PATH}")
    print(f"Embeddings: {EMBEDDINGS_DIR}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Check paths exist
    if not SAE_PATH.exists():
        raise FileNotFoundError(f"SAE model not found at {SAE_PATH}. Train an SAE first (e.g., with train_basic_sae.py)!")
    if not EMBEDDINGS_DIR.exists():
        raise FileNotFoundError(f"Embeddings not found at {EMBEDDINGS_DIR}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

    # Create protein metadata from original source file
    print("Loading protein metadata...")
    protein_metadata = UniProtMetadata(
        metadata_path=METADATA_PATH,
        uniprot_id_col="Entry",
        protein_name_col="Protein names",
        sequence_col="Sequence"
    )

    # Create dashboard cache
    print(f"Creating dashboard cache at: {CACHE_DIR}")
    cache = DashboardCache.create_dashboard_cache(
        cache_dir=CACHE_DIR,
        model_name="esm",
        model_type="esm2_t6_8M_UR50D",
        protein_metadata=protein_metadata,
        overwrite=True  # Overwrite if exists
    )

    # Check for concept enrichment results (optional)
    CONCEPT_ENRICHMENT_PATH = Path("results") / "test_counts" / "heldout_top_pairings.csv"
    if not CONCEPT_ENRICHMENT_PATH.exists():
        # Try alternate location
        CONCEPT_ENRICHMENT_PATH = Path("results") / "walkthrough_curated_valid_counts" / "concept_f1_scores.csv"
        if not CONCEPT_ENRICHMENT_PATH.exists():
            print("\nNote: Concept enrichment results not found")
            print("  To add concept enrichment, run Step 3 from the README")
            CONCEPT_ENRICHMENT_PATH = None

    # Add layer with SAE
    print(f"Adding layer: {LAYER_NAME}")
    cache.add_layer(
        layer_name=LAYER_NAME,
        sae_cls=ReLUSAE,
        sae_path=SAE_PATH,
        feature_stats_dir=SAE_PATH.parent,  # Pre-computed stats from collect_feature_activations.py
        aa_embeds_dir=EMBEDDINGS_DIR,  # For random feature sampling
        shards_to_search=[0, 1, 2, 3, 4, 5, 6, 7],  # All 8 shards
        concept_enrichment_path=CONCEPT_ENRICHMENT_PATH,  # Optional concept analysis
        overwrite=True
    )

    print()
    print("=" * 60)
    print("✅ Dashboard created successfully!")
    print("=" * 60)
    print()
    print("To view the dashboard, run:")
    print(f"  streamlit run interplm/dashboard/app.py -- --cache_dir {CACHE_DIR}")
    print()

if __name__ == "__main__":
    main()
