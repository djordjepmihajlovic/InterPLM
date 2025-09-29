#!/usr/bin/env python
"""
Examples of training different SAE architectures with custom configurations.

This script demonstrates how to train various SAE architectures (ReLU, Top-K,
Jump ReLU, Batch Top-K) with custom hyperparameters, W&B logging, checkpoint
resumption, and fidelity evaluation.

Usage:
    # Train standard ReLU SAE
    python examples/train_multiple_sae_architectures.py --architecture standard

    # Train Top-K SAE (fixed sparsity)
    python examples/train_multiple_sae_architectures.py --architecture topk

    # Train with fidelity evaluation during training
    python examples/train_multiple_sae_architectures.py --architecture fidelity

    # Resume from checkpoint
    python examples/train_multiple_sae_architectures.py --architecture resume

For a simple example with hardcoded parameters, see train_basic_sae.py instead.
"""

import os
from pathlib import Path

from interplm.train.configs import (
    TrainingRunConfig,
    DataloaderConfig,
    ReLUTrainerConfig,
    TopKTrainerConfig,
    JumpReLUTrainerConfig,
    BatchTopKTrainerConfig,
    EvaluationConfig,
    WandbConfig,
    CheckpointConfig
)
from interplm.train.fidelity import ESMFidelityConfig
from interplm.train.training_run import SAETrainingRun


def train_standard_relu():
    """Example with Standard ReLU SAE (most common)."""

    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "esm_embds" / "layer_3"
    SAVE_DIR = Path("models") / "standard_relu_sae"

    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=256,  # Larger batch size for better gradient estimates
    )

    trainer_cfg = ReLUTrainerConfig(
        activation_dim=320,  # ESM2-8M layer dimension
        dictionary_size=16384,  # Large dictionary (51x expansion)
        lr=2e-4,
        l1_penalty=5e-4,
        warmup_steps=5000,
        decay_start=40000,
        steps=50000,
        normalize_to_sqrt_d=False,  # Can enable for normalized training
        resample_steps=25000,  # Resample dead neurons periodically
    )

    eval_cfg = EvaluationConfig(
        eval_embd_dir=Path(INTERPLM_DATA) / "esm_embds" / "layer_3_eval",  # Optional eval set
        eval_steps=5000,  # Evaluate every 5000 steps
    )

    wandb_cfg = WandbConfig(
        use_wandb=True,  # Enable W&B logging
        wandb_entity="your-entity",  # Replace with your wandb entity
        wandb_project="interplm-sae",
        wandb_name="standard-relu-16k",
    )

    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=5000,
        max_ckpts_to_keep=3,
    )

    config = TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )

    # Train the model
    training_run = SAETrainingRun.from_config(config)
    training_run.run()


def train_top_k():
    """Example with Top-K SAE (fixed sparsity level)."""

    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "esm_embds" / "layer_3"
    SAVE_DIR = Path("models") / "top_k_sae"

    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=256,
    )

    trainer_cfg = TopKTrainerConfig(
        activation_dim=320,
        dictionary_size=16384,
        k=32,  # Exactly 32 features active per example
        lr=2e-4,
        warmup_steps=5000,
        decay_start=40000,
        steps=50000,
        normalize_to_sqrt_d=False,
        auxk_alpha=1/32,  # Auxiliary loss weight
    )

    eval_cfg = EvaluationConfig(eval_embd_dir=None)
    wandb_cfg = WandbConfig(
        use_wandb=True,
        wandb_entity="your-entity",  # Replace with your wandb entity
        wandb_project="interplm-sae",
        wandb_name="top-k-32",
    )
    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=5000,
        max_ckpts_to_keep=3,
    )

    config = TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )

    training_run = SAETrainingRun.from_config(config)
    training_run.run()


def train_jump_relu():
    """Example with Jump ReLU SAE (learnable thresholds)."""

    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "esm_embds" / "layer_3"
    SAVE_DIR = Path("models") / "jump_relu_sae"

    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=256,
    )

    trainer_cfg = JumpReLUTrainerConfig(
        activation_dim=320,
        dictionary_size=16384,
        lr=2e-4,
        l0_penalty=1e-3,  # L0 penalty instead of L1
        warmup_steps=5000,
        decay_start=40000,
        steps=50000,
        normalize_to_sqrt_d=False,
        bandwidth=0.001,  # Jump ReLU bandwidth parameter
        l1_penalty=0.01,  # Additional L1 penalty
    )

    eval_cfg = EvaluationConfig(eval_embd_dir=None)
    wandb_cfg = WandbConfig(
        use_wandb=True,
        wandb_entity="your-entity",  # Replace with your wandb entity
        wandb_project="interplm-sae",
        wandb_name="jump-relu-16k",
    )
    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=5000,
        max_ckpts_to_keep=3,
    )

    config = TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )

    training_run = SAETrainingRun.from_config(config)
    training_run.run()


def train_batch_top_k():
    """Example with Batch Top-K SAE (batch-level sparsity constraint)."""

    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "esm_embds" / "layer_3"
    SAVE_DIR = Path("models") / "batch_top_k_sae"

    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=256,  # Batch size matters for batch top-k
    )

    trainer_cfg = BatchTopKTrainerConfig(
        activation_dim=320,
        dictionary_size=16384,
        lr=2e-4,
        total_active_features=512,  # Total features active across batch
        warmup_steps=5000,
        decay_start=40000,
        steps=50000,
        normalize_to_sqrt_d=False,
    )

    eval_cfg = EvaluationConfig(eval_embd_dir=None)
    wandb_cfg = WandbConfig(
        use_wandb=True,
        wandb_entity="your-entity",  # Replace with your wandb entity
        wandb_project="interplm-sae",
        wandb_name="batch-top-k-512",
    )
    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=5000,
        max_ckpts_to_keep=3,
    )

    config = TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )

    training_run = SAETrainingRun.from_config(config)
    training_run.run()


def train_with_fidelity():
    """Example with downstream task fidelity evaluation during training.

    This demonstrates using ESMFidelityConfig to track how well the SAE preserves
    ESM's ability to predict masked tokens. This is slower than basic reconstruction
    evaluation but provides a more comprehensive measure of model quality.

    NOTE: Fidelity evaluation requires loading ESM and running forward passes,
    so it's significantly slower. Use sparingly (e.g., eval_steps=10000) or only
    for final model validation.
    """

    INTERPLM_DATA = os.environ.get("INTERPLM_DATA", "data")
    EMBEDDINGS_DIR = Path(INTERPLM_DATA) / "training_embeddings" / "esm2_8m" / "layer_4"
    SAVE_DIR = Path("models") / "fidelity_example_sae"
    LAYER = 4

    # First, we need to create a text file with evaluation sequences
    # This should be a subset of proteins from your held-out data
    eval_seq_file = Path(INTERPLM_DATA) / "eval_sequences.txt"

    # For this example, we'll create it from the first FASTA shard if it doesn't exist
    if not eval_seq_file.exists():
        print(f"Creating eval sequences file at {eval_seq_file}")
        from Bio import SeqIO
        fasta_file = Path(INTERPLM_DATA) / "uniprot_shards" / "shard_0.fasta"

        if not fasta_file.exists():
            raise FileNotFoundError(
                f"FASTA file not found: {fasta_file}\n"
                "Run Step 1 of the walkthrough to generate FASTA files first."
            )

        # Extract first 50 sequences for fidelity evaluation
        eval_seq_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fasta_file) as f_in, open(eval_seq_file, 'w') as f_out:
            for i, record in enumerate(SeqIO.parse(f_in, "fasta")):
                if i >= 50:  # Keep it small - fidelity is slow
                    break
                f_out.write(str(record.seq) + "\n")

        print(f"Created {eval_seq_file} with 50 sequences")

    dataloader_cfg = DataloaderConfig(
        plm_embd_dir=EMBEDDINGS_DIR,
        batch_size=256,
    )

    trainer_cfg = ReLUTrainerConfig(
        activation_dim=320,  # ESM2-8M layer dimension
        dictionary_size=1280,  # 4x expansion (smaller for faster testing)
        lr=2e-4,
        l1_penalty=5e-4,
        warmup_steps=1000,
        decay_start=8000,
        steps=10000,  # Short training for example
        normalize_to_sqrt_d=False,
    )

    # Use ESMFidelityConfig instead of EvaluationConfig
    eval_cfg = ESMFidelityConfig(
        eval_seq_path=eval_seq_file,  # Text file with sequences (one per line)
        model_name="esm2_t6_8M_UR50D",  # ESM model name
        layer_idx=LAYER,  # Which layer we're training on
        eval_steps=5000,  # Evaluate every 5000 steps (keep infrequent - it's slow!)
        eval_batch_size=8,  # Batch size for fidelity evaluation
    )

    wandb_cfg = WandbConfig(
        use_wandb=False,  # Disable for example
    )

    checkpoint_cfg = CheckpointConfig(
        save_dir=SAVE_DIR,
        save_steps=5000,
        max_ckpts_to_keep=1,
    )

    config = TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )

    # Train the model with fidelity evaluation
    print("\n" + "="*70)
    print("Training SAE with Downstream Task Fidelity Evaluation")
    print("="*70)
    print("This will track how well the SAE preserves ESM's ability to")
    print("predict masked tokens during training.")
    print("="*70 + "\n")

    training_run = SAETrainingRun.from_config(config)
    training_run.run()

    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Model saved to: {SAVE_DIR}")
    print("="*70)


def resume_from_checkpoint():
    """Example of resuming training from a checkpoint."""

    # Resume from a specific checkpoint
    model_dir = Path("models/standard_relu_sae")

    # Resume from latest checkpoint
    training_run = SAETrainingRun.from_checkpoint(
        model_dir=model_dir,
        checkpoint_number=None,  # Use latest
        use_wandb=True,  # Re-enable W&B
    )

    # Continue training
    training_run.run()


def train_sae_example(
    architecture: str = "standard",
):
    """
    Train different SAE architectures.

    Args:
        architecture: SAE architecture to train. Options: standard, topk, jump, batch_topk, fidelity, resume
    """
    if architecture == "standard":
        train_standard_relu()
    elif architecture == "topk":
        train_top_k()
    elif architecture == "jump":
        train_jump_relu()
    elif architecture == "batch_topk":
        train_batch_top_k()
    elif architecture == "fidelity":
        train_with_fidelity()
    elif architecture == "resume":
        resume_from_checkpoint()
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from: standard, topk, jump, batch_topk, fidelity, resume")


if __name__ == "__main__":
    from tap import tapify
    tapify(train_sae_example)
