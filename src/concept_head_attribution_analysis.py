import argparse
import os
import json

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from nnsight import NNsight
from collections import defaultdict

from utils.args_utils import get_model_family, dict_to_object
from utils.model_utils import load_model_and_preprocess

from head_attribution.utils import (
    get_dataset_with_target_modality,
    run_head_attribution,
    get_config,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArguments():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, help="Path to JSON config file")
    pre_args, _ = pre_parser.parse_known_args()

    args = {}
    
    # If --config is provided, load arguments from JSON
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            args.update(json.load(f))
    else:
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--work_dir", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--version", type=int, required=True)
        parser.add_argument("--num_samples", type=int, default=-1)
        parser.add_argument("--batch_size", type=int, default=1)
        
        # Prompt-related arguments
        parser.add_argument("--caption_type", type=str, choices=["consistent", "inconsistent", "no_caption", "irrelevant", "text_only"], default="inconsistent")
        parser.add_argument("--order", type=str, choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"], default="icq")
        parser.add_argument(
            "--n_candidates", type=int, default=5,
            help="Number of answer candidates in the prompt."
        )
        parser.add_argument(
            "--is_explicit_helper", type=int, default=0,
            help="Whether to have explicit helper 'Ignoring the image or caption, what is the label in xxx' in the prompt."
        )
        parser.add_argument(
            "--is_assistant_prompt", type=int, default=1,
            help="'USER' and 'ASSISTANT' which specifies user and model in the prompt."
        )
        parser.add_argument("--use_pointers", type=int, default=1, help="Whether to use 'Caption: '/ 'Image: ' for caption and image.")

        args.update(vars(parser.parse_args()))

    args = dict_to_object(args)
    args.model_family = get_model_family(args)

    return args


def get_args():
    args = parseArguments()

    args.model_str = args.model_name
    args.model_family = get_model_family(args)
    
    args.output_dir_prefix = os.path.join(args.work_dir, "outputs", f"concept_head_attribution_ver{args.version:03d}", args.dataset, args.model_name)
    
    args.output_dir = os.path.join(args.work_dir, "outputs", f"concept_head_attribution_ver{args.version:03d}", args.dataset, args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def main(args):
    print("="*80)
    print("Concept-Based Attention Head Attribution Analysis")
    print("="*80)
    
    # Load model and processor
    print("\nLoading model and processor...")
    model, processor = load_model_and_preprocess(args)
    model = NNsight(model)
    
    # Get dataset
    print("\nLoading dataset...")
    dataset = get_dataset_with_target_modality(args, "image")
    
    # Determine number of samples
    if args.num_samples == -1:
        args.num_samples = len(dataset)
    else:
        args.num_samples = min(args.num_samples, len(dataset))
    
    print(f"Running attribution on {args.num_samples} samples")
    
    # Run head attribution
    print("\nRunning head attribution...")
    all_logit_diffs, all_total_logit_diffs, all_predictions_clean, sample_idx_list, concept_types = run_head_attribution(
        args, processor, model, dataset, target_modality="image", n_samples=args.num_samples
    )
    
    # Get model config
    N_LAYERS, N_HEADS, D_MODEL, D_HEADS = get_config(args, model)
    
    print(f"\nModel: {N_LAYERS} layers, {N_HEADS} heads per layer")
    print(f"Collected {len(concept_types)} samples with concept types")
    
    # Group attribution scores by concept
    print("\nGrouping attributions by concept type...")
    concept_to_samples = defaultdict(list)
    for idx, concept_type in enumerate(concept_types):
        if concept_type is not None:
            concept_to_samples[concept_type].append(idx)
    
    unique_concepts = sorted(concept_to_samples.keys())
    print(f"Found {len(unique_concepts)} unique concepts: {unique_concepts}")
    
    # Compute mean attribution per head per concept
    print("\nComputing mean attribution scores...")
    concept_head_scores = {}
    
    for concept in unique_concepts:
        sample_indices = concept_to_samples[concept]
        # Mean attribution across samples for this concept
        # Shape: (N_LAYERS, N_HEADS)
        concept_scores = np.mean(all_logit_diffs[sample_indices, :, :], axis=0)
        concept_head_scores[concept] = concept_scores
        print(f"  Concept {concept}: {len(sample_indices)} samples")
    
    # Create DataFrame with heads as rows and concepts as columns
    print("\nCreating output DataFrame...")
    rows = []
    row_labels = []
    
    for layer_idx in range(N_LAYERS):
        for head_idx in range(N_HEADS):
            row_label = f"L{layer_idx}_H{head_idx}"
            row_labels.append(row_label)
            
            row_data = {}
            for concept in unique_concepts:
                row_data[concept] = concept_head_scores[concept][layer_idx, head_idx]
            rows.append(row_data)
    
    df = pd.DataFrame(rows, index=row_labels)
    
    # Save to CSV
    output_csv = os.path.join(args.output_dir, "concept_head_attribution_scores.csv")
    df.to_csv(output_csv)
    print(f"\n✓ Saved attribution scores to: {output_csv}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"DataFrame shape: {df.shape}")
    print(f"Rows (heads): {df.shape[0]}")
    print(f"Columns (concepts): {df.shape[1]}")
    print(f"\nAttribution score range:")
    print(f"  Min: {df.min().min():.4f}")
    print(f"  Max: {df.max().max():.4f}")
    print(f"  Mean: {df.mean().mean():.4f}")
    
    # Find top heads for each concept
    print("\n" + "="*80)
    print("Top 5 Heads per Concept")
    print("="*80)
    for concept in unique_concepts:
        top_heads = df[concept].nlargest(5)
        print(f"\nConcept {concept}:")
        for head, score in top_heads.items():
            print(f"  {head}: {score:.4f}")
    
    # Find concepts each head is most sensitive to
    print("\n" + "="*80)
    print("Top Concept per Head (first 10 heads)")
    print("="*80)
    for i, head in enumerate(row_labels[:10]):
        top_concept = df.loc[head].idxmax()
        top_score = df.loc[head].max()
        print(f"{head}: Concept {top_concept} (score: {top_score:.4f})")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
    main(args)


