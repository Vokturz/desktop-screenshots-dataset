#!/usr/bin/env python3
"""
Dataset analysis script for validated sourceforge app screenshots.

This script processes screenshots from the "Vokturz/sourceforge-app-screenshots" dataset
that have been validated as valid (is_valid==True) in the validation results.
For each valid screenshot, it performs:
1. Image analysis using VLM to extract description, keywords, and category

Results are saved as JSONL with complete analysis data.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

from src.utils.vlm_utils import do_image_analysis

load_dotenv()

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false


def load_validation_results(validation_file: Path) -> dict[str, bool]:
    """Load validation results from JSONL file.

    Args:
        validation_file: Path to the validation results JSONL file

    Returns:
        Dictionary mapping slug to validity status
    """
    validation_results = {}

    if not validation_file.exists():
        print(f"Warning: Validation file not found: {validation_file}")
        return validation_results

    try:
        with open(validation_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    result = json.loads(line)
                    if "slug" in result and "is_valid" in result:
                        validation_results[result["slug"]] = result["is_valid"]
    except Exception as e:
        print(f"Error reading validation results: {e}")

    return validation_results


def analyze_single_row(
    row: dict[str, Any], verbose: bool = False
) -> dict[str, Any] | None:
    """Analyze a single dataset row using VLM for image analysis.

    Args:
        row: Dataset row containing image, slug, title, description
        verbose: Whether to print VLM responses

    Returns:
        dict with complete analysis results including slug, image_analysis
    """
    try:
        slug = row["slug"]

        if verbose:
            print(f"Processing {slug}...")

        # Image analysis
        image_analysis = do_image_analysis(
            row,
            print_assistant_message=verbose,
            custom_system_prompt=None,
            from_src=True,
        )

        result = {
            "slug": slug,
            "image_description": image_analysis.description,
            "keywords": image_analysis.keywords,
            "category": image_analysis.category,
        }

        if verbose:
            print(f"✓ {slug}: Analysis complete")
            print(f"  Category: {image_analysis.category}")
            print(f"  Keywords: {', '.join(image_analysis.keywords)}")

        return result

    except Exception as e:
        print(f"✗ Error analyzing {row.get('slug', 'unknown')}: {str(e)}")
        return None


def load_existing_analyzed_slugs(output_path: Path) -> set[str]:
    """Load existing slugs from analysis results file.

    Args:
        output_path: Path to the analysis results JSONL file

    Returns:
        Set of slugs that have already been analyzed
    """
    existing_slugs = set()

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        result = json.loads(line)
                        if "slug" in result:
                            existing_slugs.add(result["slug"])
        except Exception as e:
            print(f"Warning: Error reading existing analysis results file: {e}")

    return existing_slugs


def analyze_validated_dataset(
    dataset_name: str = "Vokturz/sourceforge-app-screenshots",
    validation_file: str = "data/validation_results.jsonl",
    output_file: str | None = None,
    max_workers: int = 4,
    max_samples: int | None = None,
    verbose: bool = False,
) -> None:
    """Analyze validated dataset using parallel VLM calls for image analysis.

    Args:
        dataset_name: HuggingFace dataset name
        validation_file: Path to validation results JSONL file
        output_file: Output JSONL file path
        max_workers: Number of parallel workers
        max_samples: Limit number of samples to analyze (None for all valid samples)
        verbose: Whether to print detailed output
    """
    print(f"Loading dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name)

        # Use the train split or the first available split
        if "train" in dataset:
            data = dataset["train"]
        else:
            split_name: str = list(dataset.keys())[0]  # pyright: ignore
            data = dataset[split_name]
            print(f"Using split: {split_name}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load validation results
    validation_path = Path(__file__).parent.parent / validation_file
    validation_results = load_validation_results(validation_path)

    if not validation_results:
        print("No validation results found. Please run validation first.")
        return

    print(f"Loaded {len(validation_results)} validation results")
    valid_slugs = {slug for slug, is_valid in validation_results.items() if is_valid}
    print(f"Found {len(valid_slugs)} valid screenshots to analyze")

    # Filter dataset to only include valid screenshots
    valid_indices: list[int] = []
    for idx, row in enumerate(data):  # pyright: ignore
        if row["slug"] in valid_slugs:
            valid_indices.append(idx)

    if not valid_indices:
        print("No valid screenshots found in dataset.")
        return

    print(f"Found {len(valid_indices)} valid items in dataset")

    # Update data to use filtered indices
    data = data.select(valid_indices)  # pyright: ignore

    # Limit samples if requested
    if max_samples and max_samples < len(valid_indices):
        data = data.select(range(max_samples))
        print(f"Limited to {max_samples} samples")

    # Setup output file
    if output_file is None:
        output_file = "data/analysis_results.jsonl"

    # Ensure output directory exists
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing analyzed slugs to skip already processed items
    existing_slugs = load_existing_analyzed_slugs(output_path)
    if existing_slugs:
        print(f"Found {len(existing_slugs)} already analyzed slugs, skipping them")

    # Filter out already processed items
    filtered_indices: list[int] = []
    for idx, row in enumerate(data):  # pyright: ignore
        if row["slug"] not in existing_slugs:  # pyright: ignore
            filtered_indices.append(idx)

    original_size = len(data)  # pyright: ignore
    if len(filtered_indices) < original_size:
        print(
            f"Filtered {original_size - len(filtered_indices)} already processed items"
        )
        print(f"Processing {len(filtered_indices)} remaining items")

    # Update data to use filtered indices
    if filtered_indices:
        data = data.select(filtered_indices)
    else:
        data = []
    dataset_size = len(filtered_indices)

    # Skip analysis if no new data to process
    if not data:
        print("No new items to analyze. All valid items have been processed already.")
        return

    # Analyze in parallel and append results immediately
    results: list[dict[str, Any]] = []
    print(f"Appending results to: {output_path}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all analysis tasks
        future_to_idx = {
            executor.submit(analyze_single_row, row, verbose): idx  # pyright: ignore
            for idx, row in enumerate(data)  # pyright: ignore
        }

        # Process completed tasks with progress bar and append immediately
        with tqdm(total=dataset_size, desc="Analyzing") as pbar:
            with open(output_path, "a", encoding="utf-8") as f:
                for future in as_completed(future_to_idx):
                    try:
                        result = future.result()

                        # Skip if result is None (error occurred)
                        if result is None:
                            pbar.update(1)
                            continue

                        results.append(result)

                        # Append result immediately to file
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()  # Ensure data is written immediately

                        pbar.update(1)

                        # Update progress description with category stats
                        categories = {}
                        for r in results:
                            if "category" in r and r["category"]:
                                cat = r["category"]
                                categories[cat] = categories.get(cat, 0) + 1

                        top_categories = sorted(
                            categories.items(),  # pyright: ignore
                            key=lambda x: x[1],  # pyright: ignore
                            reverse=True,
                        )[:3]
                        category_str = ", ".join(
                            [f"{cat}:{count}" for cat, count in top_categories]
                        )

                        pbar.set_postfix_str(f"Categories: {category_str}")

                    except Exception as e:
                        idx = future_to_idx[future]
                        slug = data[idx].get("slug", f"index_{idx}")
                        print(f"Task failed for {slug}: {e}")
                        pbar.update(1)

    # Print summary statistics for this run
    print("\nAnalysis Summary (this run):")
    print(f"  Processed samples: {len(results)}")

    if results:
        # Category distribution
        categories = {}
        for result in results:
            if "category" in result and result["category"]:
                cat = result["category"]
                categories[cat] = categories.get(cat, 0) + 1

        if categories:
            print("  Category distribution:")
            for cat, count in sorted(
                categories.items(),  # pyright: ignore
                key=lambda x: x[1],  # pyright: ignore
                reverse=True,
            ):
                percentage = count / len(results) * 100
                print(f"    {cat}: {count} ({percentage:.1f}%)")

    print(f"  Results appended to: {output_path}")

    # Show total statistics if there were existing results
    if existing_slugs:
        total_existing = len(existing_slugs)
        total_processed = total_existing + len(results)
        print("\nTotal analysis progress:")
        print(f"  Previously analyzed: {total_existing}")
        print(f"  Newly analyzed: {len(results)}")
        print(f"  Total analyzed: {total_processed}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze validated sourceforge app screenshots using VLM for image analysis"
    )

    parser.add_argument(
        "--dataset",
        default="Vokturz/sourceforge-app-screenshots",
        help="HuggingFace dataset name (default: Vokturz/sourceforge-app-screenshots)",
    )

    parser.add_argument(
        "--validation-file",
        default="data/validation_results.jsonl",
        help="Path to validation results JSONL file (default: data/validation_results.jsonl)",
    )

    parser.add_argument(
        "--output",
        help="Output JSONL file path (default: data/analysis_results.jsonl)",
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to analyze (default: all valid samples)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed VLM responses"
    )

    parser.add_argument(
        "--test", action="store_true", help="Run with first 10 samples for testing"
    )

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.max_samples = 10
        args.verbose = True
        print("Running in TEST mode with 10 samples")

    # Check environment variables
    required_env_vars = ["MODEL", "BASE_URL", "API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print("Warning: Missing environment variables:", missing_vars)

    analyze_validated_dataset(
        dataset_name=args.dataset,
        validation_file=args.validation_file,
        output_file=args.output,
        max_workers=args.workers,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
