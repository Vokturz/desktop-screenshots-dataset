#!/usr/bin/env python3
"""
Dataset validation script for sourceforge app screenshots.

This script validates each row from the "Vokturz/sourceforge-app-screenshots" dataset
using VLM (Vision Language Model) to determine if screenshots are valid.
Results are saved as JSONL with slug and validity status.
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

from src.utils.vlm_utils import check_screenshot_validity

load_dotenv()

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false


def validate_single_row(row: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
    """Validate a single dataset row using VLM.

    Args:
        row: Dataset row containing image, slug, title, description
        verbose: Whether to print VLM responses

    Returns:
        dict with slug and is_valid fields
    """
    try:
        slug = row["slug"]
        is_valid = check_screenshot_validity(row, print_assistant_message=verbose)

        result = {"slug": slug, "is_valid": is_valid}

        if verbose:
            print(f"✓ {slug}: {'VALID' if is_valid else 'INVALID'}")

        return result

    except Exception as e:
        print(f"✗ Error validating {row.get('slug', 'unknown')}: {str(e)}")
        return {"slug": row.get("slug", "unknown"), "is_valid": False}


def load_existing_slugs(output_path: Path) -> set[str]:
    """Load existing slugs from validation results file.

    Args:
        output_path: Path to the validation results JSONL file

    Returns:
        Set of slugs that have already been validated
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
            print(f"Warning: Error reading existing results file: {e}")

    return existing_slugs


def validate_dataset_parallel(
    dataset_name: str = "Vokturz/sourceforge-app-screenshots",
    output_file: str | None = None,
    max_workers: int = 4,
    max_samples: int | None = None,
    verbose: bool = False,
) -> None:
    """Validate dataset using parallel VLM calls.

    Args:
        dataset_name: HuggingFace dataset name
        output_file: Output JSONL file path
        max_workers: Number of parallel workers
        max_samples: Limit number of samples to validate (None for all)
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

    # Limit samples if requested
    dataset_size = len(data)  # pyright: ignore
    if max_samples and max_samples < dataset_size:
        data = data.select(range(max_samples))  # pyright: ignore
        print(f"Limited to {max_samples} samples")

    print(f"Validating {dataset_size} samples with {max_workers} workers")

    # Setup output file
    if output_file is None:
        output_file = "data/validation_results.jsonl"

    # Ensure output directory exists
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing slugs to skip already processed items
    existing_slugs = load_existing_slugs(output_path)
    if existing_slugs:
        print(f"Found {len(existing_slugs)} already processed slugs, skipping them")

    # Filter out already processed items
    filtered_indices: list[int] = []
    for idx, row in enumerate(data):  # pyright: ignore
        if row["slug"] not in existing_slugs:
            filtered_indices.append(idx)

    original_size = len(data)  # pyright: ignore
    if len(filtered_indices) < original_size:
        print(
            f"Filtered {original_size - len(filtered_indices)} already processed items"
        )
        print(f"Processing {len(filtered_indices)} remaining items")

    # Update data to use filtered indices
    if filtered_indices:
        data = data.select(filtered_indices)  # pyright: ignore
    else:
        data = []
    dataset_size = len(filtered_indices)

    # Skip validation if no new data to process
    if not data:
        print("No new items to validate. All items have been processed already.")
        return

    # Validate in parallel and append results immediately
    results: list[dict[str, Any]] = []
    print(f"Appending results to: {output_path}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all validation tasks
        future_to_idx = {
            executor.submit(validate_single_row, row, verbose): idx  # pyright: ignore
            for idx, row in enumerate(data)  # pyright: ignore
        }

        # Process completed tasks with progress bar and append immediately
        with tqdm(total=dataset_size, desc="Validating") as pbar:
            with open(output_path, "a", encoding="utf-8") as f:
                for future in as_completed(future_to_idx):
                    try:
                        result = future.result()
                        results.append(result)

                        # Append result immediately to file
                        f.write(json.dumps(result) + "\n")
                        f.flush()  # Ensure data is written immediately

                        pbar.update(1)

                        # Update progress description with validation stats
                        valid_count = sum(1 for r in results if r["is_valid"])
                        pbar.set_postfix(
                            {
                                "valid": valid_count,
                                "invalid": len(results) - valid_count,
                            }
                        )

                    except Exception as e:
                        idx = future_to_idx[future]
                        slug = data[idx].get("slug", f"index_{idx}")
                        print(f"Task failed for {slug}: {e}")
                        # error_result = {"slug": slug, "is_valid": False}
                        # results.append(error_result)

                        # # Append error result immediately to file
                        # f.write(json.dumps(error_result) + "\n")
                        # f.flush()

                        pbar.update(1)

    # Print summary statistics for this run
    valid_count = sum(1 for r in results if r["is_valid"])
    invalid_count = len(results) - valid_count

    print("\nValidation Summary (this run):")
    print(f"  Processed samples: {len(results)}")
    print(
        f"  Valid: {valid_count} ({valid_count / len(results) * 100:.1f}% of processed)"
        if results
        else "  No samples processed"
    )
    if results:
        print(
            f"  Invalid: {invalid_count} ({invalid_count / len(results) * 100:.1f}% of processed)"
        )
    print(f"  Results appended to: {output_path}")

    # Show total statistics if there were existing results
    if existing_slugs:
        total_existing = len(existing_slugs)
        total_processed = total_existing + len(results)
        print("\nTotal dataset progress:")
        print(f"  Previously processed: {total_existing}")
        print(f"  Newly processed: {len(results)}")
        print(f"  Total processed: {total_processed}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Validate sourceforge app screenshots dataset using VLM"
    )

    parser.add_argument(
        "--dataset",
        default="Vokturz/sourceforge-app-screenshots",
        help="HuggingFace dataset name (default: Vokturz/sourceforge-app-screenshots)",
    )

    parser.add_argument(
        "--output",
        help="Output JSONL file path (default: data/validation_results.jsonl)",
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to validate (default: all)",
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

    validate_dataset_parallel(
        dataset_name=args.dataset,
        output_file=args.output,
        max_workers=args.workers,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
