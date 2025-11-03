#!/usr/bin/env python3
"""
OCR analysis script for validated sourceforge app screenshots.

This script processes screenshots from the "Vokturz/sourceforge-app-screenshots" dataset
that have been validated as valid (is_valid==True) in the validation results.
For each valid screenshot, it performs OCR to extract text content using the
image analysis data from analysis_results.jsonl to provide better context.

Results are saved as JSONL with slug and OCR text. Screenshots where OCR fails
are skipped (not added to the output file).
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

from src.utils.vlm_utils import do_ocr
from src.model.model import ImageAnalysis

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


def load_analysis_results(analysis_file: Path) -> dict[str, ImageAnalysis]:
    """Load image analysis results from JSONL file.

    Args:
        analysis_file: Path to the analysis results JSONL file

    Returns:
        Dictionary mapping slug to ImageAnalysis objects
    """
    analysis_results = {}

    if not analysis_file.exists():
        print(f"Warning: Analysis file not found: {analysis_file}")
        return analysis_results

    try:
        with open(analysis_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    result = json.loads(line)
                    if "slug" in result:
                        slug = result["slug"]
                        # Create ImageAnalysis object from the analysis data
                        analysis_data = {
                            "description": result.get("image_description", ""),
                            "keywords": result.get("keywords", []),
                            "category": result.get("category", ""),
                        }
                        analysis_results[slug] = ImageAnalysis.from_dict(analysis_data)
    except Exception as e:
        print(f"Error reading analysis results: {e}")

    return analysis_results


def ocr_single_row(
    row: dict[str, Any],
    image_analysis: ImageAnalysis | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Perform OCR on a single dataset row.

    Args:
        row: Dataset row containing image, slug, title, description
        image_analysis: ImageAnalysis object to provide context for OCR
        verbose: Whether to print OCR responses

    Returns:
        dict with slug and OCR text, or None if OCR fails
    """
    try:
        slug = row["slug"]

        if verbose:
            analysis_status = (
                "with analysis context"
                if image_analysis
                else "without analysis context"
            )
            print(f"Processing OCR for {slug} ({analysis_status})...")

        # Perform OCR
        ocr_text = do_ocr(
            row,
            image_analysis=image_analysis,
            print_assistant_message=verbose,
            custom_system_prompt=None,
        )

        # Skip if OCR text is empty or whitespace only
        if not ocr_text or not ocr_text.strip():
            if verbose:
                print(f"✗ {slug}: OCR returned empty result, skipping")
            return None

        result = {
            "slug": slug,
            "ocr": ocr_text.strip(),
        }

        if verbose:
            analysis_info = ""
            if image_analysis:
                analysis_info = f" (used analysis: {image_analysis.category})"
            print(f"✓ {slug}: OCR complete{analysis_info}")
            print(f"  Text length: {len(ocr_text.strip())} characters")
            # Show first 100 chars as preview
            preview = ocr_text.strip()[:100]
            if len(ocr_text.strip()) > 100:
                preview += "..."
            print(f"  Preview: {preview}")

        return result

    except Exception as e:
        if verbose:
            print(f"✗ Error performing OCR on {row.get('slug', 'unknown')}: {str(e)}")
        # Don't print error in non-verbose mode to keep output clean
        return None


def load_existing_ocr_slugs(output_path: Path) -> set[str]:
    """Load existing slugs from OCR results file.

    Args:
        output_path: Path to the OCR results JSONL file

    Returns:
        Set of slugs that have already been processed
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
            print(f"Warning: Error reading existing OCR results file: {e}")

    return existing_slugs


def ocr_validated_dataset(
    dataset_name: str = "Vokturz/sourceforge-app-screenshots",
    validation_file: str = "data/validation_results.jsonl",
    analysis_file: str = "data/analysis_results.jsonl",
    output_file: str | None = None,
    max_workers: int = 4,
    max_samples: int | None = None,
    verbose: bool = False,
) -> None:
    """Perform OCR on validated dataset using parallel processing.

    Args:
        dataset_name: HuggingFace dataset name
        validation_file: Path to validation results JSONL file
        analysis_file: Path to analysis results JSONL file
        output_file: Output JSONL file path
        max_workers: Number of parallel workers
        max_samples: Limit number of samples to process (None for all valid samples)
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
    print(f"Found {len(valid_slugs)} valid screenshots to process")

    # Load analysis results
    analysis_path = Path(__file__).parent.parent / analysis_file
    analysis_results = load_analysis_results(analysis_path)
    print(f"Loaded {len(analysis_results)} analysis results")

    # Check how many valid screenshots have analysis data
    valid_with_analysis = sum(1 for slug in valid_slugs if slug in analysis_results)
    print(
        f"Found analysis data for {valid_with_analysis}/{len(valid_slugs)} valid screenshots"
    )

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
        output_file = "data/ocr_results.jsonl"

    # Ensure output directory exists
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing OCR slugs to skip already processed items
    existing_slugs = load_existing_ocr_slugs(output_path)
    if existing_slugs:
        print(f"Found {len(existing_slugs)} already processed slugs, skipping them")

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

    # Skip processing if no new data to process
    if not data:
        print("No new items to process. All valid items have been processed already.")
        return

    # Process in parallel and append results immediately
    results: list[dict[str, Any]] = []
    skipped_count = 0
    used_analysis_count = 0
    print(f"Appending results to: {output_path}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all OCR tasks
        future_to_idx = {}
        for idx, row in enumerate(data):  # pyright: ignore
            slug = row["slug"]  # pyright: ignore
            image_analysis = analysis_results.get(slug)
            if image_analysis:
                used_analysis_count += 1
            future_to_idx[
                executor.submit(ocr_single_row, row, image_analysis, verbose)  # pyright: ignore
            ] = idx

        # Process completed tasks with progress bar and append immediately
        with tqdm(total=dataset_size, desc="OCR Processing") as pbar:
            with open(output_path, "a", encoding="utf-8") as f:
                for future in as_completed(future_to_idx):
                    try:
                        result = future.result()

                        # Skip if result is None (OCR failed or returned empty)
                        if result is None:
                            skipped_count += 1
                            pbar.update(1)
                            continue

                        results.append(result)  # pyright: ignore

                        # Append result immediately to file
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()  # Ensure data is written immediately

                        pbar.update(1)

                        # Update progress description with success stats
                        success_rate = (
                            len(results) / (len(results) + skipped_count) * 100
                            if (len(results) + skipped_count) > 0
                            else 0
                        )
                        pbar.set_postfix_str(
                            f"Success: {len(results)}, Skipped: {skipped_count}, Rate: {success_rate:.1f}%"
                        )

                    except Exception as e:
                        idx = future_to_idx[future]
                        slug = data[idx].get("slug", f"index_{idx}")
                        if verbose:
                            print(f"Task failed for {slug}: {e}")
                        skipped_count += 1
                        pbar.update(1)

    # Print summary statistics for this run
    print("\nOCR Processing Summary (this run):")
    print(f"  Successfully processed: {len(results)}")
    print(f"  Skipped (failed/empty): {skipped_count}")
    print(f"  Used analysis context: {used_analysis_count}/{dataset_size}")
    total_attempted = len(results) + skipped_count
    if total_attempted > 0:
        success_rate = len(results) / total_attempted * 100
        analysis_usage_rate = (
            used_analysis_count / dataset_size * 100 if dataset_size > 0 else 0
        )
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Analysis context usage: {analysis_usage_rate:.1f}%")

    if results:
        # Calculate text length statistics
        text_lengths = [len(result["ocr"]) for result in results]
        avg_length = sum(text_lengths) / len(text_lengths)
        min_length = min(text_lengths)
        max_length = max(text_lengths)

        print("  OCR text statistics:")
        print(f"    Average length: {avg_length:.1f} characters")
        print(f"    Min length: {min_length} characters")
        print(f"    Max length: {max_length} characters")

    print(f"  Results appended to: {output_path}")

    # Show total statistics if there were existing results
    if existing_slugs:
        total_existing = len(existing_slugs)
        total_processed = total_existing + len(results)
        print("\nTotal OCR progress:")
        print(f"  Previously processed: {total_existing}")
        print(f"  Newly processed: {len(results)}")
        print(f"  Total processed: {total_processed}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Perform OCR on validated sourceforge app screenshots"
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
        "--analysis-file",
        default="data/analysis_results.jsonl",
        help="Path to analysis results JSONL file (default: data/analysis_results.jsonl)",
    )

    parser.add_argument(
        "--output",
        help="Output JSONL file path (default: data/ocr_results.jsonl)",
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (default: all valid samples)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed OCR responses"
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

    ocr_validated_dataset(
        dataset_name=args.dataset,
        validation_file=args.validation_file,
        analysis_file=args.analysis_file,
        output_file=args.output,
        max_workers=args.workers,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
