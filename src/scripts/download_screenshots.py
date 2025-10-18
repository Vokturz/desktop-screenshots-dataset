#!/usr/bin/env python3
"""
Script to download the first screenshot for each project from sourceforge_projects.jsonl
"""

import json
import requests
from pathlib import Path
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.model.model import Metadata


def load_projects(jsonl_path: Path):
    """Load projects from JSONL file"""
    projects: list[Metadata] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    projects.append(Metadata.from_dict(json.loads(line)))
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

    return projects


def download_image(url: str, output_path: Path, timeout: int = 30):
    """Download image from URL and save to output_path"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        # Check if the response is actually an image
        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            print(
                f"Warning: URL {url} doesn't appear to be an image (content-type: {content_type})"
            )
            return False

        # Write the image data
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except IOError as e:
        print(f"Error saving to {output_path}: {e}")
        return False


def download_project(metadata: Metadata, images_dir: Path, stats_lock: Lock):
    """Download screenshot for a single project"""
    slug = metadata.slug
    screenshots = metadata.screenshots

    # Thread-safe statistics
    with stats_lock:
        result = {"status": "processing", "slug": slug}

    if len(screenshots) == 0:
        # print(f"{slug}: No screenshots available")
        result["status"] = "no_screenshots"
        return result

    # Output file path
    output_path = images_dir / f"{slug}.png"

    # Skip if already exists
    if output_path.exists():
        # print(f"{slug}: Already exists, skipping")
        result["status"] = "already_exists"
        return result

    # Get last screenshot URL
    last_screenshot = screenshots[-1]
    print(f"{slug}: Downloading from {last_screenshot}")

    # Download the image
    if download_image(last_screenshot, output_path):
        # print(f"{slug}: Successfully downloaded")
        result["status"] = "downloaded"
    else:
        print(f"{slug}: Failed to download")
        result["status"] = "failed"

    return result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download screenshots from SourceForge projects"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of concurrent downloads (default: 10, recommended range: 5-20)",
    )
    args = parser.parse_args()

    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    jsonl_path = data_dir / "sourceforge_projects.jsonl"
    images_dir = data_dir / "images"

    # Ensure images directory exists
    images_dir.mkdir(exist_ok=True)

    # Load projects
    print(f"Loading projects from {jsonl_path}")
    projects = load_projects(jsonl_path)
    print(f"Found {len(projects)} projects")

    # Statistics
    stats_lock = Lock()
    downloaded = 0
    skipped_no_screenshots = 0
    skipped_already_exists = 0
    failed = 0

    # Number of concurrent downloads
    max_workers = args.max_workers

    print(f"\nStarting parallel downloads with {max_workers} workers...")

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_project = {
            executor.submit(download_project, project, images_dir, stats_lock): project
            for project in projects
        }

        # Process completed downloads
        completed = 0
        for future in as_completed(future_to_project):
            result = future.result()
            completed += 1

            # Update statistics
            if result["status"] == "downloaded":
                downloaded += 1
            elif result["status"] == "no_screenshots":
                skipped_no_screenshots += 1
            elif result["status"] == "already_exists":
                skipped_already_exists += 1
            elif result["status"] == "failed":
                failed += 1

            # Progress indicator
            if completed % 50 == 0 or completed == len(projects):
                print(f"Progress: {completed}/{len(projects)} projects processed")

    # Print summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Total projects: {len(projects)}")
    print(f"Successfully downloaded: {downloaded}")
    print(f"No screenshots available: {skipped_no_screenshots}")
    print(f"Already existed: {skipped_already_exists}")
    print(f"Failed downloads: {failed}")
    print(f"Images saved to: {images_dir}")


if __name__ == "__main__":
    main()
