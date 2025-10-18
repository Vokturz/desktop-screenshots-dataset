#!/usr/bin/env python3
"""
Script to scrape project slugs, fetch metadata, and save to a file.
"""

import argparse
import json
import time
from pathlib import Path
import requests
from tqdm import tqdm

from src.utils.sourceforge_utils import (
    fetch_directory_page,
    get_project_metadata,
    parse_metadata,
    parse_project_slugs,
    get_mirror_metadata,
)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_FILE = DATA_DIR / "sourceforge_projects.jsonl"
FAILED_SLUGS_FILE = DATA_DIR / "sourceforge_failed_slugs.txt"


def load_existing_slugs(file_path: Path) -> set[str]:
    """Load already processed project slugs from the output file to avoid duplicates."""
    if not file_path.exists():
        return set()

    existing_slugs: set[str] = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "slug" in data:
                    existing_slugs.add(data["slug"])
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON line: {line.strip()}")
    return existing_slugs


def main(num_pages: int, start_page: int = 1):
    """
    Main function to scrape project slugs, fetch metadata, and save to a file.
    """
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Checking for existing data in {OUTPUT_FILE}...")
    existing_slugs = load_existing_slugs(OUTPUT_FILE)
    print(f"Found {len(existing_slugs)} existing projects. They will be skipped.")

    # Get all slugs for N pages
    all_slugs: list[str] = []
    end_page = start_page + num_pages
    print(
        f"\nFetching {num_pages} project pages, starting from page {start_page} up to page {end_page - 1}..."
    )
    for page_num in tqdm(range(start_page, end_page), desc="Scraping Pages"):
        try:
            html = fetch_directory_page(page_num)
            slugs = parse_project_slugs(html)
            all_slugs.extend(slugs)
            time.sleep(0.2)  # Be polite to the server
        except requests.RequestException as e:
            print(f"Error fetching page {page_num}: {e}. Skipping page.")
            continue

    unique_slugs = sorted(list(set(all_slugs)))
    print(f"Found {len(unique_slugs)} unique project slugs in total.")

    # Filter out slugs that have already been processed
    slugs_to_process = [slug for slug in unique_slugs if slug not in existing_slugs]
    print(f"Found {len(slugs_to_process)} new projects to process.")

    if not slugs_to_process:
        print("No new projects to add. Exiting.")
        return

    # Get metadata for each slug and append to the file
    new_projects_count = 0
    failed_slugs: list[str] = []
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for slug in tqdm(slugs_to_process, desc="Fetching Metadata"):
            try:
                raw_metadata = get_project_metadata(slug)
                if raw_metadata:
                    parsed_data = parse_metadata(raw_metadata)
                    if parsed_data.slug in existing_slugs:
                        continue
                    f.write(parsed_data.to_json() + "\n")
                    existing_slugs.add(parsed_data.slug)
                    new_projects_count += 1
                else:
                    failed_slugs.append(slug)
            except Exception as _:
                # Retry logic for slugs ending in '.mirror'
                if slug.endswith(".mirror"):
                    try:
                        parsed_data = get_mirror_metadata(slug)
                        parsed_data.is_mirror = True
                        existing_slugs.add(parsed_data.slug)
                        f.write(parsed_data.to_json() + "\n")
                        new_projects_count += 1
                    except Exception as e:
                        print(e)
                        failed_slugs.append(slug)
                        continue
                else:
                    continue

            time.sleep(0.2)  # To be polite to the API

    if failed_slugs:
        print(f"\nLogging {len(failed_slugs)} failed slugs to {FAILED_SLUGS_FILE}...")
        with open(FAILED_SLUGS_FILE, "w", encoding="utf-8") as f_fail:
            for slug in failed_slugs:
                f_fail.write(f"{slug}\n")

    print(f"\n✅ Done. Added {new_projects_count} new projects to {OUTPUT_FILE}.")
    if failed_slugs:
        print(f"Logged {len(failed_slugs)} slugs that failed to fetch metadata.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape project metadata from SourceForge."
    )
    parser.add_argument(
        "-p",
        "--pages",
        type=int,
        default=20,
        help="Number of directory pages to scrape for project slugs.",
    )

    parser.add_argument(
        "-s",
        "--start-page",
        type=int,
        default=1,
        help="The page number to start scraping from.",
    )
    args = parser.parse_args()

    main(num_pages=args.pages, start_page=args.start_page)
