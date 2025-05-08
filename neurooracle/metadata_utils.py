#!/usr/bin/env python3
"""
metadata_utils.py

Functionality 1: Rename each PDF in a folder based on its DOI metadata (Crossref),
                 skipping any files that are already renamed. Supports dry-run.
Functionality 2: Create a CSV file containing metadata fetched from Crossref
                 for a list of DOIs provided in a file.

Usage:
  # Preview PDF renames only:
  python metadata_utils.py --outurls found_pdfs.txt --pdf-dir pdfs --dry-run

  # Execute PDF renames:
  python metadata_utils.py --outurls found_pdfs.txt --pdf-dir pdfs

  # Create metadata CSV:
  python metadata_utils.py --outurls found_pdfs.txt --create-csv my_metadata.csv
"""
import os
import re
import requests
import argparse
import csv
import logging
from typing import Dict, List, Optional # Added typing hints

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def sanitize(text: str) -> str:
    """Removes or replaces characters unsafe for filenames."""
    # Keep alphanumeric, underscore, dot, hyphen. Replace others with underscore.
    return re.sub(r'[^\w\.-]', '_', text)

def fetch_metadata(doi: str) -> Dict:
    """Fetches metadata (title, authors, year) for a DOI from Crossref."""
    logger.debug(f"Fetching metadata for DOI: {doi}")
    url = f"https://api.crossref.org/works/{doi}"
    # Adding a User-Agent is good practice for APIs
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'NeuroOracleMetadataFetcher/1.0 (mailto:your-email@example.com; contact if issues)' # Optional: Replace with your email
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15) # Increased timeout
        resp.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)

        data = resp.json()
        # Basic check for expected response structure
        if not isinstance(data, dict) or data.get('status') != 'ok':
            logger.warning(f"CrossRef response status not OK for {doi}. Status: {data.get('status')}")
            return {} # Return empty dict on non-OK status

        msg = data.get('message', {})
        if not isinstance(msg, dict):
             logger.warning(f"Unexpected 'message' format in CrossRef response for {doi}")
             return {}

        # Extract title safely
        title_list = msg.get('title', [])
        title = title_list[0] if title_list and isinstance(title_list[0], str) else ''

        # Extract authors safely
        authors: List[str] = []
        author_list = msg.get('author', [])
        if isinstance(author_list, list):
            for a in author_list:
                 # Handle potential variations in author structure
                 if isinstance(a, dict):
                     given = a.get('given', '') or ""
                     family = a.get('family', '') or ""
                     # Ensure names are strings before stripping/joining
                     if isinstance(given, str) and isinstance(family, str):
                         full_name = f"{given} {family}".strip()
                         if full_name: authors.append(full_name)
                     elif isinstance(family, str) and family: # Handle family name only
                         authors.append(family)
                     elif isinstance(given, str) and given: # Handle given name only
                          authors.append(given)

        # Extract year safely from different date fields
        year: Optional[int] = None
        year_str: str = ""
        for date_key in ("published-print", "published-online", "created", "issued"):
            date_info = msg.get(date_key)
            if isinstance(date_info, dict):
                date_parts_list = date_info.get("date-parts", [[None]])
                if (isinstance(date_parts_list, list) and
                    date_parts_list and isinstance(date_parts_list[0], list) and
                    date_parts_list[0] and date_parts_list[0][0] is not None):
                    try:
                         # Attempt to convert year part to int, handle potential non-numeric values
                         year_candidate = int(date_parts_list[0][0])
                         # Basic validation for sensible year range
                         if 1800 < year_candidate < 2100:
                             year = year_candidate
                             year_str = str(year)
                             break # Stop searching once a valid year is found
                         else:
                            logger.debug(f"Year {year_candidate} out of range in '{date_key}' for {doi}")
                    except (ValueError, TypeError):
                         logger.debug(f"Non-integer year found ('{date_parts_list[0][0]}') in '{date_key}' for {doi}")
                         continue # Try next date key
            if year: break # Exit outer loop if year found

        return {"title": title, "authors": authors, "year": year_str} # Return year as string

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching metadata for {doi}")
        raise # Re-raise timeout
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {doi}: {e}")
        raise # Re-raise other request exceptions


# --- Core Functionality Functions ---

def rename_pdfs(pdf_dir: str, doi_list_file: str, dry_run: bool):
    """
    Renames PDFs in pdf_dir based on metadata fetched for DOIs in doi_list_file.
    Assumes original PDFs are named like <sanitized_doi>.pdf.
    """
    logger.info(f"Starting PDF rename process {'(DRY RUN)' if dry_run else ''}...")
    logger.info(f"PDF Directory: {pdf_dir}")
    logger.info(f"DOI List File: {doi_list_file}")

    # --- Read DOIs ---
    dois_to_process: Dict[str, str] = {} # Maps safe_doi_filename -> original_doi
    try:
        with open(doi_list_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Split on TAB as written by download_pdfs.py
                parts = line.strip().split('\t', 1)
                if len(parts) >= 1 and parts[0]:
                    original_doi = parts[0]
                    # Assume original PDF filenames might be based on sanitized DOIs
                    safe_doi_filename = sanitize(original_doi)
                    dois_to_process[safe_doi_filename] = original_doi
                else:
                     logger.warning(f"Could not parse DOI from line {line_num} in {doi_list_file}: '{line.strip()}'")
    except FileNotFoundError:
        logger.error(f"Error: DOI list file not found at {doi_list_file}")
        return
    except Exception as e:
        logger.error(f"Error reading DOI list file {doi_list_file}: {e}")
        return

    if not dois_to_process:
        logger.error(f"No valid DOIs found in {doi_list_file}. Aborting rename.")
        return

    logger.info(f"Found {len(dois_to_process)} DOIs to process.")

    # --- Process and Rename ---
    rename_count = 0
    skip_count = 0
    fail_count = 0

    for safe_doi_filename, original_doi in dois_to_process.items():
        old_name = f"{safe_doi_filename}.pdf"
        old_path = os.path.join(pdf_dir, old_name)

        if not os.path.exists(old_path):
            logger.debug(f"Skipping {old_name} (original file not found based on sanitized DOI)")
            skip_count += 1
            continue

        logger.info(f"Processing {old_name} (DOI: {original_doi})")
        try:
            meta = fetch_metadata(original_doi)
            if not meta or not meta.get("title"): # Check if metadata or title is missing
                 logger.warning(f"Insufficient metadata retrieved for {original_doi}, skipping rename.")
                 fail_count += 1
                 continue

        except Exception as e:
            logger.warning(f"Skipping rename for {original_doi} due to metadata fetch error: {e}")
            fail_count += 1
            continue

        # Construct new filename
        authors = meta.get("authors", [])
        # Use only the first author's last name for simplicity
        family = authors[0].split()[-1] if authors and authors[0].split() else "Unknown"
        year = meta.get("year", "YearUnknown") or "YearUnknown" # Ensure year is not empty string

        title = meta.get("title", "NoTitle") or "NoTitle" # Ensure title is not empty string
        title_words = title.split()
        # Limit snippet length for sanity
        title_snip = "_".join(title_words[:6]) if title_words else "NoTitle"

        # Sanitize all components before joining
        new_name = f"{sanitize(str(year))}_{sanitize(family)}_et_al_{sanitize(title_snip)}.pdf"
        new_path = os.path.join(pdf_dir, new_name)

        # Check if target file already exists
        if os.path.exists(new_path):
            if old_path == new_path:
                 logger.debug(f"Skipping {old_name} (already correctly named)")
                 skip_count +=1
            else:
                 logger.warning(f"Skipping rename for {old_name}: target file '{new_name}' already exists.")
                 skip_count += 1
            continue

        # Perform rename
        logger.info(f"  Renaming: '{old_name}' → '{new_name}'")
        if not dry_run:
            try:
                os.rename(old_path, new_path)
                rename_count += 1
            except OSError as e:
                logger.error(f"  FAILED to rename {old_name} to {new_name}: {e}")
                fail_count += 1
        else:
            # In dry run, still count as if renamed for reporting
            rename_count +=1

    logger.info("-" * 20)
    logger.info(f"PDF Renaming {'DRY RUN ' if dry_run else ''}Complete.")
    logger.info(f"Renamed/To Rename: {rename_count}")
    logger.info(f"Skipped: {skip_count}")
    logger.info(f"Failed: {fail_count}")


def create_metadata_csv(doi_list_file: str, output_csv_path: str):
    """
    Fetches metadata for DOIs in a list file and saves to a CSV.
    """
    logger.info(f"Starting Metadata CSV creation...")
    logger.info(f"DOI List File: {doi_list_file}")
    logger.info(f"Output CSV File: {output_csv_path}")

    # --- Read DOIs ---
    dois: List[str] = []
    try:
        with open(doi_list_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Split on TAB as written by download_pdfs.py
                parts = line.strip().split('\t', 1) # <<< CORRECTED SPLIT HERE
                if len(parts) >= 1 and parts[0]:
                    original_doi = parts[0]
                    # Basic DOI format check (optional but recommended)
                    if re.match(r'^10\.\d{4,9}/', original_doi):
                         dois.append(original_doi)
                    else:
                        logger.warning(f"Line {line_num}: Ignoring invalid DOI format '{original_doi}'")
                else:
                     logger.warning(f"Could not parse DOI from line {line_num} in {doi_list_file}: '{line.strip()}'")
    except FileNotFoundError:
        logger.error(f"Error: DOI list file not found at {doi_list_file}")
        return
    except Exception as e:
        logger.error(f"Error reading DOI list file {doi_list_file}: {e}")
        return

    if not dois:
        logger.error(f"No valid DOIs found in {doi_list_file}. Aborting CSV creation.")
        return

    logger.info(f"Found {len(dois)} valid DOIs. Fetching metadata...")

    # Define CSV headers - ensure these match NeuroOracle's needs
    # forecast.py uses 'title', 'year'. 'doi' is essential identifier.
    fieldnames = ['doi', 'title', 'year', 'authors_str']

    success_count = 0
    fail_count = 0
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_csv_path)
        if output_dir: # Ensure it's not empty (meaning file is in current dir)
            os.makedirs(output_dir, exist_ok=True)

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader() # Write the header row

            for doi in dois:
                try:
                    meta = fetch_metadata(doi)
                    if not meta: # Check if metadata fetch returned empty dict
                        logger.warning(f"No metadata retrieved for {doi}, skipping CSV row.")
                        fail_count += 1
                        continue

                    # Convert authors list to a single string, separated by semicolon
                    authors_str = "; ".join(a for a in meta.get("authors", []) if a)

                    row = {
                        'doi': doi,
                        'title': meta.get("title", ""),
                        'year': meta.get("year", ""),
                        'authors_str': authors_str
                    }
                    writer.writerow(row)
                    # logger.debug(f"✓ Fetched and wrote metadata for {doi}") # Make this debug if too verbose
                    success_count += 1
                    if success_count % 50 == 0: # Log progress every 50 DOIs
                         logger.info(f"Processed {success_count}/{len(dois)} DOIs...")

                except Exception as e:
                    logger.error(f"⚠️ Metadata fetch/write failed for {doi}: {e}")
                    fail_count += 1
                    # Optionally write a row indicating failure
                    writer.writerow({'doi': doi, 'title': 'METADATA_FETCH_FAILED', 'year': '', 'authors_str': ''})

        logger.info("-" * 20)
        logger.info(f"Metadata CSV creation complete.")
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"Output saved to {output_csv_path}")

    except IOError as e:
        logger.error(f"Error: Could not write to output CSV file {output_csv_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV creation: {e}")


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utils for PDF metadata: Rename PDFs based on metadata OR create a metadata CSV file.",
        formatter_class=argparse.RawTextHelpFormatter # Nicer help text
    )
    # Input DOI list (required for both actions)
    parser.add_argument("--outurls", required=True,
                        help="Path to the input file containing DOIs (e.g., found_pdfs.txt).\n"
                             "Expected format: DOI<tab>URL per line.")

    # Arguments for Renaming action
    parser.add_argument("--pdf-dir",
                        help="Folder of PDFs to rename (REQUIRED for renaming action).\n"
                             "Assumes PDFs are named like <sanitized_doi>.pdf.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview renames only, don't actually rename files (for renaming action).")

    # Argument for CSV Creation action
    parser.add_argument("--create-csv",
                        help="Path to the output metadata CSV file (REQUIRED for CSV creation action).\n"
                             "Example: my_metadata.csv")

    args = parser.parse_args()

    # Decide which action to perform based on arguments provided
    if args.create_csv and args.pdf_dir:
        logger.error("Error: Please specify EITHER --create-csv OR --pdf-dir, not both.")
    elif args.create_csv:
        # Run CSV creation function
        create_metadata_csv(args.outurls, args.create_csv)
    elif args.pdf_dir:
        # Run PDF renaming function
        rename_pdfs(args.pdf_dir, args.outurls, args.dry_run)
    else:
        logger.error("Error: No action specified. Please use either --create-csv or --pdf-dir.")
        parser.print_help() # Show help message if no action given