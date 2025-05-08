# utils.py
import os
import re
import logging
from typing import List, Optional, Pattern

# Configure logger for this module
logger = logging.getLogger(__name__)

# Pre-compile regex for efficiency if extract_doi is called frequently
DOI_PATTERN: Pattern[str] = re.compile(r"10.\d{4,9}/[-._;()/:A-Za-z0-9]+")

def extract_doi(text: str) -> List[str]:
    """
    Extracts Digital Object Identifiers (DOIs) from a given text.

    Args:
        text: The text to search for DOIs.

    Returns:
        A list of found DOIs.
    """
    if not text:
        return []
    try:
        return DOI_PATTERN.findall(text)
    except Exception as e:
        logger.error(f"Error extracting DOI: {e}", exc_info=True)
        return []

def normalize_text(text: str) -> str:
    """
    Normalizes text by stripping whitespace and replacing newlines.

    Args:
        text: The input string.

    Returns:
        The normalized string.
    """
    if not text:
        return ""
    return text.strip().replace("\n", " ")

def load_ocr_documents(folder_path: str) -> List[str]:
    """
    Loads and normalizes OCR documents from a specified folder.

    Args:
        folder_path: The path to the folder containing .txt OCR files.

    Returns:
        A list of normalized document contents.
    """
    docs: List[str] = []
    if not os.path.isdir(folder_path):
        logger.error(f"OCR documents folder not found: {folder_path}")
        return docs

    logger.info(f"Loading OCR documents from: {folder_path}")
    try:
        for entry in os.scandir(folder_path):
            if entry.name.endswith(".txt") and entry.is_file():
                try:
                    with open(entry.path, "r", encoding="utf-8") as f:
                        docs.append(normalize_text(f.read()))
                except IOError as e:
                    logger.error(f"Error reading OCR file {entry.path}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Unexpected error processing file {entry.path}: {e}", exc_info=True)
        logger.info(f"Loaded {len(docs)} OCR documents.")
    except OSError as e:
        logger.error(f"Error scanning directory {folder_path}: {e}", exc_info=True)
    return docs

def ensure_output_path_exists(output_path: str) -> None:
    """
    Ensures that the specified output directory exists.

    Args:
        output_path: The path to the directory.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        logger.debug(f"Ensured output path exists: {output_path}")
    except OSError as e:
        logger.error(f"Could not create output directory {output_path}: {e}", exc_info=True)
        raise # Re-raise if directory creation is critical

# Example of a config validation utility (conceptual)
# def validate_config(config: dict) -> bool:
#     required_keys = ["ocr_data_path", "opencitations_path", "output_path", "embedding_model"]
#     for key in required_keys:
#         if key not in config:
#             logger.error(f"Missing key '{key}' in configuration.")
#             return False
#     # Add more specific validations (e.g., path existence, model name format)
#     return True