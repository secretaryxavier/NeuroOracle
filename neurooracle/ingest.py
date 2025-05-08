# ingest.py
import os
import json
import pandas as pd
import logging
from typing import Dict, Any, Optional, List

# Configure logger for this module
logger = logging.getLogger(__name__)

class CorpusIngester:
    """
    Handles ingestion of OCR data and OpenCitations metadata.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the Ingester with configuration.

        Args:
            config_path: Path to the JSON configuration file.
        """
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.ocr_path: Optional[str] = self.config.get("ocr_data_path")
        opencitations_base_path: Optional[str] = self.config.get("opencitations_path")

        if not self.ocr_path:
            logger.error("ocr_data_path not found in config.")
            raise ValueError("ocr_data_path is required in config.")
        if not opencitations_base_path:
            logger.error("opencitations_path not found in config.")
            raise ValueError("opencitations_path is required in config.")

        self.meta_path: str = os.path.join(opencitations_base_path, "opencitations_meta.csv")
        self.index_path: str = os.path.join(opencitations_base_path, "index.csv")
        self.ocr_docs: List[str] = []
        self.meta_df: Optional[pd.DataFrame] = None
        self.index_df: Optional[pd.DataFrame] = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration from a JSON file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Consider adding config validation here using a schema or utils.validate_config(config)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}", exc_info=True)
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from configuration file: {config_path}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}", exc_info=True)
            raise


    def load_ocr_data(self) -> None:
        """
        Loads OCR documents from the path specified in the config.
        Uses the load_ocr_documents utility function.
        """
        if not self.ocr_path or not os.path.isdir(self.ocr_path):
            logger.error(f"OCR data path is invalid or not configured: {self.ocr_path}")
            return

        logger.info(f"Loading OCR data from {self.ocr_path}...")
        # Re-using the utility function from utils.py
        from utils import load_ocr_documents
        self.ocr_docs = load_ocr_documents(self.ocr_path)
        logger.info(f"Successfully loaded {len(self.ocr_docs)} OCR documents.")


    def load_opencitations_data(self) -> None:
        """
        Loads OpenCitations metadata and index CSV files.
        """
        logger.info("Loading OpenCitations metadata and index...")
        try:
            # For very large CSVs, consider using pd.read_csv with chunksize,
            # or specifying dtypes and usecols to optimize memory.
            self.meta_df = pd.read_csv(self.meta_path)
            logger.info(f"Loaded OpenCitations metadata from {self.meta_path} ({len(self.meta_df)} rows)")
        except FileNotFoundError:
            logger.error(f"OpenCitations metadata file not found: {self.meta_path}", exc_info=True)
            # Decide if this is a critical error or if the pipeline can continue
        except pd.errors.EmptyDataError:
            logger.error(f"OpenCitations metadata file is empty: {self.meta_path}", exc_info=True)
        except Exception as e:
            logger.error(f"Error loading OpenCitations metadata: {e}", exc_info=True)

        try:
            self.index_df = pd.read_csv(self.index_path)
            logger.info(f"Loaded OpenCitations index from {self.index_path} ({len(self.index_df)} rows)")
        except FileNotFoundError:
            logger.error(f"OpenCitations index file not found: {self.index_path}", exc_info=True)
        except pd.errors.EmptyDataError:
            logger.error(f"OpenCitations index file is empty: {self.index_path}", exc_info=True)
        except Exception as e:
            logger.error(f"Error loading OpenCitations index: {e}", exc_info=True)