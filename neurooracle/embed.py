# embed.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from utils import load_ocr_documents, ensure_output_path_exists # Assuming utils.py is in the same directory or Python path

# Configure logger for this module
logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    Handles embedding of text documents using SentenceTransformer models.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the TextEmbedder.

        Args:
            config_path: Path to the JSON configuration file.
        """
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.model_name: Optional[str] = self.config.get("embedding_model")
        self.ocr_path: Optional[str] = self.config.get("ocr_data_path")
        self.output_path: Optional[str] = self.config.get("output_path")

        if not all([self.model_name, self.ocr_path, self.output_path]):
            msg = "embedding_model, ocr_data_path, or output_path missing in config."
            logger.error(msg)
            raise ValueError(msg)
        
        ensure_output_path_exists(self.output_path) # Ensure output dir exists

        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model: SentenceTransformer = SentenceTransformer(self.model_name)
            logger.info("Sentence transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{self.model_name}': {e}", exc_info=True)
            raise

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration from a JSON file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}", exc_info=True)
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from configuration file: {config_path}", exc_info=True)
            raise

    def embed_papers(self) -> Optional[np.ndarray]:
        """
        Embeds OCR paper documents and saves the embedding matrix.

        Returns:
            The generated embedding matrix, or None if an error occurs.
        """
        if not self.ocr_path:
            logger.error("OCR path not configured for embedding.")
            return None
            
        logger.info(f"Loading documents for embedding from: {self.ocr_path}")
        docs: List[str] = load_ocr_documents(self.ocr_path)

        if not docs:
            logger.warning("No documents found or loaded from OCR path. Skipping embedding.")
            return None

        logger.info(f"Embedding {len(docs)} OCR papers using model: {self.model_name}...")
        try:
            # The library handles batching, show_progress_bar is useful for interactive runs
            embeddings: np.ndarray = self.model.encode(
                docs,
                show_progress_bar=True, # Consider making this configurable for non-interactive environments
                convert_to_numpy=True
            )
            
            output_file = os.path.join(self.output_path, "embedding_matrix.npy")
            np.save(output_file, embeddings)
            logger.info(f"Saved embeddings ({embeddings.shape}) to {output_file}")
            return embeddings
        except Exception as e:
            logger.error(f"Error during paper embedding: {e}", exc_info=True)
            return None