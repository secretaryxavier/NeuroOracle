# write.py
import os
import json
import numpy as np
import faiss # type: ignore # FAISS might not have official stubs
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
from utils import load_ocr_documents, ensure_output_path_exists # Assuming utils.py is in the same directory or Python path

# Configure logger for this module
logger = logging.getLogger(__name__)

class CitationAssistant:
    """
    Provides citation suggestions based on semantic similarity using FAISS.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the CitationAssistant.

        Args:
            config_path: Path to the JSON configuration file.
        """
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.ocr_path: Optional[str] = self.config.get("ocr_data_path")
        self.output_path: Optional[str] = self.config.get("output_path")
        self.embedding_model_name: Optional[str] = self.config.get("embedding_model")
        self.top_k_suggestions: int = self.config.get("faiss_top_k", 5)

        if not all([self.ocr_path, self.output_path, self.embedding_model_name]):
            msg = "ocr_data_path, output_path, or embedding_model missing in config."
            logger.error(msg)
            raise ValueError(msg)

        ensure_output_path_exists(self.output_path)
        
        self.embedding_matrix_path: str = os.path.join(self.output_path, "embedding_matrix.npy")
        self.model: Optional[SentenceTransformer] = None
        self.embedding_matrix: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self.index: Optional[faiss.Index] = None

        self._initialize_resources()

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

    def _initialize_resources(self) -> None:
        """Loads models, embeddings, documents, and builds FAISS index."""
        logger.info("Initializing Citation Assistant resources...")
        try:
            logger.info(f"Loading sentence transformer model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            
            logger.info(f"Loading embedding matrix from: {self.embedding_matrix_path}")
            self.embedding_matrix = np.load(self.embedding_matrix_path)
            
            logger.info(f"Loading documents from: {self.ocr_path}")
            self.documents = load_ocr_documents(self.ocr_path)

            if self.embedding_matrix is None or len(self.documents) == 0 or \
               self.embedding_matrix.shape[0] != len(self.documents):
                msg = "Mismatch between embeddings and documents, or resources are empty. Cannot build FAISS index."
                logger.error(msg)
                # Decide if this should raise an error or if the assistant can be partially active
                raise RuntimeError(msg)

            logger.info(f"Building FAISS index (IndexFlatL2) with {self.embedding_matrix.shape[0]} vectors.")
            self.index = faiss.IndexFlatL2(self.embedding_matrix.shape[1])
            self.index.add(self.embedding_matrix.astype(np.float32)) # FAISS often expects float32
            logger.info("FAISS index built successfully.")
            
        except FileNotFoundError as e:
            logger.error(f"Required file not found during initialization: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error initializing Citation Assistant resources: {e}", exc_info=True)
            raise


    def suggest_citations(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Suggests relevant citations for a given query text.

        Args:
            query_text: The text to find relevant citations for.
            top_k: Number of suggestions to return. Defaults to config value.

        Returns:
            A list of dictionaries, each containing rank and text snippet.
        """
        if not self.model or self.index is None or not self.documents:
            logger.error("Citation Assistant is not properly initialized. Cannot suggest citations.")
            return []
        
        k = top_k if top_k is not None else self.top_k_suggestions
        logger.info(f"Searching for top-{k} relevant papers for query: '{query_text[:100]}...'")

        try:
            query_vec: np.ndarray = self.model.encode([query_text], convert_to_numpy=True)
            distances, indices = self.index.search(query_vec.astype(np.float32), k)
            
            results: List[Dict[str, Any]] = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.documents): # Ensure index is valid
                    results.append({
                        "rank": i + 1,
                        "text_preview": self.documents[idx][:300] + "...", # Preview
                        "distance": float(distances[0][i]),
                        # "full_text_index": int(idx) # Optionally return index for full retrieval
                    })
            logger.info(f"Found {len(results)} citation suggestions.")
            return results
        except Exception as e:
            logger.error(f"Error during citation suggestion: {e}", exc_info=True)
            return []

    def initialize_interface(self) -> None:
        """
        Starts an interactive command-line interface for citation suggestions.
        """
        if not self.model or self.index is None:
            logger.critical("Citation Assistant cannot start: resources not initialized.")
            print("Citation Assistant failed to initialize. Check logs for details.")
            return

        print("\n--- Citation Assistant Ready ---")
        print("Enter a paragraph of text to get relevant citation suggestions.")
        print("Type 'exit' or 'quit' to close.")
        
        while True:
            try:
                text: str = input("\nEnter paragraph (or type 'exit'/'quit'): ")
                if text.strip().lower() in ["exit", "quit"]:
                    logger.info("Exiting Citation Assistant interface.")
                    break
                if not text.strip():
                    continue

                citations: List[Dict[str, Any]] = self.suggest_citations(text)
                if citations:
                    print("\nSuggested Citations:")
                    for c in citations:
                        print(f"\n[{c['rank']}] (Distance: {c['distance']:.4f})\n {c['text_preview']}")
                else:
                    print("No relevant citations found for your query.")
            except KeyboardInterrupt:
                logger.info("Citation Assistant interface interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {e}", exc_info=True)
                print("An unexpected error occurred. Please check logs.")