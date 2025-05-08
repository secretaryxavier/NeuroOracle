# cluster.py
import os
import json
import numpy as np
import umap # type: ignore # UMAP doesn't have official stubs readily available for all versions
import hdbscan # type: ignore # HDBSCAN might not have official stubs
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Optional, Tuple
from utils import ensure_output_path_exists

# Configure logger for this module
logger = logging.getLogger(__name__)

class SemanticClusterer:
    """
    Performs semantic clustering using UMAP for dimensionality reduction
    and HDBSCAN for clustering.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the SemanticClusterer.

        Args:
            config_path: Path to the JSON configuration file.
        """
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.output_path: Optional[str] = self.config.get("output_path")
        
        # UMAP parameters from config with defaults
        self.umap_n_neighbors: int = self.config.get("umap_n_neighbors", 15)
        self.umap_min_dist: float = self.config.get("umap_min_dist", 0.1)
        self.umap_metric: str = self.config.get("umap_metric", 'cosine')
        
        # HDBSCAN parameters from config with defaults
        self.hdbscan_min_cluster_size: int = self.config.get("hdbscan_min_cluster_size", 10)

        if not self.output_path:
            msg = "output_path missing in config."
            logger.error(msg)
            raise ValueError(msg)
        ensure_output_path_exists(self.output_path)


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

    def reduce_and_cluster(self, plot_clusters: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Loads embeddings, performs UMAP dimensionality reduction, and HDBSCAN clustering.
        Optionally saves a plot of the clusters.

        Args:
            plot_clusters: Whether to generate and save a scatter plot of the clusters.

        Returns:
            A tuple containing UMAP coordinates and cluster labels, or None if an error occurs.
        """
        embedding_matrix_path = os.path.join(self.output_path, "embedding_matrix.npy")
        logger.info("Running UMAP + HDBSCAN clustering...")

        try:
            X: np.ndarray = np.load(embedding_matrix_path)
            logger.info(f"Loaded embedding matrix from {embedding_matrix_path} with shape {X.shape}")
        except FileNotFoundError:
            logger.error(f"Embedding matrix not found: {embedding_matrix_path}. Cannot perform clustering.", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error loading embedding matrix: {e}", exc_info=True)
            return None

        if X.shape[0] < self.umap_n_neighbors: # Basic check for UMAP n_neighbors
             logger.warning(f"Number of samples ({X.shape[0]}) is less than n_neighbors ({self.umap_n_neighbors}). Adjust UMAP parameters or get more data.")
             # UMAP might still run or error; specific handling might be needed.
             # For now, we'll let it try and catch potential errors.

        try:
            logger.info(f"Performing UMAP reduction (n_neighbors={self.umap_n_neighbors}, min_dist={self.umap_min_dist}, metric='{self.umap_metric}')...")
            reducer = umap.UMAP(
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                metric=self.umap_metric,
                random_state=42 # for reproducibility
            )
            X_umap: np.ndarray = reducer.fit_transform(X)
            logger.info(f"UMAP reduction complete. Reduced shape: {X_umap.shape}")

            logger.info(f"Performing HDBSCAN clustering (min_cluster_size={self.hdbscan_min_cluster_size})...")
            clusterer_hdbscan = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                metric='euclidean' # HDBSCAN often works well with Euclidean on UMAP output
            )
            cluster_labels: np.ndarray = clusterer_hdbscan.fit_predict(X_umap)
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            logger.info(f"HDBSCAN clustering complete. Found {num_clusters} clusters (excluding noise points).")

            umap_coords_path = os.path.join(self.output_path, "umap_coords.npy")
            np.save(umap_coords_path, X_umap)
            logger.info(f"Saved UMAP coordinates to {umap_coords_path}")

            cluster_results_path = os.path.join(self.output_path, "cluster_results.json")
            with open(cluster_results_path, "w", encoding="utf-8") as f:
                json.dump({"labels": cluster_labels.tolist()}, f, indent=4)
            logger.info(f"Saved cluster labels to {cluster_results_path}")

            if plot_clusters:
                self._plot_clusters(X_umap, cluster_labels)
            
            logger.info("Clustering complete and results saved.")
            return X_umap, cluster_labels

        except Exception as e:
            logger.error(f"Error during UMAP/HDBSCAN processing: {e}", exc_info=True)
            return None

    def _plot_clusters(self, X_umap: np.ndarray, cluster_labels: np.ndarray) -> None:
        """Helper method to plot and save cluster visualization."""
        logger.info("Generating cluster plot...")
        try:
            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                x=X_umap[:, 0],
                y=X_umap[:, 1],
                hue=cluster_labels,
                palette="Spectral", # Using a palette good for categorical data
                s=30,
                alpha=0.7,
                legend='full'
            )
            plt.title("Semantic Clustering of Research Papers (UMAP + HDBSCAN)")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plot_path = os.path.join(self.output_path, "semantic_clusters.png")
            plt.savefig(plot_path)
            plt.close() # Close plot to free memory
            logger.info(f"Cluster plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error generating cluster plot: {e}", exc_info=True)