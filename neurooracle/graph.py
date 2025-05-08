# graph.py
import os
import json
import pandas as pd
import networkx as nx
import logging
from typing import Dict, Any, Optional
from utils import ensure_output_path_exists
import pickle # <<<<<<<<<<<<<<<<<<<<<<< ADD THIS IMPORT

# Configure logger for this module
logger = logging.getLogger(__name__)

class CitationGraphBuilder:
    def __init__(self, config_path: str = "config.json"):
        # ... (your __init__ method stays the same) ...
        self.config: Dict[str, Any] = self._load_config(config_path)
        opencitations_base_path: Optional[str] = self.config.get("opencitations_path")
        self.output_path: Optional[str] = self.config.get("output_path")

        if not opencitations_base_path or not self.output_path:
            msg = "opencitations_path or output_path missing in config."
            logger.error(msg)
            raise ValueError(msg)

        self.index_path: str = os.path.join(opencitations_base_path, "index.csv")
        ensure_output_path_exists(self.output_path)


    def _load_config(self, config_path: str) -> Dict[str, Any]:
        # ... (your _load_config method stays the same) ...
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}", exc_info=True)
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from configuration file: {config_path}", exc_info=True)
            raise

    def build_graph(self) -> Optional[nx.DiGraph]:
        logger.info("Building citation graph from OpenCitations data...")
        try:
            # For very large CSVs, consider chunking or specifying dtypes/usecols
            df = pd.read_csv(self.index_path, usecols=['citing', 'cited']) # Assuming these are the correct columns
        except FileNotFoundError:
            logger.error(f"OpenCitations index file not found: {self.index_path}", exc_info=True)
            return None
        except pd.errors.EmptyDataError:
            logger.warning(f"OpenCitations index file is empty: {self.index_path}. Cannot build graph.")
            return None
        except KeyError as e:
            logger.error(f"Missing expected columns ('citing', 'cited') in {self.index_path}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error loading citation index data: {e}", exc_info=True)
            return None

        if df.empty:
            logger.warning(f"No data in {self.index_path} to build graph.")
            return None

        logger.info(f"Creating graph from {len(df)} citation links.")
        try:
            G: nx.DiGraph = nx.from_pandas_edgelist(df, source='citing', target='cited', create_using=nx.DiGraph())

            graph_output_file = os.path.join(self.output_path, "citation_graph.gpickle")

            # ----- OLD LINE TO BE REPLACED -----
            # nx.write_gpickle(G, graph_output_file) 

            # ----- NEW LINES TO SAVE USING PICKLE -----
            with open(graph_output_file, 'wb') as f_out: # 'wb' for write binary
                pickle.dump(G, f_out)
            # -------------------------------------------

            logger.info(f"Citation graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges saved to {graph_output_file}")
            return G
        except Exception as e:
            logger.error(f"Error building or saving citation graph: {e}", exc_info=True)
            return None