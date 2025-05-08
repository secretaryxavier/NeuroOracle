# main.py
import logging
import json
import os
from typing import Dict, Any

# Import your project modules
from ingest import CorpusIngester
from embed import TextEmbedder
from graph import CitationGraphBuilder
from cluster import SemanticClusterer
from write import CitationAssistant
from forecast import TrendForecaster
from utils import ensure_output_path_exists # Assuming this function is in your utils.py


def setup_logging(config_path: str = "config.json") -> None:
    """Sets up basic logging configuration."""
    log_level_str = "INFO" # Default log level
    log_file = None # Default log file path
    output_path_default = "output" # Default output directory if not in config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = json.load(f)
        log_level_str = config.get("log_level", "INFO").upper()
        
        output_path_from_config = config.get("output_path")
        if output_path_from_config:
            ensure_output_path_exists(output_path_from_config)
            log_file = config.get("log_file", os.path.join(output_path_from_config, "neuro_oracle.log"))
        else:
            ensure_output_path_exists(output_path_default)
            log_file = config.get("log_file", os.path.join(output_path_default, "neuro_oracle.log"))
        
        if log_file:
            log_file_dir = os.path.dirname(log_file)
            if log_file_dir: # Ensure directory for log_file exists if it's not in the current directory
                 ensure_output_path_exists(log_file_dir)

    except FileNotFoundError:
        print(f"Warning: Configuration file '{config_path}' not found. Using default logging settings.")
        ensure_output_path_exists(output_path_default) # Ensure default output path for logs
        log_file = os.path.join(output_path_default, "neuro_oracle.log") # Default log file
        if log_file: ensure_output_path_exists(os.path.dirname(log_file)) # Ensure log file directory
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from '{config_path}'. Using default logging settings.")
        ensure_output_path_exists(output_path_default)
        log_file = os.path.join(output_path_default, "neuro_oracle.log")
        if log_file: ensure_output_path_exists(os.path.dirname(log_file))
    except Exception as e: # Catch other potential errors during config loading for logging
        print(f"Warning: An unexpected error occurred loading logging config: {e}. Using defaults.")
        ensure_output_path_exists(output_path_default)
        log_file = os.path.join(output_path_default, "neuro_oracle.log")
        if log_file: ensure_output_path_exists(os.path.dirname(log_file))


    log_level = getattr(logging, log_level_str, logging.INFO)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()] # Log to console by default
    if log_file:
        try:
            # Ensure the directory for the log file exists one last time
            log_file_directory = os.path.dirname(log_file)
            if log_file_directory: # If log_file is not in the root of where script is run
                os.makedirs(log_file_directory, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode='a')) # Append mode
        except Exception as e:
            print(f"Warning: Could not set up file logging to {log_file}. Error: {e}")

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    logging.info("Logging configured.")
    # Quieten overly verbose libraries if necessary
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING) # Often used by sentence_transformers


def main_pipeline(config_path: str = "config.json") -> None:
    """
    Runs the main Neuro Oracle pipeline.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Neuro Oracle Pipeline ---")

    try:
        # 1. Ingestion
        logger.info("Step 1: Data Ingestion")
        ingester = CorpusIngester(config_path=config_path)
        ingester.load_ocr_data()
        ingester.load_opencitations_data()
        logger.info("Data ingestion complete.")

        # Step 2: Text Embedding
        logger.info("Step 2: Text Embedding")
        embedder = TextEmbedder(config_path=config_path)
        embedding_matrix = embedder.embed_papers()
        if embedding_matrix is None:
            logger.error("Embedding step failed or produced no embeddings. Subsequent steps might fail.")
        logger.info("Text embedding complete.")

        # Step 3: Citation Graph Building
        logger.info("Step 3: Citation Graph Building")
        graph_builder = CitationGraphBuilder(config_path=config_path)
        citation_graph = graph_builder.build_graph() # This line does the work
        if citation_graph is None: # This will log a warning if build_graph() returns None
            logger.warning("Citation graph building FAILED or produced no graph (build_graph returned None).")
        logger.info("Citation graph building complete. (Stopping here for third test)")

        # return # <<<<<<<<<<<<<<<<<<<<<<< STOPS EXECUTION AFTER STEP 3

        # Step 4: Clustering (Commented out or will not be reached)
        logger.info("Step 4: Semantic Clustering")
        clusterer = SemanticClusterer(config_path=config_path)
        cluster_results = clusterer.reduce_and_cluster(plot_clusters=True)
        if cluster_results is None:
             logger.warning("Clustering step failed.")
        logger.info("Semantic clustering complete.")

        # Step 5: Citation Assistant (Commented out or will not be reached)
        logger.info("Step 5: Initializing Citation Assistant")
        writer = CitationAssistant(config_path=config_path)
        logger.info("Citation Assistant initialized and ready (if interactive mode is called).")

        logger.info("Now starting Citation Assistant interactive interface...")
        writer.initialize_interface() # <<<<<<<<<<< UNCOMMENT OR ADD THIS LINE

        logger.info("Citation Assistant interface finished. (Stopping here for fifth test - interactive part)")

        # return # <<<<<<<<<<<<<<<<<<<<<<< ADD THIS NEW RETURN LINE TO STOP HERE

        # Step 6: Trend Forecasting (Commented out or will not be reached)
        logger.info("Step 6: Trend Forecasting")
        forecaster = TrendForecaster(config_path=config_path)
        forecaster.analyze_trends()
        logger.info("Trend forecasting complete.")

        logger.info("--- Neuro Oracle Pipeline Finished Successfully ---")

    except Exception as e:
        logger.critical(f"Neuro Oracle pipeline failed with an unhandled exception: {e}", exc_info=True)
        logger.info("--- Neuro Oracle Pipeline Finished With Errors ---")


if __name__ == "__main__":
    CONFIG_FILE = "config.json"
    setup_logging(config_path=CONFIG_FILE)
    main_pipeline(config_path=CONFIG_FILE)
    
    # Optional interactive part (still commented out)
    # main_logger = logging.getLogger(__name__)
    # main_logger.info("Starting interactive Citation Assistant (optional)...")
    # try:
    #     assistant = CitationAssistant(config_path=CONFIG_FILE)
    #     assistant.initialize_interface()
    # except Exception as e:
    #     main_logger.error(f"Failed to start interactive Citation Assistant: {e}", exc_info=True)