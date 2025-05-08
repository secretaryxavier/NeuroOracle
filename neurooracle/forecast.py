# forecast.py (Complete with visualization fix AND expanded stop words)
import os
import json
import pandas as pd
from bertopic import BERTopic # type: ignore
# Import CountVectorizer for stop words
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, Any, List, Optional
import logging
# Assuming ensure_output_path_exists is in your utils.py file
from utils import ensure_output_path_exists

# Configure logger for this module
logger = logging.getLogger(__name__)

# Define a minimum number of documents BERTopic needs to run meaningfully
MIN_DOCS_FOR_BERTOPIC = 5 # You can adjust this threshold

class TrendForecaster:
    """
    Analyzes publication trends using BERTopic for topic modeling.
    Reads metadata (expects 'title', 'year' columns) from a CSV file specified in config.
    Generates BERTopic visualizations (barchart, topics over time).
    Includes enhanced stop word list.
    """
    def __init__(self, config_path: str = "config.json"):
        """Initializes the TrendForecaster."""
        self.config: Dict[str, Any] = self._load_config(config_path)
        opencitations_base_path: Optional[str] = self.config.get("opencitations_path")
        self.output_path: Optional[str] = self.config.get("output_path")
        self.top_n_topics_barchart: int = self.config.get("bertopic_top_n_topics", 10)
        self.default_year_imputation: int = self.config.get("bertopic_default_year", 2020)

        if not opencitations_base_path or not self.output_path:
            msg = "opencitations_path or output_path missing in config."
            logger.error(msg)
            raise ValueError(msg)

        self.meta_path: str = os.path.join(opencitations_base_path, "opencitations_meta.csv")
        ensure_output_path_exists(self.output_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration from a JSON file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config_data
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}", exc_info=True)
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from configuration file: {config_path}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}", exc_info=True)
            raise

    def _create_empty_bertopic_outputs(self, reason: str) -> None:
        """Helper to create empty placeholder files if BERTopic is skipped or fails."""
        logger.warning(f"Skipping BERTopic visualizations. Reason: {reason}")
        if not self.output_path:
             logger.error("Cannot create empty outputs: output_path not set.")
             return
        try:
            barchart_path = os.path.join(self.output_path, "topic_barchart.html")
            with open(barchart_path, "w", encoding="utf-8") as f:
                f.write(f"<html><body>BERTopic barchart not generated: {reason}</body></html>")
            logger.info(f"Empty topic barchart placeholder saved to {barchart_path}")

            trends_path = os.path.join(self.output_path, "topic_trends.html")
            with open(trends_path, "w", encoding="utf-8") as f:
                f.write(f"<html><body>BERTopic trends not generated: {reason}</body></html>")
            logger.info(f"Empty topic trends placeholder saved to {trends_path}")
        except Exception as e:
            logger.error(f"Error creating empty BERTopic output files: {e}", exc_info=True)

    def analyze_trends(self) -> None:
        """
        Loads metadata, performs topic modeling over time using BERTopic (with custom stop words),
        and saves visualizations (barchart, trends over time).
        """
        logger.info("Running topic modeling with BERTopic for trend analysis...")
        if not self.output_path:
             logger.error("Cannot analyze trends: output_path not configured.")
             return

        # --- Load Metadata ---
        try:
            metadata_cols = ['title', 'year']
            # Ensure correct dtype inference, especially for year
            metadata = pd.read_csv(self.meta_path, usecols=metadata_cols, 
                                   dtype={'title': str, 'year': 'Int64'}) # Use nullable Int64 for year
            logger.info(f"Loaded metadata for trend analysis from {self.meta_path} ({len(metadata)} rows)")
        # ... (Keep error handling for file loading as before) ...
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {self.meta_path}. Cannot perform trend analysis.", exc_info=True)
            self._create_empty_bertopic_outputs(f"Metadata file not found at {self.meta_path}")
            return
        except pd.errors.EmptyDataError:
            logger.warning(f"Metadata file is empty: {self.meta_path}. Cannot perform trend analysis.")
            self._create_empty_bertopic_outputs(f"Metadata file empty at {self.meta_path}")
            return
        except ValueError as e: # Handles if usecols are not in the CSV or dtype error
            logger.error(f"Error loading metadata columns or converting types from {self.meta_path}: {e}. Ensure 'title'(str) and 'year'(int) columns exist.", exc_info=True)
            self._create_empty_bertopic_outputs(f"Missing/invalid required columns ('title', 'year') in metadata from {self.meta_path}")
            return
        except Exception as e:
            logger.error(f"Error loading metadata for trend analysis: {e}", exc_info=True)
            self._create_empty_bertopic_outputs(f"Error loading metadata: {e}")
            return

        # --- Prepare Documents and Timestamps ---
        documents: List[str] = metadata["title"].fillna("").astype(str).tolist()
        # Handle conversion/imputation for year carefully
        try:
             # Fill NaNs (which shouldn't exist if Int64 was used, but belt-and-suspenders)
             years_filled = metadata["year"].fillna(self.default_year_imputation)
             # Convert to list of standard Python ints
             years: List[int] = [int(y) for y in years_filled]
        except Exception as e_year:
            logger.error(f"Fatal error converting 'year' column to integer list: {e_year}", exc_info=True)
            self._create_empty_bertopic_outputs(f"Fatal error processing 'year' column: {e_year}")
            return

        # --- Check Document Count ---
        if not documents or len(documents) < MIN_DOCS_FOR_BERTOPIC:
            reason = f"Not enough documents ({len(documents)}) for BERTopic. Minimum required: {MIN_DOCS_FOR_BERTOPIC}."
            logger.warning(reason)
            self._create_empty_bertopic_outputs(reason)
            logger.info("Trend analysis step completed (BERTopic skipped due to insufficient data).")
            return

        # --- Define Custom Stop Words ---
        logger.info("Defining custom stop words for BERTopic...")
        academic_stop_words = [ # <<< YOUR EXPANDED LIST GOES HERE >>>
            'et', 'al', 'e.g', 'i.e', 'etc', 'figure', 'fig', 'table', 'supplementary',
            'introduction', 'methods', 'method', 'results', 'result', 'discussion', 'data',
            'conclusions', 'conclusion', 'abstract', 'summary', 'background', 'purpose', 'objective',
            'appendix', 'acknowledgments', 'references', 'bibliography', 'materials',
            'study', 'studies', 'paper', 'papers', 'research', 'article', 'articles', 'report', 'reports',
            'analysis', 'approach', 'model', 'system', 'review', 'literature', 'section',
            'however', 'therefore', 'although', 'furthermore', 'moreover', 'addition', 'thus', 'hence',
            'based', 'using', 'used', 'shown', 'found', 'suggests', 'indicate', 'indicated',
            'demonstrates', 'provides', 'provided', 'reported', 'described', 'related', 'associated',
            'compared', 'different', 'similar', 'various', 'within', 'without', 'well',
            'among', 'between', 'given', 'taken', 'certain', 'specific', 'general',
            'potential', 'possible', 'likely', 'significant', 'significantly', 'important', 'key', 'main',
            'first', 'second', 'third', 'one', 'two', 'three', 'four', 'five', 'high', 'low', 'level', 'levels',
            'number', 'group', 'groups', 'case', 'cases', 'example', 'examples',
            'doi', 'https', 'http', 'www', 'org', 'com', 'pdf', 'html', 'xml', 'gov', 'edu',
            'copyright', 'license', 'rights', 'reserved', 'author', 'authors',
            'university', 'institute', 'department', 'inc', 'llc', 'corp',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december', 'published', 'received',
            'accepted', 'available', 'online', 'print', 'preprint', 'journal', 'volume', 'vol',
            'issue', 'pages', 'pp', 'editor', 'edited', 'version', 'appendix',
            'increase', 'decrease', 'effect', 'effects', 'impact', 'value', 'values', 'present',
            'suggested', 'observed', 'performed', 'conducted', 'analyzed', 'included', 'excluded',
            'required', 'obtained', 'developed', 'proposed', 'tested', 'validated',
            # --- ADD/REMOVE WORDS BASED ON YOUR DOMAIN & PREVIOUS RESULTS ---
        ]
        try:
            # Combine with default 'english' stop words
            vectorizer_model_base = CountVectorizer(stop_words='english')
            default_stopwords = list(vectorizer_model_base.get_stop_words())
            custom_stopwords = list(set(default_stopwords + academic_stop_words)) # Use set for uniqueness
            # Initialize vectorizer with the combined list and n-grams
            vectorizer_model = CountVectorizer(stop_words=custom_stopwords, ngram_range=(1, 2)) # Use 1 and 2-word phrases
            logger.info(f"Using CountVectorizer with {len(custom_stopwords)} total stop words.")
        except Exception as e_stopwords:
            logger.error(f"Failed to create custom stop word list: {e_stopwords}. Using default 'english'.")
            vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2)) # Fallback

        # --- Run BERTopic ---
        try:
            # Initialize BERTopic WITH the custom vectorizer
            topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)

            logger.info(f"Fitting BERTopic model on {len(documents)} documents using custom stop words...")
            topics, probs = topic_model.fit_transform(documents)
            logger.info("BERTopic model fitting complete.")

            # --- Calculate and Visualize Topics Over Time (Corrected Logic) ---
            logger.info("Calculating topic frequencies over time...")
            topics_over_time_df = None
            if years and len(years) == len(documents):
                 try:
                     topics_over_time_df = topic_model.topics_over_time(
                         docs=documents,
                         timestamps=years,
                         nr_bins=20 # Optional: Adjust number of time bins
                     )
                     logger.info("Topic frequency calculation complete.")
                 except Exception as e_calc:
                     logger.error(f"Error calculating topics over time: {e_calc}", exc_info=True)
                     self._create_empty_bertopic_outputs(f"Error calculating topics over time: {e_calc}")
            else:
                logger.warning("Years data is missing or mismatched. Cannot calculate topics over time.")

            # Visualize Barchart
            barchart_path = os.path.join(self.output_path, "topic_barchart.html")
            logger.info(f"Generating topic barchart...")
            try:
                fig_barchart = topic_model.visualize_barchart(top_n_topics=self.top_n_topics_barchart)
                if fig_barchart:
                     fig_barchart.write_html(barchart_path)
                     logger.info(f"Topic barchart saved to {barchart_path}")
                else:
                     logger.warning("Could not generate topic barchart visualization (returned None).")
                     self._create_empty_bertopic_outputs("Failed to generate barchart figure.")
            except Exception as e_bar:
                 logger.error(f"Error generating topic barchart: {e_bar}", exc_info=True)
                 self._create_empty_bertopic_outputs(f"Error generating barchart: {e_bar}")

            # Visualize Trends
            trends_path = os.path.join(self.output_path, "topic_trends.html")
            if topics_over_time_df is not None and not topics_over_time_df.empty:
                logger.info(f"Generating topic trends over time visualization...")
                try:
                    fig_trends = topic_model.visualize_topics_over_time(topics_over_time_df)
                    if fig_trends:
                        fig_trends.write_html(trends_path)
                        logger.info(f"Topic trends over time saved to {trends_path}")
                    else:
                        logger.warning("Could not generate topic trends visualization (returned None).")
                        self._create_empty_bertopic_outputs("Failed to generate trends figure.")
                except Exception as e_trends:
                     logger.error(f"Error generating topics over time visualization: {e_trends}", exc_info=True)
                     self._create_empty_bertopic_outputs(f"Error generating trends plot: {e_trends}")
            else:
                 logger.warning("Skipping topic trends visualization (data not available).")
                 self._create_empty_bertopic_outputs("Topics over time data not available.")

            logger.info("Trend analysis processing complete. Results saved to output directory.")

        except Exception as e:
            reason = f"Error during BERTopic processing: {e}"
            logger.error(reason, exc_info=True)
            self._create_empty_bertopic_outputs(reason)
            logger.info("Trend analysis step completed (BERTopic processing failed).")