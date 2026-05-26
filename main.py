import os
import nltk
from src.utils.config_loader import load_config, logger
from src.data_collection.data_loader import run_data_collection
from src.data_cleaning.cleaner import run_data_cleaning
from src.feature_engineering.engineer import run_feature_engineering
from src.analysis.eda import run_analysis
from src.models.train_models import run_model_training
from reports.generate_pdf_report import generate_report

def main():
    """Main orchestrator for the Indian Startup Intelligence & Analytics Platform."""
    logger.info("=" * 60)
    logger.info("Initializing Indian Startup Ecosystem Analysis Pipeline...")
    logger.info("=" * 60)
    
    # 1. Load config and setup directory structure
    config = load_config()
    
    # 2. Download NLTK VADER lexicons quietly
    try:
        nltk.download("vader_lexicon", quiet=True)
        logger.info("NLTK VADER lexicon verified successfully.")
    except Exception as e:
        logger.warning(f"NLTK Lexicon download warning: {e}")
        
    # 3. Step-by-step pipeline execution
    try:
        logger.info("[PIPELINE STEP 1/6] Running Data Collection...")
        run_data_collection()
        
        logger.info("[PIPELINE STEP 2/6] Running Data Cleaning & Normalizations...")
        run_data_cleaning()
        
        logger.info("[PIPELINE STEP 3/6] Running Feature Engineering...")
        run_feature_engineering()
        
        logger.info("[PIPELINE STEP 4/6] Running Statistical Analytics & Static Visualizations...")
        run_analysis()
        
        logger.info("[PIPELINE STEP 5/6] Training & Serializing Machine Learning Models...")
        run_model_training()
        
        logger.info("[PIPELINE STEP 6/6] Generating Executive Intelligence Report...")
        report_path = generate_report()
        
        logger.info("=" * 60)
        logger.info("Pipeline executed successfully and all artifacts built!")
        logger.info(f"Generated Executive Intelligence Brief: {report_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed during execution: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
