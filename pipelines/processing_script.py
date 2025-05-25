import os
from datetime import datetime
from data.s3_data_fetcher import S3DataFetcher
from data.streaming_feature_engineering import stream_enhance_features
from data.StreamingHFTDataset import create_streaming_dataset
import logging
import sys
from typing import Optional
import traceback

# comment to  trigger CI process change

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_env_var(var_name: str, default: Optional[str] = None) -> str:
    """Safely get environment variable with logging."""
    value = os.environ.get(var_name, default)
    if value is None:
        raise ValueError(f"Required environment variable {var_name} is not set")
    logger.info(f"Environment variable {var_name}: {value}")
    return value

def main():
    # Get date parameters from environment
    start_date = os.environ.get('START_DATE')
    end_date = os.environ.get('END_DATE')
    
    if not start_date or not end_date:
        raise ValueError("START_DATE and END_DATE must be provided")

    input_path = "/opt/ml/processing/input"
    output_path = "/opt/ml/processing/output"
    
    logger.info(f"Starting data processing job")
    logger.info(f"Processing data from {start_date} to {end_date}")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    
    # Initialize S3 data fetcher
    s3_fetcher = S3DataFetcher(
        bucket_name=os.environ['SOURCE_BUCKET'],
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )
    
    # Create data stream from S3 input
    logger.info("Initializing S3 data stream")
    raw_data_stream = s3_fetcher.stream_data(start_date, end_date)
    
    # Apply feature engineering
    logger.info("Applying feature engineering")
    enhanced_stream = stream_enhance_features(raw_data_stream)
    
    # Create and save dataset
    logger.info("Creating streaming dataset")
    dataset = create_streaming_dataset(
        data_stream=enhanced_stream,
        sequence_length=60,
        s3_output_config={
            'base_path': output_path
        }
    )
    
    # Process the stream
    logger.info("Processing and saving data stream")
    records_processed = 0
    for _ in dataset:  # This will process and save the data
        records_processed += 1
        if records_processed % 1000 == 0:
            logger.info(f"Processed {records_processed} records")
    
    logger.info(f"Processing complete. Total records processed: {records_processed}")

if __name__ == "__main__":
    main()
