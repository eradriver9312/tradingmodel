import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker_data_pipeline import create_data_preparation_pipeline
import logging
import sys
import os
from typing import Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file if it exists."""
    load_dotenv()
    
    # Log environment configuration (excluding sensitive values)
    logger.info("Environment Configuration:")
    logger.info(f"SOURCE_BUCKET: {os.getenv('SOURCE_BUCKET')}")
    logger.info(f"SOURCE_PREFIX: {os.getenv('SOURCE_PREFIX', 'raw_data')}")
    logger.info(f"DEST_BUCKET: {os.getenv('DEST_BUCKET')}")
    logger.info(f"DEST_PREFIX: {os.getenv('DEST_PREFIX', 'processed_data')}")
    logger.info(f"AWS_REGION: {os.getenv('AWS_REGION', 'us-east-1')}")

def initialize_sagemaker_client():
    """Initialize SageMaker client with proper error handling."""
    try:
        session = boto3.session.Session(region_name=os.getenv('AWS_REGION', 'us-east-1'))
        return session.client('sagemaker')
    except Exception as e:
        logger.error(f"Failed to initialize SageMaker client: {str(e)}")
        raise

def main():
    """Main function to run the SageMaker pipeline."""
    try:
        # Load environment variables
        load_environment()
        
        # Initialize SageMaker client
        sagemaker_client = initialize_sagemaker_client()
        logger.info("Successfully initialized SageMaker client")
        
        # Create the pipeline
        logger.info("Creating data preparation pipeline")
        pipeline = create_data_preparation_pipeline()
        
        # Start the pipeline execution
        logger.info("Starting pipeline execution")
        execution = pipeline.start()
        
        # Log execution details
        logger.info(f"Pipeline execution started with ARN: {execution.arn}")
        logger.info(f"Pipeline execution status: {execution.describe()['PipelineExecutionStatus']}")
        
        return execution.arn

    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during pipeline execution: {str(e)}")
        raise
    finally:
        logger.info("Pipeline execution process completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)
