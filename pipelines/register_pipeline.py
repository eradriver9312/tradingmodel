import boto3
import logging
from sagemaker_data_pipeline import create_data_preparation_pipeline
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Register or update the SageMaker pipeline definition."""
    try:
        # Get role ARN from environment
        role_arn = os.getenv('SAGEMAKER_ROLE')
        if not role_arn:
            raise ValueError("SAGEMAKER_ROLE environment variable is required")

        # Create pipeline instance
        pipeline = create_data_preparation_pipeline()
        
        # Upsert the pipeline
        pipeline.upsert(
            role_arn=role_arn,  # Use the role ARN from environment variable
            description="HFT data preparation and training pipeline"
        )
        
        logger.info(f"Successfully registered pipeline: {pipeline.name}")
        logger.info(f"Pipeline ARN: {pipeline.arn}")
        
        # Output the pipeline definition for debugging
        logger.info("Pipeline definition:")
        logger.info(pipeline.definition())
        
    except Exception as e:
        logger.error(f"Failed to register pipeline: {e}")
        raise

if __name__ == "__main__":
    main()