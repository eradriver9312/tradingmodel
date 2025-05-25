from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.pytorch import PyTorch
import boto3
import os
from typing import Dict
from sagemaker.workflow.parameters import ParameterString
from datetime import datetime, timedelta

# trigger ci update comment; ignore me. again. and again... and now yet again...

def get_env_config() -> Dict[str, str]:
    """Get configuration from environment variables with defaults."""
    config = {
        'SAGEMAKER_ROLE': os.getenv('SAGEMAKER_ROLE'),
        'RAW_DATA_BUCKET': 'ml-3bean-ts-raw-data',
        'TRAINING_DATA_BUCKET': 'sagemaker.training.trading.data',
        'MODEL_ARTIFACTS_BUCKET': 'ml-3bean-ts-model-artifacts',
        'RAW_DATA_PREFIX': os.getenv('RAW_DATA_PREFIX', 'raw_data'),
        'TRAINING_DATA_PREFIX': os.getenv('TRAINING_DATA_PREFIX', 'processed_data'),
        'MODEL_PREFIX': os.getenv('MODEL_PREFIX', 'models'),
        'AWS_REGION': os.getenv('AWS_REGION', 'us-east-2')
    }
    
    return config

def get_account_id():
    return boto3.client('sts').get_caller_identity()['Account']

def create_data_preparation_pipeline() -> Pipeline:
    """Create SageMaker pipeline using parameters."""
    # Get default dates (last 30 days by default)
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    default_start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    # Define pipeline parameters
    source_bucket_param = ParameterString(
        name="SourceBucket",
        default_value=os.getenv('SOURCE_BUCKET')
    )
    source_prefix_param = ParameterString(
        name="SourcePrefix",
        default_value=os.getenv('SOURCE_PREFIX', 'raw_data')
    )
    dest_bucket_param = ParameterString(
        name="DestBucket",
        default_value=os.getenv('DEST_BUCKET')
    )
    dest_prefix_param = ParameterString(
        name="DestPrefix",
        default_value=os.getenv('DEST_PREFIX', 'processed_data')
    )
    
    # Add date parameters
    start_date_param = ParameterString(
        name="StartDate",
        default_value=default_start_date
    )
    end_date_param = ParameterString(
        name="EndDate",
        default_value=default_end_date
    )
    
    config = get_env_config()
    account_id = get_account_id()
    
    processor = ScriptProcessor(
        command=['python3'],
        image_uri=f"{account_id}.dkr.ecr.{config['AWS_REGION']}.amazonaws.com/hft-processing:latest",
        role=config['SAGEMAKER_ROLE'],
        instance_count=1,
        instance_type='ml.m5.xlarge',
        base_job_name='hft-stream-processing',
        env={
            'SOURCE_BUCKET': source_bucket_param,
            'SOURCE_PREFIX': source_prefix_param,
            'DEST_BUCKET': dest_bucket_param,
            'DEST_PREFIX': dest_prefix_param,
            'START_DATE': start_date_param,
            'END_DATE': end_date_param
        }
    )
    
    processing_step = ProcessingStep(
        name="StreamingDataPreparation",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{config['RAW_DATA_BUCKET']}/{config['RAW_DATA_PREFIX']}",
                destination="/opt/ml/processing/input/data"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{config['TRAINING_DATA_BUCKET']}/{config['TRAINING_DATA_PREFIX']}"
            )
        ],
        code="pipelines/processing_script.py"
    )
    
    training_step = TrainingStep(
        name="ModelTraining",
        estimator=PyTorch(
            entry_point='sagemaker_trainer.py',
            source_dir='training',
            role=config['SAGEMAKER_ROLE'],
            instance_count=1,
            instance_type='ml.p3.2xlarge',  # GPU instance
            framework_version='2.0.1',
            py_version='py310',
            hyperparameters={
                'epochs': 100,
                'batch-size': 32,
                'learning-rate': 0.001,
                'hidden-size': 128,
                'num-layers': 2,
                'sequence-length': 60
            },
            output_path=f"s3://{config['MODEL_ARTIFACTS_BUCKET']}/{config['MODEL_PREFIX']}"
        ),
        inputs={
            'train': ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{config['TRAINING_DATA_BUCKET']}/train"
            ),
            'validation': ProcessingOutput(
                output_name="validation_data",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{config['TRAINING_DATA_BUCKET']}/validation"
            )
        }
    )
    
    pipeline = Pipeline(
        name="HFTDataPreparationPipeline",
        steps=[processing_step, training_step],
        parameters=[
            source_bucket_param,
            source_prefix_param,
            dest_bucket_param,
            dest_prefix_param,
            start_date_param,
            end_date_param
        ]
    )
    
    return pipeline
