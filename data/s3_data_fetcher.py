"""
S3 data fetcher with streaming capabilities for handling large datasets efficiently.
"""
import boto3
import pandas as pd
import io
import gzip
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from config import logger

# testing directory change to activate CI actions

class S3DataFetcher:
    def __init__(self, bucket_name: str, region_name: str = 'us-east-1', chunk_size: int = 100000):
        """
        Initialize S3 client and configuration.
        
        Args:
            bucket_name: Name of the S3 bucket
            region_name: AWS region
            chunk_size: Number of rows to process at a time
        """
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.bucket = bucket_name
        self.chunk_size = chunk_size

    def _get_file_paths(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Generate list of S3 file paths based on date range."""
        file_paths = []
        current_date = start_date
        
        while current_date <= end_date:
            year = current_date.strftime('%Y')
            month = current_date.strftime('%m')
            file_path = f"{year}/{month}/{current_date.strftime('%Y-%m-%d')}.csv.gz"
            file_paths.append(file_path)
            current_date += timedelta(days=1)
            
        return file_paths

    def _stream_s3_file(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a single S3 file in chunks.
        
        Args:
            file_path: Path to file in S3
            
        Yields:
            DataFrame chunks
        """
        try:
            # Get the object from S3
            response = self.s3_client.get_object(Bucket=self.bucket, Key=file_path)
            
            # Create a streaming body for the compressed file
            streaming_body = response['Body']
            
            # Create a gzip reader that decompresses in memory efficiently
            with gzip.GzipFile(fileobj=streaming_body) as gz:
                # Create a text stream from the decompressed data
                text_stream = io.TextIOWrapper(gz, encoding='utf-8')
                
                # Read the CSV in chunks
                for chunk in pd.read_csv(
                    text_stream,
                    chunksize=self.chunk_size,
                    parse_dates=['timestamp']  # Adjust column names as needed
                ):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error streaming file {file_path}: {e}")
            yield pd.DataFrame()  # Return empty DataFrame on error

    def stream_data(self, 
                    start_date: str, 
                    end_date: str) -> Generator[pd.DataFrame, None, None]:
        """
        Stream data from S3 for the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Yields:
            DataFrame chunks containing the data
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        file_paths = self._get_file_paths(start_dt, end_dt)
        
        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            for chunk in self._stream_s3_file(file_path):
                if not chunk.empty:
                    yield chunk

    def process_data_stream(self,
                          start_date: str,
                          end_date: str,
                          process_chunk_fn: callable) -> Generator[pd.DataFrame, None, None]:
        """
        Stream and process data with a custom processing function.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            process_chunk_fn: Function to process each chunk of data
            
        Yields:
            Processed DataFrame chunks
        """
        for chunk in self.stream_data(start_date, end_date):
            processed_chunk = process_chunk_fn(chunk)
            yield processed_chunk