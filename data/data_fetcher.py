"""
Data retrieval from the Polygon.io API with robust error handling and rate limiting.
"""
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
from tqdm import tqdm
from config import logger

class PolygonDataFetcher:
    """Handles data retrieval from the Polygon.io API with robust error handling and rate limiting."""
    
    BASE_URL = "https://api.polygon.io/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.validation_metrics = []

    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
        """Make an API request with retry logic and rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 429:  # Rate limit exceeded
                    wait_time = min(60 * (attempt + 1), 300)  # Max 5 minutes
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                return data
                
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Request failed after {retries} attempts: {e}")
                    raise
                logger.warning(f"Request failed (Attempt {attempt + 1}/{retries}): {e}. Retrying in 30 seconds...")
                time.sleep(30)
                
        raise Exception(f"Failed after {retries} attempts")

    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str, 
                        timespan: str = "minute", multiplier: int = 1) -> pd.DataFrame:
        """
        Fetch aggregate stock data for a specific ticker and date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timespan: Timespan of the aggregates ('minute', 'hour', 'day')
            multiplier: The size of the timespan multiplier
            
        Returns:
            DataFrame containing aggregated stock data
        """
        endpoint = f"aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        all_results = []

        with tqdm(desc=f"Fetching {symbol}", unit='rows') as pbar:
            while True:
                data = self._make_request(endpoint, params)
                
                if 'results' in data:
                    if len(data['results']) > 0 and 'c' not in data['results'][0]:
                        logger.error(f"Data for {symbol} does not contain 'c' (close) field.")
                        return pd.DataFrame()

                    all_results.extend(data['results'])
                    pbar.update(len(data['results']))
                    
                if len(data.get('results', [])) < 50000:
                    break
                    
                # Update start_date for next request
                last_timestamp = data['results'][-1]['t']
                start_date = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d')
                params['from'] = start_date

        # Create DataFrame and process data
        if not all_results:
            logger.warning(f"No data retrieved for {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_results)
        
        if 't' in df.columns:
            # Remove duplicate timestamps
            duplicates_before = len(df) - df.drop_duplicates(subset=['t'], keep='last').shape[0]
            if duplicates_before > 0:
                logger.info(f"Removed {duplicates_before} duplicate timestamps for {symbol}.")
            df = df.drop_duplicates(subset=['t'], keep='last')
        else:
            logger.error("'t' column not found in fetched data.")
            return pd.DataFrame()

        # Rename columns for clarity
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'n': 'transactions'
        })

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df
        
    def fetch_market_index_data(self, index_symbol: str = "SPY", 
                             start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch market index data to add market context."""
        if not start_date or not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
        return self.fetch_stock_data(index_symbol, start_date, end_date)