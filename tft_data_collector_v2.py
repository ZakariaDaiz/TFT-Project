import os
import time
import pandas as pd
from riotwatcher import TftWatcher, ApiError
from flatten_json import flatten
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set
import logging
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tft_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()

API_KEY = os.environ.get('RIOT_API_KEY')
if not API_KEY:
    logger.error("RIOT_API_KEY environment variable not set!")
    raise ValueError("RIOT_API_KEY is required")

PLATFORM_REGION = 'euw1'
MATCH_REGION = 'europe'

# Performance tuning
MAX_WORKERS_PUUID = 15  # Increased for better parallelization
MAX_WORKERS_MATCHES = 20
MAX_WORKERS_MATCH_DATA = 5
BATCH_SIZE = 100  # Process in batches for better progress tracking

watcher = TftWatcher(API_KEY)

# --- Helper Functions ---

def get_challengers(region: str = PLATFORM_REGION) -> List[Dict]:
    """Fetch Challenger league entries."""
    logger.info(f"Fetching Challengers for {region}...")
    try:
        challengers = watcher.league.challenger(region)
        entries = challengers.get('entries', [])
        logger.info(f"Found {len(entries)} Challenger players")
        return entries
    except ApiError as err:
        logger.error(f"API Error fetching challengers: {err}")
        return []
    except Exception as err:
        logger.error(f"Unexpected error fetching challengers: {err}")
        return []


def get_gms(region: str = PLATFORM_REGION) -> List[Dict]:
    """Fetch Grandmaster league entries."""
    logger.info(f"Fetching Grandmasters for {region}...")
    try:
        gms = watcher.league.grandmaster(region)
        entries = gms.get('entries', [])
        logger.info(f"Found {len(entries)} Grandmaster players")
        return entries
    except ApiError as err:
        logger.error(f"API Error fetching grandmasters: {err}")
        return []
    except Exception as err:
        logger.error(f"Unexpected error fetching grandmasters: {err}")
        return []


def resolve_puuid_single(entry: Dict, region: str) -> Optional[str]:
    """Resolve a single PUUID with retry logic."""
    if 'puuid' in entry:
        return entry['puuid']
    
    summoner_id = entry.get('summonerId')
    if not summoner_id:
        return None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            summoner_dto = watcher.summoner.by_id(region, summoner_id)
            return summoner_dto.get('puuid')
        except ApiError as err:
            if err.response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.debug(f"Failed to resolve {summoner_id}: {err}")
                return None
        except Exception as err:
            logger.debug(f"Error resolving {summoner_id}: {err}")
            return None
    return None


def resolve_puuids(entries: List[Dict], region: str = PLATFORM_REGION) -> List[str]:
    """Resolve PUUIDs in parallel with progress tracking."""
    logger.info(f"Resolving PUUIDs for {len(entries)} players...")
    puuids = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PUUID) as executor:
        future_to_entry = {
            executor.submit(resolve_puuid_single, entry, region): entry 
            for entry in entries
        }
        
        completed = 0
        for future in as_completed(future_to_entry):
            result = future.result()
            if result:
                puuids.append(result)
            
            completed += 1
            if completed % 50 == 0 or completed == len(entries):
                logger.info(f"Resolved {completed}/{len(entries)} PUUIDs ({len(puuids)} successful)")
    
    logger.info(f"Successfully resolved {len(puuids)}/{len(entries)} PUUIDs")
    return puuids


def get_match_ids_single(puuid: str, region: str, count: int) -> List[str]:
    try:
        return watcher.match.by_puuid(region, puuid, count=count)
    except ApiError as err:
        logger.debug(f"API error for PUUID {puuid[:8]}...: {err}")
        return []
    except Exception as err:
        logger.debug(f"Error fetching matches for {puuid[:8]}...: {err}")
        return []


def get_match_ids(puuids: List[str], region: str = MATCH_REGION, count: int = 20) -> List[str]:
    """Fetch match IDs in parallel with deduplication."""
    logger.info(f"Fetching Match IDs for {len(puuids)} PUUIDs (up to {count} per player)...")
    all_match_ids: Set[str] = set()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_MATCHES) as executor:   
        future_to_puuid = {
            executor.submit(get_match_ids_single, puuid, region, count): puuid
            for puuid in puuids
        }
        
        completed = 0
        for future in as_completed(future_to_puuid):
            ids = future.result()
            all_match_ids.update(ids)
            
            completed += 1
            if completed % 100 == 0 or completed == len(puuids):
                logger.info(f"Fetched matches for {completed}/{len(puuids)} players (Total unique: {len(all_match_ids)})")
    
    match_ids_list = list(all_match_ids)
    logger.info(f"Found {len(match_ids_list)} unique matches")
    return match_ids_list


def get_match_data_single(match_id: str, region: str) -> List[Dict]:
    """
    Fetch a single match and return its participant data ONLY if it is a Ranked Set 13 match.
    """
    try:
        # 1. Fetch the full match details
        match_dto = watcher.match.by_id(region, match_id)
        
        # 2. CRITICAL FILTER: Check the Queue ID
        # 1100 = Ranked TFT (Standard)
        # If it's not 1100 (e.g., 6100 for Revival), return empty list immediately.
        if match_dto['info']['queue_id'] != 1100:
            return []

        # 3. Flatten the JSON
        flat_match = flatten(match_dto)
        rows = []
        
        # 4. Extract data for each of the 8 participants
        for p_idx in range(8):
            prefix = f'info_participants_{p_idx}'
            
            # Filter keys for this specific participant
            player_data = {k: v for k, v in flat_match.items() if k.startswith(prefix)}
            
            if not player_data:
                continue
            
            # Clean up keys (remove "info_participants_0_" prefix)
            clean_player_data = {
                k.replace(f'{prefix}_', ''): v 
                for k, v in player_data.items()
            }
            
            # Add Match Metadata (so we know which game this row belongs to)
            clean_player_data['match_id'] = match_id
            clean_player_data['game_datetime'] = flat_match.get('info_game_datetime')
            clean_player_data['game_version'] = flat_match.get('info_game_version')
            clean_player_data['queue_id'] = flat_match.get('info_queue_id')
            clean_player_data['tft_set_number'] = flat_match.get('info_tft_set_number')
            
            rows.append(clean_player_data)
        
        return rows

    except ApiError as err:
        # Handle 404s or other API errors gracefully
        if err.response.status_code == 404:
            logger.debug(f"Match {match_id} not found.")
        else:
            logger.debug(f"API error for match {match_id}: {err}")
        return []
        
    except Exception as err:
        logger.debug(f"Error processing match {match_id}: {err}")
        return []


def get_match_data(match_ids: List[str], region: str = MATCH_REGION) -> pd.DataFrame:
    """
    Fetch data for a list of matches in parallel, filtering for Ranked games.
    """
    logger.info(f"Fetching data for {len(match_ids)} matches...")
    match_data_rows = []
    
    # Reduced workers to prevent hitting Rate Limits (429) too quickly
    # If you have a production key, you can increase this.
    SAFE_WORKER_COUNT = 5 
    
    with ThreadPoolExecutor(max_workers=SAFE_WORKER_COUNT) as executor:
        future_to_match = {
            executor.submit(get_match_data_single, match_id, region): match_id 
            for match_id in match_ids
        }
        
        completed = 0
        for future in as_completed(future_to_match):
            result_rows = future.result()
            
            # result_rows will be empty [] if it was the wrong Queue ID
            if result_rows:
                match_data_rows.extend(result_rows)
            
            completed += 1
            if completed % 50 == 0 or completed == len(match_ids):
                logger.info(f"Processed {completed}/{len(match_ids)} matches. Collected {len(match_data_rows)} ranked player records so far.")
    
    if not match_data_rows:
        logger.warning("No ranked matches found in this batch!")
        return pd.DataFrame()

    df = pd.DataFrame(match_data_rows)
    logger.info(f"Created DataFrame with {len(df)} total ranked player records")
    return df

def save_checkpoint(data: pd.DataFrame, tier: str, checkpoint_dir: str = 'checkpoints'):
    """Save intermediate data checkpoint."""
    Path(checkpoint_dir).mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = f"{checkpoint_dir}/{tier}_{timestamp}.csv"
    data.to_csv(filepath, index=False)
    logger.info(f"Checkpoint saved: {filepath}")

class DoubleUpDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X[X['partner_group_id'].isnull()]
    
class NaNDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X.dropna(how = 'all').dropna(axis = 'columns', how = 'all')
    
class CorruptedDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        corrupted_features = ['units_5_items_0', 'units_5_items_1',	
                            'units_5_items_2', 'units_6_items_0',
                            'units_6_items_1', 'units_6_items_2',
                            'units_7_items_0', 'units_7_items_1',	
                            'units_7_items_2', 'units_3_items_0',
                            'units_3_items_1', 'units_0_items_0',
                            'units_1_items_0', 'units_1_items_1',	
                            'units_2_items_0', 'units_2_items_1',	
                            'units_2_items_2', 'units_1_items_2',
                            'units_4_items_0', 'units_4_items_1',	
                            'units_4_items_2', 'units_0_items_1',	
                            'units_3_items_2', 'units_0_items_2',	
                            'units_8_items_0', 'units_8_items_1',	
                            'units_8_items_2']
        for feature in corrupted_features:
            try:
                X = X.drop(feature, axis = 'columns')
            except:
                continue

        return X
        
class ResetIndex(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X.reset_index(drop = True)

class DescribeMissing(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # get number of missing data points per column
        missing_values_count = X.isnull().sum()

        # how many missing values do we have?
        total_cells = np.product(X.shape)
        total_missing = missing_values_count.sum()

        # percent of missing data
        percent_missing = (total_missing / total_cells) * 100
        print('Percent Missing of Data: ' + str(percent_missing))

        return X

class TrainDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # remove features that don't help with training the data
        non_training_features = ['companion_content_ID', 'companion_item_ID',
                                'companion_skin_ID', 'companion_species',
                                'players_eliminated']
        
        for feature in non_training_features:
            try:
                X = X.drop(feature, axis = 'columns')
            except:
                continue
        
        return X
    
class OutlierRemover(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # remove outliers (10% threshold to not remove level 8 data)
        threshold = int(len(X) * 0.1)
        X = X.dropna(axis = 1, thresh = threshold)
        
        return X
        
class AugmentDropper(BaseEstimator, TransformerMixin):
    """Removes all columns related to Augments."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Drop columns containing 'augment' (case insensitive)
        cols_to_drop = [c for c in X.columns if 'augment' in c.lower()]
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} augment columns.")
            X = X.drop(columns=cols_to_drop)
        return X

def use_data_pipeline(df: pd.DataFrame, filename: str):
    """Process data through analysis and ML pipelines."""
    logger.info(f"Running data pipeline for {filename}...")
    
    if df.empty:
        logger.warning(f"Empty DataFrame for {filename}, skipping pipeline")
        return None
    
    # Analysis Pipeline
    try:
        pipe_analysis = Pipeline([
           ("double_up_dropper", DoubleUpDropper()),
           ("nandrop", NaNDropper()),
           ("corruptdropper", CorruptedDropper()),
           ("resetindex", ResetIndex()),
           ("nanpercent", DescribeMissing()),
           ("augmentdropper", AugmentDropper())
        ])
        
        df_analysis = pipe_analysis.fit_transform(df)
        
        # Save unprocessed data
        Path('data').mkdir(exist_ok=True)
        output_path_unprocessed = f'data/unprocessed_{filename}.csv'
        df_analysis.to_csv(output_path_unprocessed, index=False)
        logger.info(f"Saved analysis data: {output_path_unprocessed} ({len(df_analysis)} rows)")
    except Exception as err:
        logger.error(f"Error in analysis pipeline: {err}")
        return None

    # ML Pipeline
    try:
        pipe_ml = Pipeline([
            ("name_dropper", TrainDropper()),
            ("outlier_dropper", OutlierRemover()),
        ])

        df_ml = pipe_ml.fit_transform(df_analysis)
        
        output_path_processed = f'data/processed_{filename}.csv'
        df_ml.to_csv(output_path_processed, index=False)
        logger.info(f"Saved processed data: {output_path_processed} ({len(df_ml)} rows)")
        
        return df_ml
    except Exception as err:
        logger.error(f"Error in ML pipeline: {err}")
        return None


def process_tier(tier_name: str, fetch_func, use_masters: bool = False):
    """Generic function to process a rank tier."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {tier_name.upper()} tier")
    logger.info(f"{'='*60}")
    
    entries = fetch_func()
    if not entries:
        logger.warning(f"No {tier_name} entries found")
        return None
    
    puuids = resolve_puuids(entries)
    if not puuids:
        logger.warning(f"No PUUIDs resolved for {tier_name}")
        return None
    
    match_ids = get_match_ids(puuids)
    if not match_ids:
        logger.warning(f"No matches found for {tier_name}")
        return None
    
    match_data = get_match_data(match_ids)
    if match_data.empty:
        logger.warning(f"{tier_name} match data is empty")
        return None
    
    # Save checkpoint before pipeline
    save_checkpoint(match_data, tier_name)
    
    # Process through pipeline
    processed_data = use_data_pipeline(match_data, f'{tier_name}_match_data')
    
    return processed_data


def main():
    """Main execution function."""
    start_time = time.time()
    logger.info("="*70)
    logger.info("TFT DATA COLLECTOR V3 - OPTIMIZED")
    logger.info("="*70)
    logger.info(f"Platform Region: {PLATFORM_REGION}")
    logger.info(f"Match Region: {MATCH_REGION}")
    logger.info(f"API Key: {'*' * 20}{API_KEY[-8:]}")
    
    results = {}
    
    # Process Challengers
    results['challenger'] = process_tier('challenger', get_challengers)
    
    # Process Grandmasters
    #results['grandmaster'] = process_tier('grandmaster', get_gms)
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*70)
    for tier, data in results.items():
        if data is not None:
            logger.info(f"{tier.capitalize()}: {len(data)} processed records")
        else:
            logger.info(f"{tier.capitalize()}: No data collected")
    logger.info(f"\nTotal execution time: {elapsed/60:.2f} minutes")
    logger.info("="*70)


if __name__ == "__main__":
    main()