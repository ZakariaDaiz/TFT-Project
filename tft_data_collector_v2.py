import os
import time
import requests
import psycopg2
from psycopg2.extras import Json
from riotwatcher import TftWatcher, ApiError

# --- Configuration ---
API_KEY = 'RGAPI-YOUR-KEY-HERE' # REPLACE THIS
PLATFORM_REGION = 'euw1'
MATCH_REGION = 'europe'

# PostgreSQL Configuration
DB_HOST = "localhost"
DB_NAME = "tft_data"
DB_USER = "postgres"
DB_PASS = "password"

# --- Data Dragon / Static Data from riot's official site ---

def get_latest_ddragon_version():
    """Fetch the latest Data Dragon version."""
    try:
        versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
        return versions[0]
    except Exception as e:
        print(f"Error getting DDragon version, defaulting to latest known: {e}")
        return "14.1.1" # Fallback

def load_tft_static_data():
    """
    Fetches TFT data from Data Dragon to map IDs (TFT13_Sion) to Names (Sion).
    Returns dictionaries for champions and items.
    """
    version = get_latest_ddragon_version()
    print(f"Using Data Dragon Version: {version}")
    
    # 1. Champions values
    champ_map = {}
    try:
        url = f"http://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/tft-champion.json"
        data = requests.get(url).json()
        for champ_id, champ_data in data['data'].items():
            # Mapping: "TFT13_Sion" -> "Sion"
            champ_map[champ_id] = champ_data.get('name', champ_id)
    except Exception as e:
        print(f"Error loading champion data: {e}")

    # 2. Items values
    item_map = {}
    try:
        url = f"http://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/tft-item.json"
        data = requests.get(url).json()
        for item_id, item_data in data['data'].items():
            item_map[item_id] = item_data.get('name', item_id)
    except Exception as e:
        print(f"Error loading item data: {e}")
        
    return champ_map, item_map

# --- Database Setup ---

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn

def init_db():
    """Create necessary tables if they don't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Matches Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS match_metadata (
            match_id VARCHAR(50) PRIMARY KEY,
            data_version VARCHAR(20),
            game_datetime BIGINT,
            game_length FLOAT,
            game_version VARCHAR(100)
        );
    """)
    
    # Participants Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS participants (
            id SERIAL PRIMARY KEY,
            match_id VARCHAR(50) REFERENCES match_metadata(match_id),
            puuid VARCHAR(100),
            placement INT,
            level INT,
            gold_left INT,
            last_round INT,
            time_eliminated FLOAT,
            augments JSONB,  -- Storing augments as JSON array
            traits JSONB,    -- Storing active traits as JSON
            units JSONB      -- Storing units and their items as JSON
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized.")

# --- ETL ---

class TftETL:
    def __init__(self, api_key, region, match_region):
        self.watcher = TftWatcher(api_key)
        self.region = region
        self.match_region = match_region
        self.champ_map, self.item_map = load_tft_static_data()
        
    def resolve_name(self, internal_id):
        """Resolves TFT13_Sion -> Sion using loaded map."""
        return self.champ_map.get(internal_id, internal_id)

    def resolve_item(self, item_id):
        """Resolves item ID/Name -> Display Name."""
        # DDragon items can be keyed by ID number or string ID
        # We try to lookup directly
        return self.item_map.get(str(item_id), str(item_id))

    def extract_league_players(self, tier='challenger'):
        """Extracts PUUIDs from a specific league tier."""
        print(f"Extracting {tier} players...")
        players = []
        try:
            if tier == 'challenger':
                league = self.watcher.league.challenger(self.region)
            elif tier == 'grandmaster':
                league = self.watcher.league.grandmaster(self.region)
            else:
                return []
            
            entries = league.get('entries', [])
            print(f"Found {len(entries)} entries in {tier}.")
            
            # For testing, verify first 5
            for entry in entries[:5]: 
                summ_id = entry['summonerId']
                try:
                    summoner = self.watcher.summoner.by_id(self.region, summ_id)
                    players.append(summoner['puuid'])
                except ApiError as e:
                    if e.response.status_code == 429:
                        print("Rate limit hit during summoner lookup. Sleeping...")
                        time.sleep(5)
                    else:
                        print(f"Error resolving summoner {summ_id}: {e}")
                time.sleep(0.5) # Courtesy sleep
                
        except ApiError as e:
            print(f"Error fetching league {tier}: {e}")
            
        return players

    def extract_matches(self, puuids, count=20):
        """Get unique match IDs for a list of PUUIDs."""
        match_ids = set()
        print(f"Fetching matches for {len(puuids)} players...")
        
        for i, puuid in enumerate(puuids):
            try:
                ids = self.watcher.match.by_puuid(self.match_region, puuid, count=count)
                match_ids.update(ids)
                if i % 5 == 0:
                    print(f"Processed {i+1}/{len(puuids)} players. Total unique matches: {len(match_ids)}")
            except ApiError as e:
                print(f"Error fetching matches for {puuid}: {e}")
            time.sleep(0.5)
            
        return list(match_ids)

    def transform_match(self, match_data, match_id):
        """
        Cleans and maps raw API data to our DB schema structure.
        Resolves unit names and item names.
        """
        info = match_data['info']
        
        # 1. Metadata
        metadata = {
            'match_id': match_id,
            'data_version': match_data.get('metadata', {}).get('data_version'),
            'game_datetime': info['game_datetime'],
            'game_length': info['game_length'],
            'game_version': info['game_version']
        }
        
        # 2. Participants
        participants = []
        for p in info['participants']:
            # Transform Units: Map "TFT13_Sion" -> "Sion"
            mapped_units = []
            for u in p.get('units', []):
                u_name = self.resolve_name(u.get('character_id'))
                
                # Transform Items inside Units
                u_items = [self.resolve_item(item) for item in u.get('itemNames', [])]
                
                mapped_units.append({
                    'name': u_name,
                    'tier': u.get('tier'),
                    'items': u_items,
                    'rarity': u.get('rarity')
                })
            
            p_data = {
                'match_id': match_id,
                'puuid': p['puuid'],
                'placement': p['placement'],
                'level': p['level'],
                'gold_left': p['gold_left'],
                'last_round': p['last_round'],
                'time_eliminated': p['time_eliminated'],
                'augments': p.get('augments', []),
                'traits': p.get('traits', []),
                'units': mapped_units
            }
            participants.append(p_data)
            
        return metadata, participants

    def load_to_postgres(self, metadata, participants):
        """Inserts transformed data into PostgreSQL."""
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            # 1. Insert Metadata
            cur.execute("""
                INSERT INTO match_metadata (match_id, data_version, game_datetime, game_length, game_version)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO NOTHING;
            """, (metadata['match_id'], metadata['data_version'], metadata['game_datetime'], 
                  metadata['game_length'], metadata['game_version']))
            
            # 2. Insert Participants
            # Only insert if match didn't exist (checked by metadata conflict) 
            # or handle duplicates appropriately. Here we assume new matches.
            for p in participants:
                cur.execute("""
                    INSERT INTO participants (match_id, puuid, placement, level, gold_left, last_round, time_eliminated, augments, traits, units)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (p['match_id'], p['puuid'], p['placement'], p['level'], p['gold_left'], 
                      p['last_round'], p['time_eliminated'], Json(p['augments']), Json(p['traits']), Json(p['units'])))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"Error saving match {metadata['match_id']}: {e}")
        finally:
            cur.close()
            conn.close()

    def run(self):
        print("Starting ETL Process...")
        
        # 1. Initialize DB
        try:
            init_db()
        except Exception as e:
            print(f"DB Connection failed: {e}. Please check your DB credentials.")
            return

        # 2. Extract Players
        challengers = self.extract_league_players('challenger')
        grandmasters = self.extract_league_players('grandmaster')
        all_puuids = list(set(challengers + grandmasters))
        
        # 3. Extract Match IDs
        match_ids = self.extract_matches(all_puuids, count=10) # Reduced count for demo
        
        # 4. Process Matches (Extract -> Transform -> Load)
        print(f"Processing {len(match_ids)} matches...")
        for i, match_id in enumerate(match_ids):
            try:
                # Extract
                match_data = self.watcher.match.by_id(self.match_region, match_id)
                
                # Transform
                metadata, participants = self.transform_match(match_data, match_id)
                
                # Load
                self.load_to_postgres(metadata, participants)
                
                if i % 10 == 0:
                    print(f"Saved {i}/{len(match_ids)} matches to DB.")
                    
            except ApiError as e:
                print(f"Failed to fetch/save match {match_id}: {e}")
                time.sleep(1) # Error backoff
        
        print("ETL Process Completed!")

if __name__ == "__main__":
    if API_KEY == 'RGAPI-YOUR-KEY-HERE':
        print("Please update the API_KEY in the script before running.")
    else:
        etl = TftETL(API_KEY, PLATFORM_REGION, MATCH_REGION)
        etl.run()
