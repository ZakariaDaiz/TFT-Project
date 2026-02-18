import os
import json
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from riotwatcher import TftWatcher, ApiError

# --- Configuration ---
load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")
PLATFORM_REGION = "euw1"       # Regional endpoint for league/summoner data
MATCH_REGION = "europe"        # Routing endpoint for match data
MATCHES_PER_PLAYER = 10
RAW_DATA_PATH = "data/raw_matches.json"
OUTPUT_CSV_PATH = "data/match_data.csv"


# =====================
# EXTRACT
# =====================

def get_latest_ddragon_version():
    """Fetch the latest Data Dragon version for champion/item name resolution."""
    try:
        versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
        return versions[0]
    except Exception as e:
        print(f"[WARN] Could not fetch DDragon version, using fallback: {e}")
        return "14.1.1"


def load_static_data():
    """Load champion and item name mappings from Data Dragon."""
    version = get_latest_ddragon_version()
    print(f"Using Data Dragon v{version}")

    champ_map = {}
    try:
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/tft-champion.json"
        data = requests.get(url).json()
        for champ_id, champ_data in data["data"].items():
            champ_map[champ_id] = champ_data.get("name", champ_id)
    except Exception as e:
        print(f"[WARN] Could not load champion data: {e}")

    item_map = {}
    try:
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/tft-item.json"
        data = requests.get(url).json()
        for item_id, item_data in data["data"].items():
            item_map[item_id] = item_data.get("name", item_id)
    except Exception as e:
        print(f"[WARN] Could not load item data: {e}")

    return champ_map, item_map


def get_league_puuids(watcher, region, tier):
    """Get PUUIDs from a league tier (challenger or grandmaster)."""
    print(f"Fetching {tier} players...")
    try:
        if tier == "challenger":
            league = watcher.league.challenger(region)
        elif tier == "grandmaster":
            league = watcher.league.grandmaster(region)
        else:
            return []

        entries = league.get("entries", [])
        puuids = [e["puuid"] for e in entries if "puuid" in e][:100]
        print(f"  Found {len(puuids)} {tier} players")
        return puuids

    except ApiError as e:
        print(f"[ERROR] Failed to fetch {tier} league: {e}")
        return []


def get_match_ids(watcher, match_region, puuids, count=MATCHES_PER_PLAYER):
    """Collect unique match IDs for a list of player PUUIDs."""
    match_ids = set()
    print(f"Fetching match IDs for {len(puuids)} players...")

    for i, puuid in enumerate(puuids):
        try:
            ids = watcher.match.by_puuid(match_region, puuid, count=count)
            match_ids.update(ids)
        except ApiError as e:
            if e.response.status_code == 429:
                print("  Rate limited, sleeping 10s...")
                time.sleep(10)
            else:
                print(f"  [WARN] Error for player {i}: {e}")
        time.sleep(0.5)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(puuids)} players processed — {len(match_ids)} unique matches")

    print(f"  Total unique matches: {len(match_ids)}")
    return list(match_ids)


def fetch_match_data(watcher, match_region, match_ids):
    """Fetch full match details for each match ID. Returns list of raw match dicts."""
    raw_matches = []
    print(f"Fetching details for {len(match_ids)} matches...")

    for i, match_id in enumerate(match_ids):
        try:
            match = watcher.match.by_id(match_region, match_id)
            raw_matches.append(match)
        except ApiError as e:
            if e.response.status_code == 429:
                print("  Rate limited, sleeping 10s...")
                time.sleep(10)
            else:
                print(f"  [WARN] Failed to fetch {match_id}: {e}")
        time.sleep(0.5)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(match_ids)} matches fetched")

    print(f"  Fetched {len(raw_matches)} matches")
    return raw_matches


def save_raw_json(raw_matches, path=RAW_DATA_PATH):
    """Cache raw API responses to JSON so transforms can be re-run without API calls."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw_matches, f)
    print(f"Saved raw data to {path}")


def load_raw_json(path=RAW_DATA_PATH):
    """Load previously cached raw match data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================
# TRANSFORM
# =====================

def flatten_match(match, champ_map, item_map):
    """
    Extract one row per participant from a raw match response.
    Returns a list of flat dicts ready for DataFrame conversion.
    """
    info = match.get("info", {})
    match_id = match.get("metadata", {}).get("match_id", "")
    game_version = info.get("game_version", "")

    rows = []
    for p in info.get("participants", []):
        # Skip Double Up games
        if p.get("partner_group_id") is not None:
            return []

        row = {
            "placement": p.get("placement"),
            "level": p.get("level"),
            "gold_left": p.get("gold_left"),
            "last_round": p.get("last_round"),
            "time_eliminated": p.get("time_eliminated"),
            "total_damage_to_players": p.get("total_damage_to_players"),
            "players_eliminated": p.get("players_eliminated"),
        }

        # Augments — store as separate columns
        augments = p.get("augments", [])
        for j, aug in enumerate(augments):
            row[f"augment_{j}"] = aug

        # Traits — one column per trait with its active tier
        for trait in p.get("traits", []):
            trait_name = trait.get("name", "")
            if trait.get("tier_current", 0) > 0:
                row[f"trait_{trait_name}"] = trait.get("num_units", 0)

        # Units — character name + tier + items
        for j, unit in enumerate(p.get("units", [])):
            char_id = unit.get("character_id", "")
            row[f"unit_{j}_name"] = champ_map.get(char_id, char_id)
            row[f"unit_{j}_tier"] = unit.get("tier")

            item_names = unit.get("itemNames", [])
            for k, item in enumerate(item_names):
                row[f"unit_{j}_item_{k}"] = item_map.get(item, item)

        rows.append(row)

    return rows


def clean_dataframe(df):
    """Basic cleaning: drop empty columns/rows, reset index."""
    # Drop columns that are entirely NaN
    df = df.dropna(axis="columns", how="all")
    # Drop rows that are entirely NaN
    df = df.dropna(how="all")
    # Reset index
    df = df.reset_index(drop=True)
    return df


# =====================
# LOAD
# =====================

def save_to_csv(df, path=OUTPUT_CSV_PATH):
    """Write the cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


# =====================
# MAIN
# =====================

if __name__ == "__main__":
    if not API_KEY:
        print("Error: Set RIOT_API_KEY in your .env file.")
        exit(1)

    watcher = TftWatcher(API_KEY)
    champ_map, item_map = load_static_data()

    # --- Extract ---
    challenger_puuids = get_league_puuids(watcher, PLATFORM_REGION, "challenger")
    grandmaster_puuids = get_league_puuids(watcher, PLATFORM_REGION, "grandmaster")
    all_puuids = list(set(challenger_puuids + grandmaster_puuids))

    match_ids = get_match_ids(watcher, MATCH_REGION, all_puuids)
    raw_matches = fetch_match_data(watcher, MATCH_REGION, match_ids)

    # Cache raw data for re-processing without API calls
    save_raw_json(raw_matches)

    # --- Transform ---
    all_rows = []
    for match in raw_matches:
        all_rows.extend(flatten_match(match, champ_map, item_map))

    df = pd.DataFrame(all_rows)
    df = clean_dataframe(df)

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns[:15])}...")

    # --- Load ---
    save_to_csv(df)
    print("\nETL complete.")
