# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This project is made purely for educational purpose, to put it in my CV and show recruters that i can handle data manipulation and model trainings.
We dont need our project to be anywhere near production level or to have any usefull accuracy, even tho i would like to use technologies that are coherent with today's standards.

## Project Overview

TFT (Teamfight Tactics) Data Analysis project: an ETL pipeline that collects match data from the Riot Games API for Challenger/Grandmaster players, stores it in a CSV file, and trains a CatBoost model to predict match placements.

- **Python 3.10**
- **Dependencies:** see `requirements.txt` (ETL deps only — ML deps like catboost, scikit-learn, matplotlib, seaborn to be added later)

## Running

```bash
pip install -r requirements.txt

# Set your Riot API key in .env:
# RIOT_API_KEY=RGAPI-your-key-here

python etl.py
```

Configuration (region, match count) is set at the top of `etl.py`. Outputs `data/match_data.csv`.

## Architecture

### ETL Pipeline (`etl.py`)

Simple functional ETL with three phases:

1. **Extract** — `get_league_puuids()` fetches Challenger/GM PUUIDs, `get_match_ids()` collects match IDs, `fetch_match_data()` gets full match details. Raw responses are cached to `data/raw_matches.json` so transforms can be re-run without API calls.
2. **Transform** — `flatten_match()` extracts one row per participant with placement, augments, traits, and units (resolving IDs via Data Dragon). `clean_dataframe()` drops empty columns/rows.
3. **Load** — `save_to_csv()` writes the cleaned DataFrame to `data/match_data.csv`.

### Data Flow

```
Riot API → extract functions → raw JSON (cached to data/raw_matches.json)
         → flatten_match()   → one dict per participant
         → clean_dataframe() → pandas DataFrame
         → save_to_csv()     → data/match_data.csv
```

### ML Model

CatBoost classifier predicting placement (to be implemented in a notebook). Previous results:
- 2-bin (top 4 vs bottom 4): ~80% accuracy
- 4-bin (paired placements): ~55% accuracy
- 8-bin (per-placement): ~30% accuracy

### Key Directories

- `data/` — Generated CSVs and cached raw JSON (gitignored)
- `eda/` — Exploratory data analysis visualizations
- `confusion_matrixes/` — Model performance confusion matrices
