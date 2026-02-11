import os
import pandas as pd
import numpy as np

import catboost
from pathlib import Path
from catboost import CatBoostClassifier

PROJECT = Path(__file__).parent
CSV_PATH = PROJECT / "data" / "match_data.csv"
EDA_DIR = PROJECT / "eda"
EDA_DIR.mkdir(exist_ok=True)
 

df = pd.read_csv(CSV_PATH)

# Drop a single column
df = df.drop(columns=["game_version","match_id"])
df = df.fillna(0)

y = df['placement']

cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

model = CatBoostClassifier(iterations = 100)
model.fit(df,y,cat_features,verbose = 100)

