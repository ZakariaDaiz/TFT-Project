import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import os

# --- 1. Define Pipeline Classes (Fixed & Consolidated) ---

class DoubleUpDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[X['partner_group_id'].isnull()]

class NaNDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.dropna(how='all').dropna(axis='columns', how='all')

class CorruptedDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # BUG FIX: Indentation corrected so 'return X' happens AFTER the loop
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
                X = X.drop(feature, axis='columns')
            except:
                continue
        
        return X  # <--- This was indented too far inside the loop in your v2 script

class ResetIndex(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.reset_index(drop=True)

class DescribeMissing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        missing_values_count = X.isnull().sum()
        total_cells = X.size
        total_missing = missing_values_count.sum()
        percent_missing = (total_missing / total_cells) * 100
        print(f'Percent Missing of Data: {percent_missing}')
        return X

# These ML classes were missing from your v2 script but are needed for the ML pipeline
class TrainDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        non_training_features = ['companion_content_ID', 'companion_item_ID',
                                'companion_skin_ID', 'companion_species',
                                'gold_left', 'players_eliminated']
        for feature in non_training_features:
            try:
                X = X.drop(feature, axis='columns')
            except:
                continue
        return X

class OutlierRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        threshold = int(len(X) * 0.1)
        X = X.dropna(axis=1, thresh=threshold)
        return X

class GetAugmentDummies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        augments = ['augments_0', 'augments_1', 'augments_2']
        # Check if columns exist before dummies to prevent errors
        existing_augments = [col for col in augments if col in X.columns]
        if existing_augments:
            X = pd.get_dummies(X, columns=existing_augments)
        return X

# --- 2. Main Execution ---

def process_checkpoint(checkpoint_path, output_filename):
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load the CSV
        df = pd.read_csv(checkpoint_path, low_memory=False)
        print(f"Loaded {len(df)} rows.")
        
        # --- Analysis Pipeline ---
        print("Running Analysis Pipeline...")
        pipe_analysis = Pipeline([
           ("double_up_dropper", DoubleUpDropper()),
           ("nandrop", NaNDropper()),
           ("corruptdropper", CorruptedDropper()),
           ("resetindex", ResetIndex()),
           ("nanpercent", DescribeMissing())
        ])
        
        df_analysis = pipe_analysis.fit_transform(df)
        
        # Save Unprocessed (Analysis) Data
        os.makedirs('data', exist_ok=True)
        unprocessed_path = f'data/unprocessed_{output_filename}.csv'
        df_analysis.to_csv(unprocessed_path, index=False)
        print(f"Saved unprocessed data to: {unprocessed_path}")

        # --- ML Pipeline ---
        print("Running ML Pipeline...")
        pipe_ml = Pipeline([
            ("name_dropper", TrainDropper()),
            ("outlier_dropper", OutlierRemover()),
            ("augmentdummies", GetAugmentDummies())
        ])

        df_ml = pipe_ml.fit_transform(df_analysis)
        
        # Save Processed (ML) Data
        processed_path = f'data/processed_{output_filename}.csv'
        df_ml.to_csv(processed_path, index=False)
        print(f"Saved processed data to: {processed_path}")
        print("Done!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Point this to your specific checkpoint file
    CHECKPOINT_FILE = 'checkpoints/challenger_20251130_184137.csv'
    process_checkpoint(CHECKPOINT_FILE, 'challenger_match_data')