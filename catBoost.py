import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

PROJECT = Path(__file__).parent
CSV_PATH = PROJECT / "data" / "match_data.csv"
CM_DIR = PROJECT / "confusion_matrixes"
CM_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

y_8 = df['placement']

# Drop columns we dont need
df = df.drop(columns=[
    "game_version", "match_id", "placement",
    # Leaky features: consequences of placement, not causes
    "last_round", "time_eliminated", "total_damage_to_players", "players_eliminated",
], errors='ignore')
df = df.fillna(0)

cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Binned targets
y_4 = y_8.map({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4})
y_2 = y_8.map({1: "Top4", 2: "Top4", 3: "Top4", 4: "Top4",
               5: "Bot4", 6: "Bot4", 7: "Bot4", 8: "Bot4"})

CONFIGS = [
    {"label": "8-class", "y": y_8, "filename": "confusion_matrix_8.png"},
    {"label": "4-class", "y": y_4, "filename": "confusion_matrix_4.png"},
    {"label": "2-class", "y": y_2, "filename": "confusion_matrix_2.png"},
]


def train_and_evaluate(X, y, label, filename):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function='MultiClass',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[{label}] Accuracy: {acc:.3f}")

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(f"CatBoost — {label} Placement")
    plt.tight_layout()
    plt.savefig(CM_DIR / filename)
    plt.show()

    return model, acc


# === Run all three models ===
for config in CONFIGS:
    model, acc = train_and_evaluate(df, config["y"], config["label"], config["filename"])

# === Feature importance from the 8-class model ===
model_8, _ = train_and_evaluate(df, y_8, "8-class (importance)", "confusion_matrix_8.png")
print("\nTop 20 features by importance:")
importances = pd.Series(model_8.get_feature_importance(), index=df.columns)
print(importances.sort_values(ascending=False).head(20))

# Save the 8-class model
model_8.save_model(str(PROJECT / "catboost_model.cbm"))
print("\nModel saved to catboost_model.cbm")
