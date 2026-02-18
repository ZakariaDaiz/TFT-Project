import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

PROJECT = Path(__file__).parent
CSV_PATH = PROJECT / "data" / "match_data.csv"
EDA_DIR = PROJECT / "eda"
EDA_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

y = df['placement']

# Drop columns we dont need
df = df.drop(columns=["game_version", "match_id", "placement"])
df = df.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42
)

cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

model = CatBoostClassifier(iterations=100)
model.fit(X_train, y_train, cat_features=cat_features, verbose=100)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
disp.plot(cmap="Blues")
plt.title("CatBoost — Placement Confusion Matrix")
plt.tight_layout()
plt.savefig(PROJECT / "confusion_matrixes" / "catboost_placement.png")
plt.show()

# Save model
model.save_model(str(PROJECT / "catboost_model.cbm"))
print("Model saved to catboost_model.cbm")
