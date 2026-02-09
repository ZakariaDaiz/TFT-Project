import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT = Path(__file__).parent
CSV_PATH = PROJECT / "data" / "match_data.csv"
EDA_DIR = PROJECT / "eda"
EDA_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Data Overview ──
print("\n=== Data Overview ===")
print(df.dtypes)
print(df.describe())

# ── Missing Values ──
print("\n=== Missing Values ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
missing_df = missing_df[missing_df["count"] > 0].sort_values("pct", ascending=False)
print(f"{len(missing_df)} columns have missing values out of {df.shape[1]}")
print(missing_df.head(20))

fig, ax = plt.subplots(figsize=(12, 5))
missing_df.head(30)["pct"].plot(kind="bar", ax=ax)
ax.set_title("Top 30 Columns by Missing %")
ax.set_ylabel("Missing %")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(EDA_DIR / "missing_values.png", dpi=150)
plt.close()

# ── Placement Distribution ──
fig, ax = plt.subplots(figsize=(8, 5))
df["placement"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Placement Distribution")
ax.set_xlabel("Placement")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(EDA_DIR / "placement_distribution.png", dpi=150)
plt.close()

# ── Numeric Features vs Placement ──
numeric_cols = ["level", "gold_left", "last_round", "time_eliminated",
                "total_damage_to_players", "players_eliminated"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, col in zip(axes.flat, numeric_cols):
    df.groupby("placement")[col].mean().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Avg {col} by Placement")
    ax.set_ylabel(col)
plt.tight_layout()
plt.savefig(EDA_DIR / "numeric_vs_placement.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, col in zip(axes.flat, numeric_cols):
    sns.boxplot(data=df, x="placement", y=col, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig(EDA_DIR / "numeric_boxplots.png", dpi=150)
plt.close()

# ── Trait Analysis ──
trait_cols = [c for c in df.columns if c.startswith("trait_")]
print(f"\n=== Trait Analysis ===")
print(f"{len(trait_cols)} trait columns")

trait_presence = df[trait_cols].notna().sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
trait_presence.head(20).plot(kind="bar", ax=ax, color="coral")
ax.set_title("Top 20 Most Common Traits")
ax.set_ylabel("# Games Present")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(EDA_DIR / "top_traits.png", dpi=150)
plt.close()

# Average placement when a trait is active
trait_avg_placement = {}
for col in trait_cols:
    mask = df[col].notna() & (df[col] > 0)
    if mask.sum() >= 50:
        trait_avg_placement[col] = df.loc[mask, "placement"].mean()

trait_placement_df = pd.Series(trait_avg_placement).sort_values()

fig, ax = plt.subplots(figsize=(12, 6))
trait_placement_df.plot(kind="barh", ax=ax, color="steelblue")
ax.axvline(x=4.5, color="red", linestyle="--", label="avg (4.5)")
ax.set_xlabel("Avg Placement (lower = better)")
ax.set_title("Average Placement by Trait (min 50 games)")
ax.legend()
plt.tight_layout()
plt.savefig(EDA_DIR / "trait_avg_placement.png", dpi=150)
plt.close()

# ── Unit Analysis ──
unit_name_cols = [c for c in df.columns if c.endswith("_name")]
all_units = pd.Series(df[unit_name_cols].values.flatten()).dropna()
unit_counts = all_units.value_counts()
print(f"\n=== Unit Analysis ===")
print(f"{len(unit_counts)} unique units")

fig, ax = plt.subplots(figsize=(12, 5))
unit_counts.head(20).plot(kind="bar", ax=ax, color="mediumseagreen")
ax.set_title("Top 20 Most Played Units")
ax.set_ylabel("Times Fielded")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(EDA_DIR / "top_units.png", dpi=150)
plt.close()

# ── Correlation Heatmap ──
core_cols = ["placement"] + numeric_cols
corr = df[core_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix — Core Numeric Features")
plt.tight_layout()
plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

print(f"\nAll charts saved to {EDA_DIR}/")
print("EDA complete.")
