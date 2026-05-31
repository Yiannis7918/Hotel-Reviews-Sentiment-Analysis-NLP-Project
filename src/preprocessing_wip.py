"""
Preprocessing script for the Booking.com hotel reviews sentiment project.

This script starts from the INITIAL raw CSV and recreates the cleaned table used in
Preprocessing.ipynb, then prints/saves the same main tables from the notebook:
- cleaned dataframe head
- dataframe shape
- label counts for rating > 6.5
- final binary label counts for rating > 7
- traveller_type counts > 5

Default usage from the project root:
    python notebooks/preprocessing.py --input "data/booking_reviews copy(1).csv"

Example:
    python preprocessing.py --input "booking_reviews copy(1).csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IMPORTANT_COLUMNS = [
    "review_title",
    "reviewed_at",
    "hotel_name",
    "avg_rating",
    "nationality",
    "rating",
    "review_text",
    "tags",
]

FINAL_COLUMNS = [
    "review_title",
    "hotel_name",
    "avg_rating",
    "nationality",
    "rating",
    "review_text",
    "year",
    "month",
    "season",
    "trip_type",
    "traveller_type",
    "label",
]


def get_season(month_name: str) -> str:
    """Return season name from an English month name."""
    if month_name in ["December", "January", "February"]:
        return "Winter"
    if month_name in ["March", "April", "May"]:
        return "Spring"
    if month_name in ["June", "July", "August"]:
        return "Summer"
    if month_name in ["September", "October", "November"]:
        return "Autumn"
    return np.nan


def load_raw_reviews(input_path: str | Path) -> pd.DataFrame:
    """Load the initial Booking.com reviews CSV."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    return pd.read_csv(input_path)


def clean_reviews(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Booking.com reviews dataframe.

    The notebook already starts from `booking_reviews_cleaned.csv` with shape
    (26056, 12). To recreate that from the initial CSV, we:
    1. keep rows with the columns needed for analysis,
    2. parse `reviewed_at` into year/month/season,
    3. keep only rows whose Booking.com `tags` field has enough structured parts,
    4. extract trip_type and traveller_type from the first two tag parts,
    5. create the final binary sentiment label where rating > 7 is positive.
    """
    missing_columns = [col for col in IMPORTANT_COLUMNS if col not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df = raw_df.copy()

    # Keep rows that have the fields used in the notebook tables/plots.
    df = df.dropna(subset=IMPORTANT_COLUMNS).copy()

    # Convert Booking.com date strings, e.g. "11 July 2021".
    df["reviewed_at"] = pd.to_datetime(df["reviewed_at"], errors="coerce")
    df = df.dropna(subset=["reviewed_at"]).copy()

    # The notebook's cleaned dataframe has 26056 rows. This is achieved by keeping
    # only rows with at least 4 structured tag parts, e.g.:
    # "Leisure trip~Couple~Double Room~Stayed 2 nights~Submitted via mobile".
    tag_parts = df["tags"].astype(str).str.split("~")
    df = df[tag_parts.map(len) >= 4].copy()
    tag_parts = df["tags"].astype(str).str.split("~")

    # Match the simple extraction used for the notebook exploration.
    df["trip_type"] = tag_parts.str[0]
    df["traveller_type"] = tag_parts.str[1]

    df["year"] = df["reviewed_at"].dt.year
    df["month"] = df["reviewed_at"].dt.month_name()
    df["season"] = df["month"].apply(get_season)

    # Final label used in the notebook after testing rating > 6.5 first.
    df["label"] = (df["rating"] > 7).astype(int)

    return df[FINAL_COLUMNS].reset_index(drop=True)


def save_notebook_tables(cleaned_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save the same important table outputs shown in the notebook."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df.head().to_csv(output_dir / "table_01_cleaned_head.csv", index=False)
    pd.DataFrame({"shape": [str(cleaned_df.shape)]}).to_csv(
        output_dir / "table_02_cleaned_shape.csv", index=False
    )

    label_65_counts = (cleaned_df["rating"] > 6.5).astype(int).value_counts().rename_axis("label").to_frame("count")
    label_65_counts.to_csv(output_dir / "table_03_label_counts_rating_gt_6_5.csv")

    label_70_counts = cleaned_df["label"].value_counts().rename_axis("label").to_frame("count")
    label_70_counts.to_csv(output_dir / "table_04_final_label_counts_rating_gt_7.csv")

    traveller_counts = cleaned_df["traveller_type"].value_counts()
    traveller_counts_gt5 = traveller_counts[traveller_counts > 5].to_frame("count")
    traveller_counts_gt5.to_csv(output_dir / "table_05_traveller_type_counts_gt_5.csv")


def save_plots(cleaned_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save the notebook's exploratory plots as PNG files."""
    # Imported only when needed so the table-only pipeline stays lightweight.
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=cleaned_df, y="rating")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_01_rating_boxplot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=cleaned_df, x="rating", kde=True, color="C0")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_02_rating_histplot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.countplot(data=cleaned_df, x="season", hue="label")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_03_season_by_label.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.countplot(data=cleaned_df, x="trip_type", hue="label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_04_trip_type_by_label.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.countplot(data=cleaned_df, x="traveller_type", hue="label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_05_traveller_type_by_label.png", dpi=150)
    plt.close()


def print_summary(cleaned_df: pd.DataFrame) -> None:
    """Print the same key outputs visible in the notebook."""
    print("\nCleaned dataframe head:")
    print(cleaned_df.head())

    print("\nCleaned dataframe shape:")
    print(cleaned_df.shape)

    print("\nLabel counts with threshold rating > 6.5, as tested first in the notebook:")
    print((cleaned_df["rating"] > 6.5).astype(int).value_counts().rename_axis("label"))

    print("\nFinal label counts with threshold rating > 7:")
    print(cleaned_df["label"].value_counts().rename_axis("label"))

    print("\nTraveller type counts > 5:")
    counts = cleaned_df["traveller_type"].value_counts()
    print(counts[counts > 5].to_frame("count"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Booking.com hotel reviews from the initial CSV.")
    parser.add_argument(
        "--input",
        default="data/booking_reviews copy(1).csv",
        help="Path to the initial raw Booking.com reviews CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/booking_reviews_cleaned.csv",
        help="Path where the cleaned CSV will be saved.",
    )
    parser.add_argument(
        "--tables-dir",
        default="outputs/preprocessing_tables",
        help="Folder where CSV versions of the notebook tables will be saved.",
    )
    parser.add_argument(
        "--plots-dir",
        default="outputs/preprocessing_plots",
        help="Folder where notebook plots will be saved if --save-plots is used.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save the exploratory plots from the notebook as PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = load_raw_reviews(args.input)
    cleaned_df = clean_reviews(raw_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    save_notebook_tables(cleaned_df, args.tables_dir)
    if args.save_plots:
        save_plots(cleaned_df, args.plots_dir)

    print_summary(cleaned_df)
    print(f"\nSaved cleaned CSV to: {output_path}")
    print(f"Saved table CSV files to: {Path(args.tables_dir)}")
    if args.save_plots:
        print(f"Saved plot PNG files to: {Path(args.plots_dir)}")


if __name__ == "__main__":
    main()
