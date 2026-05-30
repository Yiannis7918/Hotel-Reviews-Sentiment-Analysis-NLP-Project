"""
Exploratory Data Analysis / dataset preparation pipeline
for the Booking hotel reviews NLP project.

This script is based on the notebook:
notebooks/Exploratory_Data_Analysis.ipynb

Purpose:
- Load the original raw CSV.
- Run the same EDA-derived feature extraction steps from the notebook.
- Create the reusable working CSV: booking_reviews_cleaned.csv.

Important:
This is still part of EDA / dataset preparation.
The actual NLP preprocessing should happen later in preprocess.py.
"""

from pathlib import Path
import argparse

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "booking_reviews copy.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "booking_reviews_cleaned.csv"


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def print_section(title: str) -> None:
    """Print a clean section title in the terminal."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_basic_eda(df: pd.DataFrame, title: str = "DATAFRAME OVERVIEW") -> None:
    """
    Print the basic exploration checks used in the notebook:
    head, info, shape, missing values, and columns.
    """
    print_section(title)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nInfo:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())


def column_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact overview table with dtype, nulls, non-nulls,
    null percentage, and unique values.
    Useful inside notebooks.
    """
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null_count": df.notnull().sum(),
        "null_count": df.isnull().sum(),
        "null_percentage": (df.isnull().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
    })


def show_column_samples(df: pd.DataFrame, n: int = 3) -> None:
    """
    Print sample values from each column.
    This helps us understand what each column actually contains.
    """
    print_section(f"SAMPLE VALUES FROM EACH COLUMN — first {n} non-null values")

    for col in df.columns:
        print("\n" + "-" * 80)
        print(f"COLUMN: {col}")
        print("-" * 80)

        sample_values = df[col].dropna().head(n).values

        if len(sample_values) == 0:
            print("No non-null values.")
        else:
            for i, value in enumerate(sample_values, start=1):
                print(f"Sample {i}: {value}")


# ---------------------------------------------------------------------
# Notebook-equivalent transformation steps
# ---------------------------------------------------------------------

def load_data(input_path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load the original CSV."""
    input_path = Path(input_path)
    return pd.read_csv(input_path)


def extract_language_from_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract language from the meta column.

    Notebook equivalent:
    df["language"] = df["meta"].str.extract(r"'language':\\s*'([^']+)'")
    """
    df = df.copy()
    df["language"] = df["meta"].str.extract(r"'language':\s*'([^']+)'")
    return df


def drop_initial_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that were removed in the notebook before creating date/tag features.

    Notebook equivalent:
    columns_to_drop = [
        'index', 'reviewed_by', 'images', 'crawled_at', 'url',
        'hotel_url', 'raw_review_text', 'meta'
    ]
    df = df.drop(columns_to_drop, axis=1)
    """
    columns_to_drop = [
        "index",
        "reviewed_by",
        "images",
        "crawled_at",
        "url",
        "hotel_url",
        "raw_review_text",
        "meta",
    ]

    df = df.copy()
    return df.drop(columns=columns_to_drop, errors="ignore")


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with any missing values.

    Notebook equivalent:
    df = df.dropna()
    """
    return df.dropna().copy()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert reviewed_at to date and create:
    - year
    - month
    - season

    Notebook equivalent:
    df['date'] = pd.to_datetime(df['reviewed_at'], format='%d %B %Y')
    df['year'] = df['date'].dt.year.astype(str)
    df['month'] = df['date'].dt.strftime('%B')
    df['season'] = df['date'].dt.month.apply(get_season)
    df['month'] = pd.Categorical(...)
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["reviewed_at"], format="%d %B %Y")

    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.strftime("%B")

    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["season"] = df["date"].dt.month.apply(get_season)

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    df["month"] = pd.Categorical(
        df["month"],
        categories=month_order,
        ordered=True,
    )

    return df


def drop_reviewed_at(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop reviewed_at after extracting date features.

    Notebook equivalent:
    df = df.drop(['reviewed_at'], axis=1)
    """
    df = df.copy()
    return df.drop(columns=["reviewed_at"], errors="ignore")


def split_tags_into_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split tags into separate columns.

    Notebook equivalent:
    split_df = df["tags"].str.split("~", expand=True)
    split_df.columns = [
        "trip_type", "traveller_type", "room_type",
        "stay_length", "submission_type", "extra"
    ]

    df[["trip_type", "traveller_type", "room_type", "stay_length", "submission_type"]] =
        split_df[["trip_type", "traveller_type", "room_type", "stay_length", "submission_type"]]
    """
    df = df.copy()

    split_df = df["tags"].str.split("~", expand=True)

    # The notebook expects 6 columns.
    # In case a future CSV produces fewer/more split parts, align safely.
    expected_columns = [
        "trip_type",
        "traveller_type",
        "room_type",
        "stay_length",
        "submission_type",
        "extra",
    ]

    split_df = split_df.reindex(columns=range(len(expected_columns)))
    split_df.columns = expected_columns

    df[[
        "trip_type",
        "traveller_type",
        "room_type",
        "stay_length",
        "submission_type",
    ]] = split_df[[
        "trip_type",
        "traveller_type",
        "room_type",
        "stay_length",
        "submission_type",
    ]]

    return df


def drop_final_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop extra columns after feature extraction.

    Notebook equivalent:
    drop_columns = ['submission_type','tags','room_type','date']
    df = df.drop(drop_columns, axis=1)
    """
    drop_columns = ["submission_type", "tags", "room_type", "date"]

    df = df.copy()
    return df.drop(columns=drop_columns, errors="ignore")


def drop_language_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop language after checking that all rows are en-gb.

    Notebook equivalent:
    df["language"].value_counts()
    df = df.drop(['language'], axis=1)
    """
    df = df.copy()
    return df.drop(columns=["language"], errors="ignore")


def run_exploration_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    save_output: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the exact EDA-derived cleaning/feature-extraction flow from the notebook.

    Returns:
        Final dataframe matching booking_reviews_cleaned.csv.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if verbose:
        print_section("1. LOADING RAW DATA")
        print(f"Input path: {input_path}")

    df = load_data(input_path)

    if verbose:
        print_basic_eda(df, title="2. RAW DATA OVERVIEW")

    # Extract language from meta
    df = extract_language_from_meta(df)

    if verbose:
        print_section("3. LANGUAGE EXTRACTED FROM META")
        print(df["language"].value_counts(dropna=False))

    # Drop unused raw columns
    df = drop_initial_unused_columns(df)

    if verbose:
        print_section("4. AFTER INITIAL COLUMN DROP")
        print("Shape:", df.shape)
        print("Missing values:")
        print(df.isnull().sum())

    # Drop missing rows
    df = drop_missing_rows(df)

    if verbose:
        print_section("5. AFTER FIRST DROPNA")
        print("Shape:", df.shape)
        print("Missing values:")
        print(df.isnull().sum())

    # Add date features
    df = add_date_features(df)

    if verbose:
        print_section("6. AFTER DATE FEATURE EXTRACTION")
        print(df[["reviewed_at", "date", "year", "month", "season"]].head())

    # Drop reviewed_at
    df = drop_reviewed_at(df)

    # Split tags into features
    df = split_tags_into_features(df)

    if verbose:
        print_section("7. AFTER TAG FEATURE EXTRACTION")
        print(df[["tags", "trip_type", "traveller_type", "room_type", "stay_length", "submission_type"]].head())

    # Drop tag-related helper columns
    df = drop_final_unused_columns(df)

    if verbose:
        print_section("8. AFTER FINAL COLUMN DROP BEFORE FINAL DROPNA")
        print("Shape:", df.shape)
        print(df.head())

    # Second dropna from notebook
    df = drop_missing_rows(df)

    if verbose:
        print_section("9. AFTER SECOND DROPNA")
        print("Shape:", df.shape)
        print(df.info())

        print_section("10. LANGUAGE CHECK")
        print(df["language"].value_counts(dropna=False))

    # Drop language because all remaining rows are en-gb
    df = drop_language_column(df)

    if verbose:
        print_section("11. FINAL CLEANED DATASET")
        print("Shape:", df.shape)
        print(df.head())
        print("\nFinal overview table:")
        print(column_overview_table(df))

    if save_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        if verbose:
            print_section("12. SAVED CLEANED DATASET")
            print(f"Output path: {output_path}")

    return df


# ---------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reusable EDA pipeline for the hotel reviews dataset."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the original raw CSV file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where the cleaned CSV should be saved.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the pipeline without saving the cleaned CSV.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run the pipeline without printing detailed EDA output.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_exploration_pipeline(
        input_path=args.input,
        output_path=args.output,
        save_output=not args.no_save,
        verbose=not args.quiet,
    )
