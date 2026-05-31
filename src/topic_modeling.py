"""
topic_modeling.py

Reusable topic modeling and word-analysis pipeline for the Hotel Reviews Sentiment Analysis NLP Project.

What this script does:
1. Loads a CSV dataset.
2. Uses `review_text` as the main NLP text column.
3. Optionally removes PERSON names using spaCy NER.
4. Builds a TF-IDF matrix.
5. Trains an NMF topic model.
6. Prints top words per topic.
7. Assigns each review to a topic.
8. Maps topic IDs to readable topic names.
9. Creates summary tables and plots.
10. Finds words more characteristic of positive and negative reviews.
11. Saves a new CSV with topic columns.

Run from project root:
    python src/topic_modeling.py

Optional with NER name removal:
    python src/topic_modeling.py --remove-names
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer


# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "booking_reviews_with_two_transformer_predictions.csv"
FALLBACK_INPUT_PATH = PROJECT_ROOT / "data" / "booking_reviews_cleaned.csv"

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "booking_reviews_with_topics.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "documentation" / "topic_modeling_outputs"


# ---------------------------------------------------------------------
# Topic names based on the initial NMF output from the notebook
# ---------------------------------------------------------------------

DEFAULT_TOPIC_NAME_MAP: Dict[int, str] = {
    0: "Amenities & Parking",
    1: "Room Comfort",
    2: "Location",
    3: "Value & Breakfast",
    4: "Staff & Service",
    5: "Perfect Stay",
    6: "Great Overall Experience",
    7: "Breakfast & Hospitality",
}


# ---------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------

CUSTOM_STOP_WORDS = list(
    ENGLISH_STOP_WORDS.union(
        [
            # Generic hotel/review vocabulary
            "hotel",
            "room",
            "rooms",
            "stay",
            "stayed",
            "place",
            "apartment",
            "apartments",
            "really",
            "just",
            "good",
            "great",
            "nice",
            "booking",
            "com",
            "guest",
            "guests",
            "review",
            "reviews",
            "comments",
            "available",
            "location",
            "night",
            "nights",
            "did",
            "does",
            "don",
            "didn",
            "wasn",
            "isn",
            "thing",
            "things",
            "lot",
            "bit",
        ]
    )
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def ensure_output_dir(output_dir: Path) -> None:
    """Create output folder if it does not exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the input CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return pd.read_csv(input_path)


def clean_text_basic(text: str) -> str:
    """
    Basic text normalization for vectorization.

    This is intentionally light cleaning:
    - lowercases text
    - removes URLs
    - removes digits
    - removes extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_rating_sentiment_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create POSITIVE / NEGATIVE sentiment labels from numerical rating,
    if the column does not already exist.

    Rule used in the notebook:
    rating >= 8 -> POSITIVE
    rating < 8  -> NEGATIVE
    """
    df = df.copy()

    if "rating_sentiment_binary" not in df.columns:
        if "rating" not in df.columns:
            raise ValueError("Cannot create rating_sentiment_binary because 'rating' column is missing.")

        df["rating_sentiment_binary"] = df["rating"].apply(
            lambda rating: "POSITIVE" if rating >= 8 else "NEGATIVE"
        )

    return df


def create_rating_5_star_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert original 1-10 ratings into 1-5 star labels,
    if the column does not already exist.
    """
    df = df.copy()

    if "rating_5_star_label" not in df.columns:
        if "rating" not in df.columns:
            raise ValueError("Cannot create rating_5_star_label because 'rating' column is missing.")

        def rating_to_5_star_label(rating: float) -> str:
            if rating <= 2:
                return "1 star"
            if rating <= 4:
                return "2 stars"
            if rating <= 6:
                return "3 stars"
            if rating <= 8:
                return "4 stars"
            return "5 stars"

        df["rating_5_star_label"] = df["rating"].apply(rating_to_5_star_label)

    return df


def create_star_transformer_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert star_transformer_label into POSITIVE / NEGATIVE,
    if the star transformer column exists.

    Rule:
    4 stars / 5 stars -> POSITIVE
    1 star / 2 stars / 3 stars -> NEGATIVE
    """
    df = df.copy()

    if "star_transformer_label" in df.columns and "star_transformer_binary" not in df.columns:
        def star_label_to_binary(label: str) -> str:
            return "POSITIVE" if label in ["4 stars", "5 stars"] else "NEGATIVE"

        df["star_transformer_binary"] = df["star_transformer_label"].apply(star_label_to_binary)

    return df


# ---------------------------------------------------------------------
# Optional NER cleaning
# ---------------------------------------------------------------------

def remove_person_names_with_spacy(texts: Iterable[str], model_name: str = "en_core_web_sm") -> List[str]:
    """
    Remove PERSON entities from texts using spaCy NER.

    Requires:
        python -m spacy download en_core_web_sm
    """
    try:
        import spacy
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "spaCy and tqdm are required for NER cleaning. "
            "Install with: pip install spacy tqdm && python -m spacy download en_core_web_sm"
        ) from exc

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc

    cleaned_texts: List[str] = []

    for text in tqdm(list(texts), desc="Removing PERSON names with spaCy NER"):
        doc = nlp(str(text))

        person_char_spans = [
            (ent.start_char, ent.end_char)
            for ent in doc.ents
            if ent.label_ == "PERSON"
        ]

        kept_tokens = []

        for token in doc:
            token_is_person = any(
                token.idx >= start and token.idx < end
                for start, end in person_char_spans
            )

            if not token_is_person:
                kept_tokens.append(token.text)

        cleaned_texts.append(" ".join(kept_tokens))

    return cleaned_texts


# ---------------------------------------------------------------------
# TF-IDF + NMF topic modeling
# ---------------------------------------------------------------------

def build_tfidf_matrix(
    texts: pd.Series,
    max_features: int = 5000,
    min_df: int = 5,
    max_df: float = 0.8,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: Optional[List[str]] = None,
):
    """Build TF-IDF matrix from review texts."""
    if stop_words is None:
        stop_words = CUSTOM_STOP_WORDS

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        preprocessor=clean_text_basic,
    )

    X_tfidf = vectorizer.fit_transform(texts.fillna("").astype(str))
    return vectorizer, X_tfidf


def train_nmf_topic_model(
    X_tfidf,
    n_topics: int = 8,
    random_state: int = 42,
    max_iter: int = 500,
):
    """Train NMF topic model."""
    model = NMF(
        n_components=n_topics,
        random_state=random_state,
        max_iter=max_iter,
    )

    topic_matrix = model.fit_transform(X_tfidf)
    return model, topic_matrix


def get_topics_table(
    model: NMF,
    feature_names: np.ndarray,
    n_words: int = 15,
) -> pd.DataFrame:
    """Return a dataframe with top words per topic."""
    rows = []

    for topic_idx, topic in enumerate(model.components_):
        top_word_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        rows.append({"topic_id": topic_idx, "top_words": ", ".join(top_words)})

    return pd.DataFrame(rows)


def print_topics(topics_table: pd.DataFrame) -> None:
    """Pretty print topics."""
    print("\n" + "=" * 80)
    print("TOP WORDS PER TOPIC")
    print("=" * 80)

    for _, row in topics_table.iterrows():
        print(f"Topic {row['topic_id']}:")
        print(row["top_words"])
        print("-" * 100)


def assign_topics(
    df: pd.DataFrame,
    topic_matrix: np.ndarray,
    topic_name_map: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Assign strongest topic to each review and map readable topic names."""
    df = df.copy()
    df["topic_id"] = topic_matrix.argmax(axis=1)

    if topic_name_map is None:
        topic_name_map = DEFAULT_TOPIC_NAME_MAP

    df["topic_name"] = df["topic_id"].map(topic_name_map).fillna(
        df["topic_id"].apply(lambda x: f"Topic {x}")
    )

    return df


# ---------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------

def create_topic_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Create review count per topic."""
    topic_counts = df["topic_name"].value_counts().reset_index()
    topic_counts.columns = ["topic_name", "review_count"]
    return topic_counts


def create_topic_rating_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create average/median rating per topic."""
    return (
        df.groupby("topic_name")
        .agg(
            review_count=("review_text", "count"),
            avg_rating=("rating", "mean"),
            median_rating=("rating", "median"),
        )
        .sort_values("avg_rating")
    )


def create_topic_by_column_table(
    df: pd.DataFrame,
    column: str,
    normalize: str = "index",
) -> Optional[pd.DataFrame]:
    """Create crosstab of topic_name by a selected column."""
    if column not in df.columns:
        return None

    return pd.crosstab(df["topic_name"], df[column], normalize=normalize).round(2)


def create_sample_reviews_by_topic(df: pd.DataFrame, n_samples: int = 5) -> pd.DataFrame:
    """Create a table of sample reviews for every topic."""
    sample_frames = []

    columns_to_keep = [
        col
        for col in [
            "topic_id",
            "topic_name",
            "review_text",
            "rating",
            "hotel_name",
            "season",
            "trip_type",
            "traveller_type",
        ]
        if col in df.columns
    ]

    for topic_name in sorted(df["topic_name"].dropna().unique()):
        sample = df[df["topic_name"] == topic_name][columns_to_keep].head(n_samples)
        sample_frames.append(sample)

    if not sample_frames:
        return pd.DataFrame()

    return pd.concat(sample_frames, ignore_index=True)


# ---------------------------------------------------------------------
# Distinctive words positive/negative
# ---------------------------------------------------------------------

def get_distinctive_words_by_sentiment(
    df: pd.DataFrame,
    text_col: str = "review_text",
    sentiment_col: str = "rating_sentiment_binary",
    min_count: int = 20,
    n_words: int = 15,
    stop_words: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare word usage rates in POSITIVE vs NEGATIVE reviews.

    This is better than simple frequency because simple frequency mostly
    shows generic hotel vocabulary.
    """
    if stop_words is None:
        stop_words = CUSTOM_STOP_WORDS

    required_cols = {text_col, sentiment_col}
    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns for distinctive word analysis: {missing_cols}")

    positive_texts = df[df[sentiment_col] == "POSITIVE"][text_col].fillna("").astype(str)
    negative_texts = df[df[sentiment_col] == "NEGATIVE"][text_col].fillna("").astype(str)

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_features=5000,
        ngram_range=(1, 1),
        preprocessor=clean_text_basic,
    )

    X_pos = vectorizer.fit_transform(positive_texts)
    words = vectorizer.get_feature_names_out()
    positive_counts = np.asarray(X_pos.sum(axis=0)).flatten()

    X_neg = vectorizer.transform(negative_texts)
    negative_counts = np.asarray(X_neg.sum(axis=0)).flatten()

    word_compare = pd.DataFrame(
        {
            "word": words,
            "positive_count": positive_counts,
            "negative_count": negative_counts,
        }
    )

    word_compare["positive_rate"] = word_compare["positive_count"] / max(
        word_compare["positive_count"].sum(), 1
    )
    word_compare["negative_rate"] = word_compare["negative_count"] / max(
        word_compare["negative_count"].sum(), 1
    )

    word_compare["negative_vs_positive_ratio"] = (
        (word_compare["negative_rate"] + 1e-6)
        / (word_compare["positive_rate"] + 1e-6)
    )

    word_compare["positive_vs_negative_ratio"] = (
        (word_compare["positive_rate"] + 1e-6)
        / (word_compare["negative_rate"] + 1e-6)
    )

    distinctive_negative_words = (
        word_compare[word_compare["negative_count"] >= min_count]
        .sort_values("negative_vs_positive_ratio", ascending=False)
        .head(n_words)
    )

    distinctive_positive_words = (
        word_compare[word_compare["positive_count"] >= min_count]
        .sort_values("positive_vs_negative_ratio", ascending=False)
        .head(n_words)
    )

    return word_compare, distinctive_negative_words, distinctive_positive_words


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    kind: str = "bar",
    figsize: Tuple[int, int] = (10, 5),
    rotation: int = 45,
) -> None:
    """Save a simple bar plot."""
    ax = data.plot(x=x, y=y, kind=kind, legend=False, figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if kind == "bar":
        plt.xticks(rotation=rotation, ha="right")
    else:
        plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_horizontal_bar_plot(
    data: pd.DataFrame,
    ratio_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Save a horizontal bar plot."""
    plot_df = data.sort_values(ratio_col)

    ax = plot_df.plot(
        x="word",
        y=ratio_col,
        kind="barh",
        legend=False,
        figsize=figsize,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Word")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_topic_by_category_plot(
    table: pd.DataFrame,
    title: str,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Save stacked bar chart for topic by categorical variable."""
    ax = table.plot(kind="bar", stacked=True, figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Proportion")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=table.columns.name, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_plots(
    df: pd.DataFrame,
    topic_counts: pd.DataFrame,
    topic_rating_summary: pd.DataFrame,
    distinctive_negative_words: pd.DataFrame,
    distinctive_positive_words: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate and save topic-modeling plots."""
    ensure_output_dir(output_dir)

    save_bar_plot(
        topic_counts,
        x="topic_name",
        y="review_count",
        title="Number of Reviews per Topic",
        xlabel="Topic",
        ylabel="Number of Reviews",
        output_path=output_dir / "01_topic_counts.png",
        figsize=(10, 5),
    )

    rating_plot_df = topic_rating_summary.reset_index()
    save_bar_plot(
        rating_plot_df,
        x="topic_name",
        y="avg_rating",
        title="Average Rating per Topic",
        xlabel="Topic",
        ylabel="Average Rating",
        output_path=output_dir / "02_average_rating_per_topic.png",
        figsize=(10, 5),
    )

    topic_by_season = create_topic_by_column_table(df, "season")
    if topic_by_season is not None:
        save_topic_by_category_plot(
            topic_by_season,
            title="Topic Distribution by Season",
            output_path=output_dir / "03_topic_by_season.png",
        )

    topic_by_trip_type = create_topic_by_column_table(df, "trip_type")
    if topic_by_trip_type is not None:
        save_topic_by_category_plot(
            topic_by_trip_type,
            title="Topic Distribution by Trip Type",
            output_path=output_dir / "04_topic_by_trip_type.png",
        )

    topic_by_traveller = create_topic_by_column_table(df, "traveller_type")
    if topic_by_traveller is not None:
        save_topic_by_category_plot(
            topic_by_traveller,
            title="Topic Distribution by Traveller Type",
            output_path=output_dir / "05_topic_by_traveller_type.png",
            figsize=(13, 6),
        )

    save_horizontal_bar_plot(
        distinctive_negative_words,
        ratio_col="negative_vs_positive_ratio",
        title="Words More Characteristic of Negative Reviews",
        xlabel="Negative / Positive Usage Ratio",
        output_path=output_dir / "06_distinctive_negative_words.png",
    )

    save_horizontal_bar_plot(
        distinctive_positive_words,
        ratio_col="positive_vs_negative_ratio",
        title="Words More Characteristic of Positive Reviews",
        xlabel="Positive / Negative Usage Ratio",
        output_path=output_dir / "07_distinctive_positive_words.png",
    )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_topic_modeling_pipeline(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    text_col: str = "review_text",
    n_topics: int = 8,
    remove_names: bool = False,
    save_output: bool = True,
    save_plot_files: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full reusable topic modeling pipeline.

    Returns:
        DataFrame with topic_id and topic_name columns.
    """
    if input_path is None:
        input_path = DEFAULT_INPUT_PATH if DEFAULT_INPUT_PATH.exists() else FALLBACK_INPUT_PATH

    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    if verbose:
        print("\n" + "=" * 80)
        print("TOPIC MODELING PIPELINE")
        print("=" * 80)
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        print(f"Output dir: {output_dir}")

    ensure_output_dir(output_dir)

    df = load_dataset(input_path)

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' does not exist in dataset.")

    if verbose:
        print("\nDataset shape:", df.shape)
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")

    # Make sure helper label columns exist
    df = create_rating_sentiment_binary(df)
    df = create_rating_5_star_label(df)
    df = create_star_transformer_binary(df)

    topic_text_col = text_col

    if remove_names:
        if verbose:
            print("\nRemoving PERSON names with spaCy NER...")
        df["review_text_no_names"] = remove_person_names_with_spacy(df[text_col])
        topic_text_col = "review_text_no_names"

    if verbose:
        print("\nBuilding TF-IDF matrix...")

    texts = df[topic_text_col].fillna("").astype(str)
    tfidf_vectorizer, X_tfidf = build_tfidf_matrix(texts)

    if verbose:
        print("TF-IDF matrix shape:", X_tfidf.shape)
        print("\nTraining NMF topic model...")

    nmf_model, topic_matrix = train_nmf_topic_model(X_tfidf, n_topics=n_topics)

    if verbose:
        print("Topic matrix shape:", topic_matrix.shape)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics_table = get_topics_table(nmf_model, feature_names, n_words=15)

    if verbose:
        print_topics(topics_table)

    df = assign_topics(df, topic_matrix, DEFAULT_TOPIC_NAME_MAP)

    topic_counts = create_topic_counts(df)
    topic_rating_summary = create_topic_rating_summary(df)
    sample_reviews = create_sample_reviews_by_topic(df)

    topic_by_season = create_topic_by_column_table(df, "season")
    topic_by_trip_type = create_topic_by_column_table(df, "trip_type")
    topic_by_traveller = create_topic_by_column_table(df, "traveller_type")

    word_text_col = "review_text_no_names" if remove_names else text_col

    word_compare, distinctive_negative_words, distinctive_positive_words = get_distinctive_words_by_sentiment(
        df,
        text_col=word_text_col,
        sentiment_col="rating_sentiment_binary",
        min_count=20,
        n_words=15,
    )

    topics_table.to_csv(output_dir / "topics_top_words.csv", index=False)
    topic_counts.to_csv(output_dir / "topic_counts.csv", index=False)
    topic_rating_summary.to_csv(output_dir / "topic_rating_summary.csv")
    sample_reviews.to_csv(output_dir / "sample_reviews_by_topic.csv", index=False)
    word_compare.to_csv(output_dir / "word_compare_positive_negative.csv", index=False)
    distinctive_negative_words.to_csv(output_dir / "distinctive_negative_words.csv", index=False)
    distinctive_positive_words.to_csv(output_dir / "distinctive_positive_words.csv", index=False)

    if topic_by_season is not None:
        topic_by_season.to_csv(output_dir / "topic_by_season.csv")

    if topic_by_trip_type is not None:
        topic_by_trip_type.to_csv(output_dir / "topic_by_trip_type.csv")

    if topic_by_traveller is not None:
        topic_by_traveller.to_csv(output_dir / "topic_by_traveller_type.csv")

    if save_plot_files:
        if verbose:
            print("\nSaving plots...")
        save_plots(
            df,
            topic_counts,
            topic_rating_summary,
            distinctive_negative_words,
            distinctive_positive_words,
            output_dir,
        )

    if save_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\nSaved dataset with topics to: {output_path}")

    if verbose:
        print(f"Saved topic modeling outputs to: {output_dir}")
        print("\nTopic counts:")
        print(topic_counts)
        print("\nTopic rating summary:")
        print(topic_rating_summary)
        print("\nDistinctive negative words:")
        print(distinctive_negative_words[["word", "negative_count", "positive_count", "negative_vs_positive_ratio"]])
        print("\nDistinctive positive words:")
        print(distinctive_positive_words[["word", "positive_count", "negative_count", "positive_vs_negative_ratio"]])

    return df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run topic modeling on hotel review text.")

    parser.add_argument("--input", type=str, default=None, help="Path to input CSV.")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV with topic columns.")
    parser.add_argument("--output-dir", type=str, default=None, help="Folder for summary tables and plots.")
    parser.add_argument("--text-col", type=str, default="review_text", help="Text column to use.")
    parser.add_argument("--n-topics", type=int, default=8, help="Number of NMF topics.")
    parser.add_argument("--remove-names", action="store_true", help="Use spaCy NER to remove PERSON names.")
    parser.add_argument("--no-save-output", action="store_true", help="Do not save updated dataframe.")
    parser.add_argument("--no-plots", action="store_true", help="Do not save plots.")
    parser.add_argument("--quiet", action="store_true", help="Reduce printed output.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input) if args.input else None
    output_path = Path(args.output) if args.output else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_topic_modeling_pipeline(
        input_path=input_path,
        output_path=output_path,
        output_dir=output_dir,
        text_col=args.text_col,
        n_topics=args.n_topics,
        remove_names=args.remove_names,
        save_output=not args.no_save_output,
        save_plot_files=not args.no_plots,
        verbose=not args.quiet,
    )
