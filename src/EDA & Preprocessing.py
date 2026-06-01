import pandas as pd
import re
import logging

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
COLUMNS_TO_DROP = ['index', 'reviewed_by', 'images', 'crawled_at', 'url',
                   'hotel_url', 'raw_review_text', 'meta']

MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

ROOM_CATEGORIES = {
    'dormitory':   r'dorm|dormitory|bunk|capsule|bed in',
    'single':      r'single',
    'double':      r'double',
    'twin':        r'twin',
    'triple':      r'triple',
    'quadruple':   r'quadruple',
    'suite':       r'suite|junior suite|penthouse',
    'studio':      r'studio',
    'apartment':   r'apartment',
    'family_room': r'family room|family suite',
    'villa':       r'villa',
    'bungalow':    r'bungalow',
    'chalet':      r'chalet',
    'cottage':     r'cottage',
    'house':       r'house|townhouse|holiday home',
    'tent':        r'tent|glamping',
    'loft':        r'loft',
}

PATTERN_TRIP      = re.compile(r'business trip|leisure trip', re.IGNORECASE)
PATTERN_STAYS     = re.compile(r'stayed \d+ nights?', re.IGNORECASE)
PATTERN_TRAVELLER = re.compile(r'solo traveller|couple|family with young children|people with friends|group|pet', re.IGNORECASE)
PATTERN_SUBMITTED = re.compile(r'submitted.*$', re.IGNORECASE)

REQUIRED_COLUMNS = ['reviewed_at', 'rating', 'tags']


# ── Validation ────────────────────────────────────────────────────────────────
def validate_dataframe(df):
    """Check that required columns exist and data is not empty."""
    logger.info(f"Validating dataframe with shape {df.shape}")

    if df.empty:
        raise ValueError("DataFrame is empty!")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Validation passed")


def validate_room_categories(df):
    """
    Check what percentage of rows fall into 'other' category.
    Warns if too many rooms are uncategorized — may indicate
    ROOM_CATEGORIES needs updating for this dataset.
    """
    if 'residence_type' not in df.columns:
        logger.warning("residence_type column not found, skipping validation")
        return

    total = len(df)
    other_count = (df['residence_type'] == 'other').sum()
    unknown_count = (df['residence_type'] == 'unknown').sum()
    other_pct = other_count / total * 100
    unknown_pct = unknown_count / total * 100

    logger.info(f"Room categorization: {other_pct:.1f}% uncategorized ('other'), "
                f"{unknown_pct:.1f}% missing ('unknown')")

    if other_pct > 20:
        logger.warning(
            f"{other_pct:.1f}% of room types could not be categorized. "
            f"Consider updating ROOM_CATEGORIES for this dataset. "
            f"Uncategorized examples: {df[df['residence_type'] == 'other']['tags'].head(5).tolist()}"
        )

    category_dist = df['residence_type'].value_counts()
    logger.info(f"Room category distribution:\n{category_dist}")


def validate_tag_extraction(df):
    """
    Check null rates for extracted tag columns.
    High null rates may indicate patterns don't match this dataset.
    """
    tag_columns = ['trip_type', 'traveller_type', 'stay_length']
    for col in tag_columns:
        if col not in df.columns:
            continue
        null_pct = df[col].isna().sum() / len(df) * 100
        logger.info(f"{col} null rate: {null_pct:.1f}%")
        if null_pct > 50:
            logger.warning(
                f"{col} has {null_pct:.1f}% null values. "
                f"The regex pattern may not match this dataset's tag format."
            )


def validate_label(df):
    """Check label distribution and warn if heavily imbalanced."""
    if 'label' not in df.columns:
        return
    dist = df['label'].value_counts(normalize=True) * 100
    logger.info(f"Label distribution:\n{dist}")
    minority_pct = dist.min()
    if minority_pct < 10:
        logger.warning(
            f"Dataset is heavily imbalanced — minority class is only {minority_pct:.1f}%. "
            f"Consider using class_weight='balanced' or SMOTE."
        )


# ── Helper functions ──────────────────────────────────────────────────────────
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def categorize_room(text):
    if pd.isna(text):
        return 'unknown'
    text_lower = text.lower()
    for category, pattern in ROOM_CATEGORIES.items():
        if re.search(pattern, text_lower):
            return category
    return 'other'


def extract_tag(tag_text, pattern):
    if pd.isna(tag_text):
        return None
    match = pattern.search(tag_text)
    return match.group().lower() if match else None


def extract_nights(text):
    if pd.isna(text):
        return None
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


# ── Main preprocessing function ───────────────────────────────────────────────
def prepare_dataframe(df):
    logger.info("Starting preprocessing pipeline")
    logger.info(f"Initial shape: {df.shape}")

    # Validate input
    validate_dataframe(df)

    # Drop unnecessary columns
    before = df.shape[1]
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    logger.info(f"Dropped {before - df.shape[1]} columns")

    # Drop nulls
    before = len(df)
    df = df.dropna()
    logger.info(f"Dropped {before - len(df)} rows with nulls")

    # Date features
    df['date']   = pd.to_datetime(df['reviewed_at'], format='%d %B %Y')
    df['year']   = df['date'].dt.year.astype(str)
    df['month']  = df['date'].dt.strftime('%B')
    df['season'] = df['date'].dt.month.apply(get_season)
    df['month']  = pd.Categorical(df['month'], categories=MONTH_ORDER, ordered=True)
    df = df.drop(columns=['reviewed_at', 'date', 'language'], errors='ignore')
    logger.info("Date features created")

    # Room category
    df['residence_type'] = df['tags'].apply(categorize_room)
    validate_room_categories(df)

    # Tag extraction
    df['trip_type']      = df['tags'].apply(lambda x: extract_tag(x, PATTERN_TRIP))
    df['stays_length']   = df['tags'].apply(lambda x: extract_tag(x, PATTERN_STAYS))
    df['traveller_type'] = df['tags'].apply(lambda x: extract_tag(x, PATTERN_TRAVELLER))
    df['stay_length']    = df['stays_length'].apply(extract_nights)
    df = df.drop(columns=['stays_length', 'tags'], errors='ignore')

    # Validate tag extraction
    validate_tag_extraction(df)

    # Drop nulls again after extraction
    before = len(df)
    df = df.dropna()
    logger.info(f"Dropped {before - len(df)} rows with nulls after tag extraction")

    # Create label
    df['label'] = (df['rating'] > 7).astype(int)
    validate_label(df)

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


# ── Ground truth split ────────────────────────────────────────────────────────
def split_ground_truth(df, frac=0.1, random_state=42):
    ground_truth = df.sample(frac=frac, random_state=random_state)
    df = df.drop(ground_truth.index)
    logger.info(f"Ground truth size: {len(ground_truth)}")
    logger.info(f"Remaining dataset size: {len(df)}")
    return df, ground_truth


# ── Save ──────────────────────────────────────────────────────────────────────
def save_data(df, ground_truth,
              data_path='data/processed/booking_reviews_cleaned.csv',
              gt_path='data/ground_truth.csv'):
    df.to_csv(data_path, index=False)
    ground_truth.to_csv(gt_path, index=False)
    logger.info(f"Saved cleaned data to {data_path}")
    logger.info(f"Saved ground truth to {gt_path}")