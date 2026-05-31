# Hotel-Reviews-Sentiment-Analysis-NLP-Project

This project analyzes hotel guest reviews using exploratory data analysis, sentiment analysis, and pretrained transformer models.

The goal is to understand how guest ratings relate to the written review text, identify positive and negative sentiment patterns, and explore cases where the numerical rating and the review text express different signals.

## Project Overview

Hotel reviews often contain both a numerical rating and written feedback. While the rating shows the guest's overall satisfaction, the review text may contain more detailed information about the guest experience.

This project uses both signals:

- The numerical rating is used to create rating-based sentiment labels.
- The review text is analyzed using pretrained transformer sentiment models.
- Disagreement cases are examined to identify hidden complaints inside otherwise positive reviews.

For example, a guest may give a high rating overall but still mention issues such as noise, smell, cleanliness, or poor breakfast. These cases are useful from a business perspective because they reveal operational problems that may not be obvious from the rating alone.

## Dataset

The dataset contains hotel reviews collected from Booking.com.

Main columns used in the project include:

- `review_title`
- `hotel_name`
- `avg_rating`
- `nationality`
- `rating`
- `review_text`
- `year`
- `month`
- `season`
- `trip_type`
- `traveller_type`
- `stay_length`

The original dataset was cleaned and transformed into a working dataset: ..data/booking_reviews_cleaned.csv
Additional transformer predictions were saved into:  data/booking_reviews_with_two_transformer_predictions.csv

# Pipeline Overview: 

data/booking_reviews copy.csv
        │
        ▼
src/exploration.py
        │
        ├─ clean columns
        ├─ extract language from meta
        ├─ extract year / month / season
        ├─ extract trip_type / traveller_type / stay_length from tags
        ▼
data/booking_reviews_cleaned.csv
        │
        ▼
Sentiment_Analysis_Transf.ipynb
        │
        ├─ default transformer sentiment pipeline
        ├─ rating-based binary labels
        ├─ classification report + confusion matrix
        ├─ specific 1–5 star transformer
        ├─ rating-based 5-star labels
        ├─ comparison + plots
        ▼
data/booking_reviews_with_two_transformer_predictions.csv


# Project Workflow

## 1. Exploratory Data Analysis

### The EDA phase includes:
- Dataset shape and column inspection
- Missing value analysis
- Data type inspection
- Rating distribution
- Language extraction from metadata
- Date feature extraction
- Season creation
- Tag feature extraction
- Traveler type and trip type analysis

The reusable EDA pipeline is stored in:  src/exploration.py

## 2. Rating-Based Sentiment Labels

Since the dataset does not contain manually annotated sentiment labels, the numerical rating is used as a proxy label.

For binary sentiment: 
rating >= 8  → POSITIVE
rating < 8   → NEGATIVE

## 3. Default Pretrained Transformer

The first transformer model uses the default Hugging Face sentiment-analysis pipeline.
This model predicts: POSITIVE / NEGATIVE
The model predictions are compared with the rating-based binary sentiment labels.

## 4. Specific Pretrained Transformer

A second pretrained transformer model is used:
nlptown/bert-base-multilingual-uncased-sentiment
This model predicts review sentiment as: 1 star / 2 stars / 3 stars / 4 stars / 5 stars
This is more relevant for hotel reviews because the original dataset also contains numerical guest ratings.
The original 1–10 ratings are converted into 1–5 star labels and compared with the model predictions.

# Results Summary

## Default Binary Transformer
The default transformer achieved approximately 62% agreement with the rating-based binary sentiment labels.
The model performed better on positive reviews than negative reviews. However, many high-rated reviews were predicted as negative because the review text contained complaints or mixed feedback.
This shows that rating-based sentiment and text-based sentiment are related, but not identical.

<img width="1935" height="566" alt="image" src="https://github.com/user-attachments/assets/9072f083-d8ae-4cb6-b973-b883c1de92b6" />
<img width="1370" height="762" alt="image" src="https://github.com/user-attachments/assets/7ef8ed24-1e7c-4497-a17d-1cbe29e72781" />
<img width="1047" height="869" alt="image" src="https://github.com/user-attachments/assets/c35ab55c-af6f-448a-a2d2-4b8433486edd" />
classification report:
<img width="1073" height="458" alt="image" src="https://github.com/user-attachments/assets/1a5272f7-41cf-4284-8925-e4004694fa2c" />
confusion matrix: 
<img width="1486" height="1137" alt="image" src="https://github.com/user-attachments/assets/cf4b68cf-660b-4a50-8e6a-4941adf0be3b" />



## Specific 5-Star Transformer
The specific pretrained transformer achieved lower exact agreement with the rating-derived 1–5 star labels.
This is expected because predicting the exact star category is harder than binary positive/negative classification. The dataset is also highly imbalanced, with most reviews having high ratings.
However, the model is useful because it provides a more detailed text-based sentiment signal.
<img width="3241" height="1856" alt="image" src="https://github.com/user-attachments/assets/2614bc56-0db8-4712-ab37-cb3e7ba3f3ec" />
<img width="1380" height="772" alt="image" src="https://github.com/user-attachments/assets/f790e67b-1949-4c84-b254-2caee06c3cef" />
<img width="832" height="425" alt="image" src="https://github.com/user-attachments/assets/4bfa5e08-a0c1-4034-9510-d7d2c980403f" />
classification report:
<img width="1196" height="608" alt="image" src="https://github.com/user-attachments/assets/96f2e275-110f-4a12-ab7e-1ccf77d70608" />
confusion matrix:
<img width="1461" height="1229" alt="image" src="https://github.com/user-attachments/assets/95d3e53f-0aeb-458e-aa52-72bae88b6b59" />


classification report for both transformers:
<img width="1170" height="448" alt="image" src="https://github.com/user-attachments/assets/80bf04f3-943e-4e84-9226-8ee57916f578" />
confusion matrix for bothj trasnformers:
<img width="1498" height="1131" alt="image" src="https://github.com/user-attachments/assets/475ca25a-2fe0-4a79-afdd-9b4b2d0af2d7" />



## Business Insights

The analysis shows that written reviews can reveal information that is hidden behind high numerical ratings.

Important disagreement cases include:
High rating but negative transformer prediction
High rating but low star prediction
Low rating but positive transformer prediction

These cases may reveal mixed feedback or specific operational issues such as:
bad smell
noise
cleanliness problems
uncomfortable beds
poor breakfast
room size issues
service problems

This makes transformer-based sentiment analysis useful not only for prediction, but also for business insight extraction.


# Repository Structure:
Hotel-Reviews-Sentiment-Analysis-NLP-Project/
│
├── data/
│   ├── booking_reviews copy.csv
│   ├── booking_reviews_cleaned.csv
│   └── booking_reviews_with_two_transformer_predictions.csv
│
├── notebooks/
│   ├── Exploratory_Data_Analysis.ipynb
│   └── Sentiment_Analysis_Transf.ipynb
│
├── src/
│   ├── exploration.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│
├── documentation/
│
├── README.md
└── .gitignore

## required packages:
pip install pandas numpy matplotlib seaborn scikit-learn nltk transformers torch datasets evaluate accelerate tqdm
pip install sentence-transformers umap-learn spacy


# Tools and Libraries

## This project uses:
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Hugging Face Transformers
PyTorch
tqdm

# How to run: 





# Current Status

## Completed:
Exploratory data analysis
Data cleaning
Feature extraction from tags and dates
Binary rating-based sentiment labels
Default pretrained transformer sentiment analysis
Specific pretrained transformer star-rating analysis
Classification reports and confusion matrices

## Next steps:
Topic modeling
Classical machine learning baseline models
Comparison between transformer models and baseline models
Deeper analysis of disagreement cases

#Notes
The sentiment labels used in this project are derived from numerical ratings, not manually annotated sentiment labels. Therefore, they should be interpreted as proxy labels.

This means that disagreement between transformer predictions and rating-based labels is not always an error. In many cases, it may reveal useful mixed feedback in the review text.


# Contributors
