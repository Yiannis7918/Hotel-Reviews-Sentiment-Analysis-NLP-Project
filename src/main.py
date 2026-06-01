# main.py
import argparse
import pandas as pd
import logging
from src.preprocess import prepare_dataframe, split_ground_truth, save_data
from src.train import train_lr_tfidf, train_lr_minilm, save_models
from src.evaluate import evaluate_model, evaluate_ground_truth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Hotel Review Sentiment Analysis')
    parser.add_argument('--data',         type=str, default='data/raw/booking_reviews.csv')
    parser.add_argument('--processed',    type=str, default='data/processed/booking_reviews_cleaned.csv')
    parser.add_argument('--ground_truth', type=str, default='data/ground_truth.csv')
    parser.add_argument('--models_dir',   type=str, default='models')
    parser.add_argument('--test_size',    type=float, default=0.2)
    parser.add_argument('--mode',         type=str, default='train', choices=['train', 'evaluate'])
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'train':
        # 1. Load
        logger.info("Loading data...")
        df = pd.read_csv(args.data)

        # 2. Preprocess
        logger.info("Preprocessing...")
        df = prepare_dataframe(df)
        df, ground_truth = split_ground_truth(df)
        save_data(df, ground_truth, args.processed, args.ground_truth)

        # 3. Train (next step)

        # 4. Evaluate (next step)

    elif args.mode == 'evaluate':
        # Load and evaluate on ground truth (next step)
        pass


if __name__ == '__main__':
    main()