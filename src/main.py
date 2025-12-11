# main.py
import argparse
import pandas as pd
from train import compare_between_models, create_best_model
from predict import predict_new_data

def main():
    parser = argparse.ArgumentParser(description="Bike Sharing ML Pipeline")
    
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=['train', 'predict'],
        help="Mode: 'train' to train models, 'predict' to predict new data"
    )
    
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the dataset (CSV) or new data for prediction"
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        df = pd.read_csv(args.data)
        results_df = compare_between_models(df)
        print(results_df)
        create_best_model(df)
        
    elif args.mode == 'predict':
        X_new = pd.read_csv(args.data)
        predict_new_data(X_new)
        
if __name__ == "__main__":
    main()
