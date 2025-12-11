# predict.py
import pandas as pd
import joblib

def predict_new_data(X_new: pd.DataFrame, model_path='models/catboost_pipeline.pkl', save_path='predictions/predicted_results.csv'):
    model_pipeline = joblib.load(model_path)
    y_pred = model_pipeline.predict(X_new)
    X_new['predicted_cnt'] = y_pred
    X_new.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")
