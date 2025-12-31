# 🚲 Bike Sharing Demand Prediction

## 📌 Project Overview

This project aims to predict **bike rental demand (`cnt`)** based on temporal, weather, and usage-related features using **Machine Learning regression models**.

The project is built with a **clean ML pipeline architecture**, including:

* Feature engineering
* Preprocessing pipelines
* Model comparison
* Model persistence
* Train / Predict separation

---

## 🧠 Problem Statement

Given historical bike-sharing data, we want to predict the **total number of rented bikes (`cnt`) per hour** using features such as:

* Time-related features (hour, weekday, month, season)
* Weather conditions
* User behavior (registered vs casual users)

---

## ⚙️ Feature Engineering & Preprocessing

### 🔹 Feature Engineering

* **Cyclical encoding** for:

  * hour
  * weekday
  * month
  * season
* **Rush hours feature** derived from:

  * registered users
  * casual users

### 🔹 Preprocessing

* Numerical scaling using `StandardScaler`
* One-hot encoding for categorical variables
* All steps combined using `Pipeline` and `ColumnTransformer`

---

## 🤖 Models Used

The following regression models were evaluated:

* **CatBoostRegressor** ✅ *(Best Model)*
* XGBoost Regressor
* LightGBM Regressor

### 📊 Evaluation Metrics

* R² Score
* RMSE
* MAE

CatBoost achieved the best performance on the test set and was selected as the final model.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train Models & Save Best Pipeline

```bash
python src/main.py --mode train --data data/hour.csv
```

This will:

* Compare multiple models
* Train the best model
* Save the pipeline to `models/catboost_pipeline.pkl`
* Save test data to `data/test_split.csv`

---

### 3️⃣ Run Prediction on Test Data

```bash
python src/main.py --mode predict --data data/test_split.csv
```

---

## 📈 Example Results

| Model    | R² (Test) | RMSE (Test) |
| -------- | --------- | ----------- |
| CatBoost | ~0.91     | ~63         |
| XGBoost  | ~0.90     | ~70         |
| LightGBM | ~0.90     | ~68         |

---

## 🧪 Machine Learning Best Practices Applied

✔ Modular code structure
✔ Reproducible pipelines
✔ No data leakage
✔ Train/Test split respecting time order
✔ Model persistence with `joblib`
✔ Clean Git workflow

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Feature-engine
* CatBoost, XGBoost, LightGBM
* Git & GitHub

---

## 📌 Future Improvements

* Hyperparameter tuning with Optuna
* Cross-validation with TimeSeriesSplit
* Feature importance visualization
* REST API deployment (FastAPI)

---

## 👤 Author

**Mohamed Hussam**
Machine Learning Engineer

---
