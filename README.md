# Customer-Churn-Model
A machine learning project prediction whether the customer churn or not (stay or leave)
# Customer Churn Prediction

A machine learning project to predict customer churn for a banking dataset, helping identify customers who are likely to leave so retention strategies can be applied proactively.

## 📋 Overview

This project builds and compares multiple classification models to predict whether a bank customer will churn (`Exited = 1`) or stay (`Exited = 0`), based on demographic and account-related features.

## 🔧 Tech Stack

- **Python**
- **Pandas, NumPy** — data manipulation
- **Scikit-learn** — modeling, preprocessing, evaluation
- **Matplotlib, Seaborn** — visualization
- **Streamlit** — interactive web app for live predictions

## 📊 Dataset Features

- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`
- `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- Engineered features: `BalanceToSalaryRatio`, `IsZeroBalance`, `AgeGroup`, `ProductPerYear`, `Geo_Gender`

## 🛠️ Workflow

1. **Data Preprocessing** — handled missing values, encoded categorical variables (Label Encoding)
2. **Feature Engineering** — created derived features (e.g., BalanceToSalaryRatio, AgeGroup, Geo_Gender)
3. **Feature Selection** — used correlation analysis to identify and remove weak/irrelevant features
4. **Scaling** — applied StandardScaler to normalize continuous numerical features
5. **Handling Class Imbalance** — addressed imbalanced target classes using `class_weight='balanced'`
6. **Model Training** — trained and compared multiple models:
   - Logistic Regression (default & balanced)
   - Random Forest Classifier (hyperparameter tuned via RandomizedSearchCV)
7. **Evaluation** — assessed models using Accuracy, Precision, Recall, F1-score, and ROC-AUC
8. **Deployment** — built an interactive Streamlit app for real-time churn prediction

## 📈 Model Performance

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression (Default) | 82.4% | 0.66 | 0.22 | 0.33 | - |
| Logistic Regression (Balanced) | 83.3% | 0.58 | 0.53 | 0.55 | - |
| **Random Forest (Tuned)** | **81.0%** | **0.51** | **0.67** | **0.58** | **0.843** |

**Final Model:** Random Forest (Tuned) was selected as the best model due to its superior Recall and F1-score for the churn class, which is critical for minimizing missed at-risk customers in a real-world business context.

## 🚀 How to Run

```bash
# Clone the repository
git clone <your-repo-link>
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure
