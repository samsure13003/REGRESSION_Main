# 🏠 California Housing — Multiple Linear Regression

Predicting **median house value** for California districts using multiple linear regression, with a full pipeline covering data loading, EDA, feature scaling, model training, evaluation, and model persistence.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

House prices are driven by a mix of socioeconomic and geographic factors — income, age of housing stock, occupancy, and location. This project builds a regression pipeline that predicts **median house value (`Price`)** from 8 numeric features, using scikit-learn's built-in **California Housing dataset**.

The workflow includes:
- Loading the dataset directly from `sklearn.datasets`
- Exploratory Data Analysis (EDA) with correlation heatmaps
- Train-test split and feature scaling with `StandardScaler`
- Training a **Linear Regression** model and inspecting coefficients
- Evaluating performance with MSE, MAE, RMSE, R², and adjusted R²
- Diagnosing model fit with prediction and residual plots
- Persisting the trained model with `pickle` for reuse without retraining

---

## 🗂️ Dataset

**Source:** [California Housing Dataset — StatLib / scikit-learn](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html), loaded via `sklearn.datasets.fetch_california_housing`

- **20,640 instances**, derived from the **1990 U.S. Census** (one row per census block group)
- **8 numeric input features + 1 target (`Price`)**
- No missing values

### Attribute Information

| Feature | Description |
|---|---|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |
| Price (target) | Median house value, in $100,000s (capped at 5.0) |

---

## 📊 Exploratory Data Analysis

- Built a DataFrame from the raw feature array and appended the target as `Price`
- Checked structure (`.info()`), null counts (none found), and summary statistics (`.describe()`)
- Computed the correlation matrix and visualized it with a **seaborn heatmap**

**Key insight:** `MedInc` (median income) shows by far the strongest correlation with `Price`, making it the most influential single predictor.

---

## 🏗️ Feature Preparation

1. Split into independent features (`X`, all columns except `Price`) and target (`y = Price`).
2. Train-test split: `test_size=0.33`, `random_state=10`.
3. Standardized features using `StandardScaler` — fit on the training set, applied to both train and test.

---

## 🤖 Model Trained

| Model | Notes |
|---|---|
| **Linear Regression** | Fit on scaled training features; coefficients and intercept inspected directly |

### Evaluation Metrics (on held-out test set)

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Adjusted R² Score

### Diagnostic Plots

- **Predicted vs. Actual** scatter plot to visually assess fit
- **Residual KDE plot** to check the distribution of errors
- **Residuals vs. Predicted** scatter plot to check for homoscedasticity

**Key insight:** The residual plot shows a clear negative diagonal trend rather than random noise. This is largely explained by `Price` being **capped at 5.0** in the dataset (any sample with a true value pinned at the cap produces `residual = 5 - y_pred`, a straight line), along with a handful of extreme outliers. This suggests a non-linear model (e.g., Random Forest, Gradient Boosting) could capture the relationship better than plain linear regression.

---

## 💾 Model Persistence

The trained model is serialized with **pickle** so it can be reused without retraining:

```python
import pickle
pickle.dump(regression, open('regressor.pkl', 'wb'))

# Later, in another session/script:
model = pickle.load(open('regressor.pkl', 'rb'))
model.predict(X_test)
```

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — preprocessing, model training, and evaluation

---

## 📁 Project Structure

```
├── Multiple_Linear_Regression.ipynb   # Full pipeline: EDA, training, evaluation, pickling
├── regressor.pkl                      # Serialized trained model (generated on run)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the project
```bash
jupyter notebook Multiple_Linear_Regression.ipynb
```

---

## 📈 Results

> Exact metric values (MSE, MAE, RMSE, R², adjusted R²) can be reproduced by running `Multiple_Linear_Regression.ipynb` end-to-end.

---

## 🔮 Future Work

- Address the capped target variable (e.g., remove or flag capped samples) to reduce residual bias
- Try regularized variants (Ridge, Lasso, ElasticNet) to compare against plain Linear Regression
- Try non-linear models (Random Forest, Gradient Boosting) to better capture non-linear structure
- Deploy the pickled model via a simple Flask/FastAPI app for real-time price prediction

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- Dataset: [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html), via `sklearn.datasets.fetch_california_housing`
