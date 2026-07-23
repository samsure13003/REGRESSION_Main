# 🔥 Algerian Forest Fires — FWI Prediction with Regression

Predicting the **Fire Weather Index (FWI)** for two regions of Algeria using weather and fuel-moisture data, with a full pipeline covering data cleaning, EDA, feature selection, and regularized regression modeling (Linear, Ridge, Lasso, ElasticNet).

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Forest fires are influenced by a combination of weather conditions and fuel moisture. The **Canadian Fire Weather Index (FWI) System** captures this through indices such as FFMC, DMC, DC, ISI, and BUI. This project builds a regression pipeline that predicts the **FWI value** from daily weather observations, using the **Algerian Forest Fires Dataset**.

The workflow includes:
- Cleaning and structuring the raw dataset
- Exploratory Data Analysis (EDA) to understand fire patterns across regions and months
- Multicollinearity analysis and feature selection
- Feature scaling with `StandardScaler`
- Training and comparing **Linear Regression, Lasso, Ridge, and ElasticNet**, including their cross-validated variants (`LassoCV`, `RidgeCV`, `ElasticNetCV`)

---

## 🗂️ Dataset

**Source:** [Algerian Forest Fires Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset)

- **244 instances** combining two regions of Algeria:
  - **Bejaia** region (northeast Algeria) — 122 instances
  - **Sidi Bel-abbes** region (northwest Algeria) — 122 instances
- **Time period:** June 2012 to September 2012
- **11 input attributes + 1 output attribute (FWI)**

### Attribute Information

| Feature | Description |
|---|---|
| Date | Day, month (June–September), year (2012) |
| Temperature | Noon temperature (°C): 22 to 42 |
| RH | Relative Humidity (%): 21 to 90 |
| Ws | Wind speed (km/h): 6 to 29 |
| Rain | Total rainfall for the day (mm): 0 to 16.8 |
| FFMC | Fine Fuel Moisture Code: 28.6 to 92.5 |
| DMC | Duff Moisture Code: 1.1 to 65.9 |
| DC | Drought Code: 7 to 220.4 |
| ISI | Initial Spread Index: 0 to 18.5 |
| BUI | Buildup Index: 1.1 to 68 |
| FWI | Fire Weather Index (target): 0 to 31.1 |
| Classes | `fire` / `not fire` (binary label, also used as an auxiliary feature) |

### Included Data Files

| File | Description |
|---|---|
| `Algerian_forest_fires_dataset_UPDATE.csv` | Raw dataset as sourced from UCI (with header row, region split at row 122, some formatting issues) |
| `Algerian_forest_fires_cleaned_dataset.csv` | Cleaned dataset produced by the EDA notebook — stripped column names, corrected dtypes, added `Region` column, nulls removed |
| `Algerian_forest_fires_Cle.csv` | Cleaned dataset copy used directly in model training |

---

## 🧹 Data Cleaning

Performed in `Ridge__Lasso_Regression.ipynb`:

1. Loaded the raw CSV (header offset due to a title row).
2. Identified and dropped rows with missing values (including a stray duplicate header row at index 122, which separates the two regions).
3. Added a `Region` column (`0` = Bejaia, `1` = Sidi Bel-abbes) based on row position.
4. Stripped whitespace from column names.
5. Cast numeric columns to proper `int`/`float` types.
6. Exported the result to `Algerian_forest_fires_cleaned_dataset.csv`.

---

## 📊 Exploratory Data Analysis

- Class balance check for `fire` vs `not fire` (visualized with a pie chart)
- Density/histogram plots for all numerical features
- Correlation matrix and heatmap to inspect relationships between weather indices
- Box plots to inspect the spread and outliers in `FWI`
- **Monthly fire analysis** per region using count plots (`month` vs `Classes`)

**Key insight:** Most fires occurred in **August**, with high fire activity concentrated in **June, July, and August**, while **September** saw comparatively fewer fires — consistent across both regions.

---

## 🏗️ Feature Engineering & Selection

Performed in `Model_Training.ipynb`:

1. Dropped `day`, `month`, `year` (not used as model features).
2. Encoded `Classes` as binary (`0` = not fire, `1` = fire).
3. Split data into features (`X`) and target (`y = FWI`), with a **75/25 train-test split** (`random_state=42`).
4. Computed the correlation matrix on the training set and visualized multicollinearity with a heatmap.
5. Applied a **correlation threshold of 0.85** to drop highly correlated (redundant) features.
6. Standardized features using `StandardScaler` (fit on train, applied to test) — verified visually with before/after box plots.

---

## 🤖 Models Trained

All models were trained on the scaled features to predict `FWI`, evaluated using **Mean Absolute Error (MAE)** and **R² Score**:

| Model | Notes |
|---|---|
| **Linear Regression** | Baseline model |
| **Lasso Regression** | L1 regularization for sparsity |
| **LassoCV** | 5-fold cross-validated alpha selection |
| **Ridge Regression** | L2 regularization |
| **RidgeCV** | 5-fold cross-validated alpha selection |
| **ElasticNet** | Combined L1 + L2 regularization |
| **ElasticNetCV** | 5-fold cross-validated alpha/l1_ratio selection |

Each model's predictions were visualized against actual `FWI` values using scatter plots to inspect fit quality.

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — preprocessing, model training, and cross-validation

---

## 📁 Project Structure

```
├── Algerian_forest_fires_dataset_UPDATE.csv        # Raw dataset
├── Algerian_forest_fires_cleaned_dataset.csv        # Cleaned dataset (output of EDA notebook)
├── Algerian_forest_fires_Cle.csv                     # Cleaned dataset (used for modeling)
├── Ridge__Lasso_Regression.ipynb                     # Data cleaning + EDA notebook
├── Model_Training.ipynb                              # Feature selection + model training notebook
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the project
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/algerian-forest-fires-fwi-prediction.git
   cd algerian-forest-fires-fwi-prediction
   ```
2. Run the data cleaning & EDA notebook first:
   ```bash
   jupyter notebook Ridge__Lasso_Regression.ipynb
   ```
3. Then run the model training notebook:
   ```bash
   jupyter notebook Model_Training.ipynb
   ```

---

## 📈 Results

The regularized regression models (Ridge, Lasso, ElasticNet — and their cross-validated variants) were compared against a baseline Linear Regression model on held-out test data using MAE and R². Regularization helped control multicollinearity among the FWI sub-indices (FFMC, DMC, DC, ISI, BUI) after the correlation-based feature selection step.

> Exact metric values can be reproduced by running `Model_Training.ipynb` end-to-end.

---

## 🔮 Future Work

- Hyperparameter tuning via `GridSearchCV` for finer alpha/l1_ratio search
- Try tree-based models (Random Forest, XGBoost) for comparison
- Deploy the best model via a simple Flask/FastAPI web app for real-time FWI prediction
- Add classification modeling for the `fire` / `not fire` label as a complementary task

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- Dataset: [UCI Machine Learning Repository — Algerian Forest Fires Dataset](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset)
