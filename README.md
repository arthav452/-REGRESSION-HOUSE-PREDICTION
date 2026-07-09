# 🏠 House Price Regression Project

A regression project that predicts house prices from structural and location features. It includes a dataset, a Jupyter notebook that builds and compares six regression models, and a Streamlit web app that lets anyone upload the data, explore it, compare models, and get live price predictions.

## Project Structure

```
.
├── house_price_regression_dataset.csv   # Dataset (6,000 rows)
├── liner_reg.ipynb                      # Model development notebook
└── app.py                               # Streamlit web app
```

## Dataset

`house_price_regression_dataset.csv` contains **6,000 rows** and **8 columns**, with no missing values.

| Column | Type | Description |
|---|---|---|
| `Square_Footage` | int | Total floor area of the house |
| `Num_Bedrooms` | int | Number of bedrooms |
| `Num_Bathrooms` | int | Number of bathrooms |
| `Year_Built` | int | Year the house was constructed |
| `Lot_Size` | float | Size of the lot the house sits on |
| `Garage_Size` | int | Garage capacity |
| `Neighborhood_Quality` | int | Neighborhood quality score |
| `House_Price` | float | **Target** — sale price of the house |

## Notebook (`liner_reg.ipynb`)

The notebook walks through a full regression workflow:

1. **Data exploration** — shape, dtypes, min/max/mean/median/std, missing values, and duplicate checks.
2. **Outlier detection** — IQR-based outlier counts and percentages for every numeric feature.
3. **Visualization** — boxplots per feature, a `Square_Footage` vs `House_Price` scatter plot, a `Year_Built` vs `House_Price` regression plot, and a correlation heatmap.
4. **Train/test split** — 80/20 split of the seven input features against `House_Price`.
5. **Model training & evaluation** — six regressors trained and scored with R²:

   | Algorithm | Key Hyperparameters |
   |---|---|
   | Linear Regression | default |
   | Decision Tree Regressor | `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2` |
   | Random Forest Regressor | `n_estimators=100`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2` |
   | Support Vector Regressor (SVR) | `kernel='rbf'`, `C=100`, `gamma='scale'`, `epsilon=0.1` |
   | K-Nearest Neighbors Regressor | `n_neighbors=5`, `weights='uniform'`, `metric='minkowski'` |
   | Gradient Boosting Regressor | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3` |

6. **Model comparison table** — R² scores collected side by side for all six algorithms.
7. **Tree visualization** — plotted decision paths for the Decision Tree and a sample tree from the Random Forest.

## Streamlit App (`app.py`)

An interactive web app built on top of the same modeling pipeline, organized into four tabs:

- **📋 Overview** — row/column counts, missing value summary, data preview, and summary statistics.
- **📊 EDA** — IQR outlier table, per-feature boxplot selector, scatter plot, regression plot, and correlation heatmap.
- **🤖 Model Comparison** — trains all six regressors on the uploaded data and ranks them by R², MAE, and RMSE, highlighting the best performer.
- **🔮 Predict** — enter house features and get a price prediction from any single model, or compare predictions from all six models at once.

### Running the App

1. Install dependencies:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
   ```
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. In the browser tab that opens, upload `house_price_regression_dataset.csv` using the sidebar file uploader.
4. Adjust the test size / random state in the sidebar if desired, then explore the **Overview**, **EDA**, **Model Comparison**, and **Predict** tabs.

**Required columns** for any uploaded CSV:
`Square_Footage, Year_Built, Num_Bedrooms, Num_Bathrooms, Lot_Size, Garage_Size, Neighborhood_Quality, House_Price`

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
