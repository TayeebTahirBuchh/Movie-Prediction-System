# IMDb India Movie Rating Prediction

A machine-learning regression project that predicts IMDb ratings for Indian movies using metadata such as release year, vote count, genre, director, and cast.

**Author:** Tayeeb Tahir Buchh

- [GitHub](https://github.com/TayeebTahirBuchh)
- [LinkedIn](https://linkedin.com/in/tayeebtahirbuchh)
- [Portfolio](https://TayeebTahirBuchh.github.io)

## Project overview

The goal is to estimate a movie's numeric IMDb rating from selected movie metadata. Since the target is a continuous numeric value, this is a **regression task**, not a movie recommendation system.

The upgraded notebook compares Ridge Regression and Random Forest Regression using a reproducible train/validation/test workflow.

## Dataset

The notebook expects the IMDb Movies India dataset, commonly distributed as `IMDb Movies India.csv`.

Download the dataset from Kaggle:
- [IMDb Movies India dataset](https://www.kaggle.com/datasets)

The dataset is not included in this repository. After downloading it, create a `data` folder and place the CSV here:

```text
Movie-Prediction-System/
├── data/
│   └── IMDb Movies India.csv
├── movie-prediction-system.ipynb
├── requirements.txt
└── README.md
```

The notebook reads the CSV using `ISO-8859-1` encoding. If your downloaded copy has a different filename or location, update `DATA_PATH` in the first data-loading section.

## Features used

- `Year`: movie release year
- `Votes`: number of IMDb votes
- `Genre`: genre information
- `Director`: director name
- `Actor 1`, `Actor 2`, `Actor 3`: principal cast fields

**Target:** `Rating`

Rows without a valid target rating are excluded rather than being assigned an artificial rating of zero.

## Methodology

1. Load and inspect the dataset.
2. Check missing values and prepare the target and feature columns.
3. Explore rating, year, vote-count, and genre distributions.
4. Split the data into training, validation, and test partitions.
5. Impute missing values, scale numeric features, and one-hot encode categorical features inside a scikit-learn preprocessing pipeline.
6. Train and compare Ridge Regression and Random Forest Regression.
7. Select a model using validation RMSE.
8. Refit the selected model on training plus validation data.
9. Evaluate once on the held-out test set.
10. Inspect residuals and save the fitted pipeline.

Preprocessing is fitted only on the training data, avoiding leakage from validation and test partitions. Director and actor average-rating features computed across the full dataset are intentionally excluded because they can leak information from the target.

## Evaluation metrics

- **MAE (Mean Absolute Error):** average absolute prediction error in rating points.
- **RMSE (Root Mean Squared Error):** penalizes larger prediction errors more strongly.
- **R² (Coefficient of Determination):** measures how much target variation is explained by the model relative to a mean baseline.

No model performance numbers are claimed in this README because the upgraded notebook has not yet been executed against your local copy of the dataset.

## Setup

Python 3.10 or later is recommended.

Create and activate a virtual environment:

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook movie-prediction-system.ipynb
```

Run the notebook cells from top to bottom after placing the dataset in `data/`.

## Model artifact

The final fitted preprocessing-and-regression pipeline is saved to:

```text
artifacts/movie_rating_model.joblib
```

The dataset and generated model artifact are not committed to the repository by default.

## Limitations

- IMDb ratings and vote counts can change over time.
- Vote count may not be available at the moment a prediction is needed; this affects how the model can be used.
- Random splitting does not measure performance on future releases.
- This is an educational machine-learning project, not an official IMDb service or a recommendation engine.

## Future improvements

- Evaluate with a time-based split to simulate predicting ratings for future releases.
- Tune hyperparameters using cross-validation restricted to the training data.
- Compare additional regression algorithms.
- Examine feature importance and prediction errors across release periods and genres.
- Clearly define when the features would be available in a real prediction scenario.
