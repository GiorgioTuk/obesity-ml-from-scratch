# Obesity Level Estimation — ML Algorithms from Scratch

Classic machine-learning algorithms implemented **from scratch with NumPy** and applied to the
[Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
dataset (UCI). scikit-learn is used only for metrics and to cross-check every from-scratch model against its library counterpart.

Homework for the *Fondamenti di Intelligenza Artificiale* (Foundations of AI) course, B.Sc., A.Y. 2023/24 — revised in 2026.

**Author:** Giorgio Tuccinardi

## What is implemented

| Task | Target | From-scratch models |
|---|---|---|
| Regression | `Weight` (kg) | Linear regression (closed form + ridge), linear regression (gradient descent), feed-forward neural network (backprop, mini-batch GD) |
| Binary classification | obese vs. not obese | Logistic regression (GD + L2), K-Nearest Neighbors (Euclidean / Manhattan), decision tree (information gain) |

Every model has documented hyperparameter tuning on a validation set.

## Results (test set, 313 samples)

| Regression of `Weight` | RMSE [kg] | MAE [kg] | R² |
|---|---|---|---|
| Baseline (predict the mean) | 25.08 | 20.70 | 0.00 |
| Linear regression (closed form / GD) | 17.97 | 13.97 | 0.49 |
| Neural network (64-32, ReLU) | **12.11** | **7.36** | **0.77** |

| Classification (obese vs. not obese) | Accuracy | F1 |
|---|---|---|
| Baseline (majority class) | 0.527 | — |
| Logistic regression | 0.744 | 0.758 |
| Decision tree (max_depth 15, min_samples_split 10) | 0.898 | 0.892 |
| KNN (k = 3, Manhattan) | **0.946** | **0.943** |

The from-scratch KNN, decision tree and linear regression give exactly the same predictions as scikit-learn. Logistic
regression and the MLP get very close results. The full discussion is in the *Critical analysis* section of the notebook.

## A note on target leakage

In this dataset the obesity class was assigned from the BMI (`Weight / Height²`). A fixed rule `BMI >= 30` already gets 99.7%
accuracy without any learning. So `Weight` is excluded from the classification features, and `NObeyesdad` is excluded from
the regression features. Section 6 of the notebook shows how putting them back inflates the results: classifier accuracy goes
up to 97–99%, and the regression RMSE drops from 18 kg to 5 kg.

## Repository structure

```
.
├── obesity_ml_from_scratch.ipynb   # the notebook (executed, with outputs)
├── data/ObesityDataSet.csv         # dataset (UCI id 544, CC BY 4.0)
├── requirements.txt
└── README.md
```

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook obesity_ml_from_scratch.ipynb
```

The whole notebook runs in about 20 seconds. Results are reproducible, with a fixed seed (42).

## Changes from the original 2024 submission

- **Target leakage removed.** The original reported 100% classification accuracy because `Height` and `Weight` were both inputs.
- **Proper train / validation / test split** (70/15/15). Hyperparameters used to be chosen on the test set.
- **Standardisation fitted on the training set only.** Before, the test set was standardised with its own statistics.
- **Bugs fixed.** Logistic regression grid search returned the weights of the last combination instead of the best one.
  The neural network applied ReLU to its regression output. KNN used unscaled features.
- **Better encoding.** Ordinal attributes (`CAEC`, `CALC`) are now encoded as ordered integers. Duplicate records are removed.
- **More tuning and checks.** Ridge penalty, L2 for logistic regression, KNN distance metric, `min_samples_split` for the tree.
  Added a gradient check for backpropagation, baselines, and precision / recall / F1 / confusion matrices.
- **Faster code.** KNN distances are vectorised and the tree's split search is vectorised (seconds instead of minutes).
- **Reproducibility.** Fixed seed, local copy of the dataset, translated to English.

## Dataset

Palechor, F. M., & de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical
condition in individuals from Colombia, Peru and Mexico.* Data in Brief, 25. Distributed by the UCI Machine Learning Repository
under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
