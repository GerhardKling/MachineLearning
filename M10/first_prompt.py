"""
First code generated based on the prompt

"""

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 0. Prepare monthly prices
# ============================================================

# Assumptions:
# - data contains only numeric commodity-price columns.
# - Its index contains monthly dates.
#
# If dates are in a column, first use, for example:
# data["Date"] = pd.to_datetime(data["Date"])
# data = data.set_index("Date")

prices = data.copy().sort_index()

# Use a monthly index.
if not isinstance(prices.index, pd.PeriodIndex):
    prices.index = pd.to_datetime(prices.index).to_period("M")
else:
    prices.index = prices.index.asfreq("M")

if prices.index.has_duplicates:
    raise ValueError("There must be only one observation per month.")

# Insert missing calendar months so that shift(1) means one month.
prices = prices.reindex(
    pd.period_range(prices.index.min(), prices.index.max(), freq="M")
)
prices.index.name = "Month"

if (prices <= 0).any().any():
    raise ValueError("Log returns require strictly positive prices.")

# Edit these names to match your columns.
metals = ["Copper", "Zinc", "Aluminium", "Nickel"]
lags = [1, 2, 3]

# ============================================================
# 1. Create a new DataFrame of monthly log returns
# ============================================================

log_returns = np.log(prices / prices.shift(1))

# Returns are decimals: 0.01 is approximately 1%.

# ============================================================
# 2. Classify movements and create the Copper target y
# ============================================================

# Exact thresholds for simple price changes of +1% and -1%.
# These are slightly different from log returns of +/-0.01.
upper = np.log1p(0.01)
lower = np.log1p(-0.01)

movements = pd.DataFrame(
    np.select(
        [log_returns > upper, log_returns < lower],
        [1, -1],
        default=0
    ),
    index=log_returns.index,
    columns=log_returns.columns
)

# Preserve missing returns as missing, rather than neutral.
movements = movements.where(log_returns.notna())

y = movements["Copper"].rename("y")

# ============================================================
# 3. Create lagged features X
# ============================================================

# Each row dated t contains information from before month t.
X = pd.concat(
    [
        movements[metals].shift(lag).add_suffix(f"_lag{lag}")
        for lag in lags
    ],
    axis=1
)

# Combine predictors and target to keep their dates aligned.
model_data = pd.concat([X, y], axis=1)

# ============================================================
# 4. Reserve the final 12 calendar months for testing
# ============================================================

test_start = prices.index[-1] - 11

train = model_data.loc[model_data.index < test_start].dropna()
test = model_data.loc[model_data.index >= test_start]

# Do not silently replace missing test months with older months.
if len(test) != 12 or test.isna().any().any():
    raise ValueError(
        "The final 12 months must have complete targets and lagged features."
    )

if train.empty or train["y"].nunique() < 2:
    raise ValueError("Training data must contain at least two target classes.")

X_train = train.drop(columns="y").astype(int)
y_train = train["y"].astype(int)

X_test = test.drop(columns="y").astype(int)
y_test = test["y"].astype(int)

# Treat down/neutral/up as categories.
# Neutral (0) is the reference category for each predictor.
model = make_pipeline(
    OneHotEncoder(
        categories=[[-1, 0, 1] for _ in X_train.columns],
        drop=[0 for _ in X_train.columns],
        handle_unknown="error"
    ),
    LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0
    )
)

model.fit(X_train, y_train)

# In-sample fitted classifications.
pred_train = model.predict(X_train)

# Out-of-sample one-month-ahead classifications.
pred_test = model.predict(X_test)

# ============================================================
# 5. Evaluate in-sample and test performance
# ============================================================

labels = [-1, 0, 1]
names = ["Down", "Neutral", "Up"]

def evaluate(actual, predicted, title):
    print(f"\n{title}")
    print(f"Accuracy: {accuracy_score(actual, predicted):.3f}")
    print(
        classification_report(
            actual,
            predicted,
            labels=labels,
            target_names=names,
            digits=3,
            zero_division=0
        )
    )

    cm = pd.DataFrame(
        confusion_matrix(actual, predicted, labels=labels),
        index=["Actual down", "Actual neutral", "Actual up"],
        columns=["Predicted down", "Predicted neutral", "Predicted up"]
    )
    print(cm)


evaluate(y_train, pred_train, "IN-SAMPLE PERFORMANCE")
evaluate(y_test, pred_test, "FINAL 12 MONTHS: OUT-OF-SAMPLE PERFORMANCE")

# Benchmark 1: always predict the training sample's most common class.
majority_class = y_train.mode().iloc[0]
pred_majority = np.full(len(y_test), majority_class)

# Benchmark 2: predict that Copper repeats its previous movement.
pred_persistence = X_test["Copper_lag1"].to_numpy()

comparison = pd.DataFrame(
    {
        "Accuracy": [
            accuracy_score(y_test, prediction)
            for prediction in [pred_test, pred_majority, pred_persistence]
        ],
        "Macro F1": [
            f1_score(
                y_test, prediction,
                labels=labels,
                average="macro",
                zero_division=0
            )
            for prediction in [pred_test, pred_majority, pred_persistence]
        ]
    },
    index=["Logistic regression", "Majority class", "Previous movement"]
)

print("\nTEST PERFORMANCE AGAINST BENCHMARKS")
print(comparison.round(3))

# Month-by-month results, including predicted probabilities.
results = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": pred_test,
        "Correct": y_test.to_numpy() == pred_test
    },
    index=y_test.index
)

probabilities = model.predict_proba(X_test)
class_names = {-1: "P_down", 0: "P_neutral", 1: "P_up"}

for column, class_value in enumerate(
    model.named_steps["logisticregression"].classes_
):
    results[class_names[class_value]] = probabilities[:, column]

print("\nMONTH-BY-MONTH TEST RESULTS")
print(results.round(3))