# Fraud Flagr

Which mobile-money transactions should be sent to manual review, and what does each choice cost in dollars rather than in F1?

**Owner:** fraud operations | **Decision cadence:** per transaction, scored in near real time | **Data:** 6,362,620 simulated transactions

## Bottom line

- On a 1,272,524-transaction test set holding 1,627 fraud cases and $228.5 billion of value, the tuned **Random Forest disrupts $7.49M of legitimate volume while catching $2.240B of fraud**, against the logistic baseline's $438.45M of legitimate volume disrupted for a comparable catch rate.
- **XGBoost misses the least fraud** at $25.36M against the forest's $32.20M, but wrongly flags roughly ten times more legitimate value at $74.22M. The choice between them is a function of review-team capacity, not of model quality, and is presented that way rather than resolved by picking the higher score.
- The fraud base rate is **0.13%**, which rules out accuracy as a usable metric before any modeling starts. Every model here is class-weighted and every comparison is on the dollar profile of its confusion matrix.
- Four balance columns in this dataset partially encode the label, because flagged transactions are cancelled. Both tree pipelines drop them. The logistic regression is retained as a labeled leakage-included reference point, not as a competing model.

## The decision and why it is hard

Fraud is 0.13% of transactions here. A model that predicts "not fraud" for everything is 99.87% accurate and worthless, so the entire problem lives in a corner of the confusion matrix that accuracy cannot see.

The asymmetry runs both ways and neither side is free. A missed fraud is a direct dollar loss. A false positive is an analyst review, a delayed or blocked legitimate payment, and a customer who may not come back. At 1.27 million transactions in the test window, a false-positive rate that looks small in percentage terms becomes an unworkable alert volume, which is why the comparison below reports dollars of legitimate volume flagged rather than just a rate.

## Architecture

```
PaySim mobile-money transactions (6,362,620 rows)
   |  kagglehub download
   v
EDA targeted at the decision
   |  class imbalance: 0.13% fraud
   |  fraud concentrated in TRANSFER and CASH_OUT
   |  fraud amounts cluster in a narrow moderate band
   v
Feature engineering
   |  one-hot transaction type        hour of day
   |  sender account age              transactions per day
   |  merchant-receiver flag          amount-to-balance ratio
   |  dummy-destination flag          high-value (top decile) flag
   |  division-by-zero -> -1, infinity -> -2 (sentinels, not dropped)
   v
LEAKAGE CHECK
   |  flagged transactions are CANCELLED in this dataset, so
   |  oldbalanceOrig / newbalanceOrig / oldbalanceDest / newbalanceDest
   |  partially encode the label itself
   |
   +--> tree pipelines: DROP all four
   +--> linear model:   RETAINED as a labeled leakage-included reference
   v
80/20 train/test split, seed 42
   |
   +----------------+----------------+
   v                v                v
Logistic        Random Forest     XGBoost
class-weighted  class-weighted    scale_pos_weight from class ratio
RandomizedSearchCV  RandomizedSearchCV  GridSearchCV
(C, log-uniform)    (n_estimators,      (max_depth, learning_rate,
threshold-aware      max_depth, splits,  n_estimators)
F1 scorer            leaves, features)   + search-surface heatmaps
   |                |                |
   +----------------+----------------+
                    v
        Financial impact conversion
        TP -> fraud caught ($)
        FN -> fraud missed ($)
        FP -> legitimate volume flagged ($)
        against $228.5B total test-set value
                    |
                    v
        Model comparison on the COST PROFILE,
        not on a single score
```

## Data

[PaySim synthetic financial dataset](https://www.kaggle.com/datasets/ealaxi/paysim1), loaded via `kagglehub`. This is a simulator-generated dataset built to mimic real mobile-money transaction patterns. It is not real bank data, and it is used because labeled real-world fraud data is not publicly available at this scale.

Test split: 1,272,524 transactions, 1,627 fraudulent (0.128%), $228,513,915,447 total transacted value.

## Feature engineering

Raw transaction columns carry little behavioral context, so the model works on engineered signals instead:

| Feature | What it captures |
|---|---|
| One-hot transaction type | Fraud is nearly confined to TRANSFER and CASH_OUT |
| Hour of transaction | Time-of-day behavioral pattern |
| Sender account age | How long this sender has been active |
| Transactions per day | Velocity, an established fraud signal |
| Merchant-receiver flag | Whether the destination is a merchant account |
| Amount-to-balance ratio | Balance-draining transfers, a common fraud pattern |
| Dummy-destination flag | Receiver with zero old and new balance, a possible mule or fake account |
| High-value flag | Top decile of transaction amounts, where losses are largest |

The amount-to-balance ratio produces division-by-zero and infinity cases. These are encoded as sentinel values of -1 and -2 rather than dropped or filled with the mean, because a transfer out of a zero-balance account is itself an anomaly worth keeping visible to the model.

## The leakage check

In this dataset, transactions detected as fraud are cancelled. That means the post-transaction balance fields (`oldbalanceOrig`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) partially encode the outcome they are supposed to predict, and any model given them reports inflated performance.

Both tree pipelines drop all four columns and are the models the recommendation rests on.

The logistic regression is retained **with** those columns as an explicit leakage-included reference, the way a leakage-audit model is kept to measure how much signal a forbidden feature carries rather than to compete on the leaderboard. Its numbers below are labeled accordingly and are not a fair comparison against the tree models.

The instructive result is that even with the leaked information, the linear model wrongly flags $438M of legitimate volume. The binding constraint for the linear model is the shape of its decision boundary, not the feature set.

## Results

Test set: 1,272,524 transactions, 1,627 fraudulent, $228.5B total value.

| Model | Fraud caught ($) | Fraud missed ($) | Legit flagged ($) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **Random Forest (tuned)** | 2,240,125,418 | 32,202,323 | **7,492,997** | **0.98** | 0.88 |
| **XGBoost (tuned)** | **2,246,970,479** | **25,357,262** | 74,218,334 | 0.87 | **0.93** |
| Logistic Regression (leakage-included reference) | 2,243,414,365 | 28,913,376 | 438,450,823 | 0.50 | 0.91 |

Confusion matrices on the same test set:

| Model | True positives | False positives | False negatives |
|---|---:|---:|---:|
| Random Forest (tuned) | 1,438 | 27 | 189 |
| XGBoost (tuned) | 1,514 | 233 | 113 |
| Logistic Regression (leakage-included) | 1,488 | 1,493 | 139 |

Reading across: the Random Forest sends 27 false alerts into the review queue, XGBoost sends 233, and the leakage-included linear model sends 1,493. That ratio, not the recall difference, is what determines whether the queue is workable.

## Recommended operating policy

```
DEFAULT: deploy the tuned Random Forest.
  - catches $2.240B of fraud
  - disrupts $7.49M of legitimate volume (27 false alerts in the test window)
  - the smallest review-queue load of any model tested, at the highest precision

SWITCH to tuned XGBoost when BOTH hold:
  - the review team can absorb roughly 8.6x the alert volume, and
  - the cost of a missed fraud carries regulatory or reputational weight
    beyond the dollar amount
  XGBoost buys approximately $6.8M less fraud missed for that alert load.

NOT DEPLOYABLE: the logistic regression as reported here. It is a
leakage-included reference only.
```

The honest framing is that the correct model depends on a number this project does not have: the cost of one analyst review. With that constant, the two tree models collapse into a single comparable dollar figure and the choice stops being a judgment call. Deriving it is the first thing I would add.

## Limitations

**PaySim is synthetic.** The dollar figures describe a simulator, not a real book, and should not be read as a business impact claim. What transfers is the evaluation discipline: scoring on dollars, treating false positives as operational load rather than a rounding error, and checking every feature for availability at decision time. That last habit is what caught the balance leakage.

**The linear model has not been re-run leak-free.** Its row above is a labeled reference, so there is no like-for-like linear comparison in the results table. That is stated rather than implied.

**No time-based split.** The train/test split is random rather than chronological, so the models are not tested against fraud patterns that emerge after the training window, which is the failure mode that matters most in production fraud detection.

**No cost-per-review constant**, which is why the Random Forest against XGBoost choice is presented as a condition rather than a decision.

**No threshold sweep.** Both tree models are evaluated at their default decision thresholds. Given the cost asymmetry, the operating point should itself be swept and chosen against a stated cost ratio rather than left at the default, which is the largest single gap in this project.

## What I would do next

- Re-run the logistic regression on the leak-free feature set, so the model comparison is like-for-like across all three.
- Add a cost-per-review constant and a threshold sweep, converting the Random Forest against XGBoost trade-off into a single dollar figure instead of a capacity condition.
- Move to a chronological train/test split so the evaluation reflects fraud patterns drifting over time.
- Add SHAP explanations at the transaction level, since a fraud analyst needs to know why an alert fired, not only that it did.
- Test a voting or stacking ensemble over the three models, and an unsupervised outlier detector as a complement for fraud patterns absent from the training labels.

## Repo contents

```
Fraud Flagr Latest.ipynb    Full analysis notebook, Exhibits 1-10
fraud flagr latest.py       Exported script version of the notebook
Fraud Flagr 3.pdf           Static export for reading without a Jupyter environment
requirements.txt            Pinned dependency versions
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook "Fraud Flagr Latest.ipynb"
```

Requires `kagglehub` credentials to auto-download the dataset, or download it manually from the Kaggle link above and point the notebook at the local CSV.
