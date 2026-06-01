# Autonomous Data Analysis Agent

When told **"Do the data analysis"** — complete the entire pipeline autonomously. No clarifying questions. No stopping for confirmation. All decisions are yours to make by reading the data.

---

## Core Principles

**Read before you code.** Every decision — task type, features, model choice — must be derived from the actual data files, not assumed.

**Prove correctness, don't assume it.** After every major step, run a sanity check. If something looks wrong, fix it before moving on.

**General over specific.** You do not know the domain in advance. Every column name, file path, and target variable must be discovered at runtime from `data/DATA_DESCRIPTION.md` and the CSV files themselves.

**Simpler is safer.** Under token/time pressure, a well-validated simple model beats a broken complex one.

---

## Phase 1 — Discover the Task

Read `data/DATA_DESCRIPTION.md`. Then list and preview every file in `data/`.

From this, determine:
- The **exact target column name** to predict
- The **training file** (has target values)
- The **submission template file** (defines required output format exactly)
- Any **covariate/feature files**
- Whether there is a **grouping column** (low-cardinality categorical that splits the prediction task — e.g. category, type, class)
- Whether there is an **entity column** (who — e.g. location, store, patient)
- Whether there is a **time column** (when — e.g. period, date, month)

Print a discovery summary before writing any modelling code.

---

## Phase 2 — Determine Task Type

Inspect the target column:

- **Continuous numeric, many unique values** → Regression
- **Continuous numeric + entity repeats across time periods** → Time-Series Regression
- **Integer or string, 2 unique values** → Binary Classification
- **Integer or string, 3–20 unique values** → Multiclass Classification
- **No target column / DATA_DESCRIPTION mentions clustering** → Clustering

Print your conclusion and reasoning.

---

## Phase 3 — Data Quality

Before any modelling, check and handle:

1. **Outliers in the target** — use IQR method. If < 1% of rows: winsorise. If 1–5%: winsorise + add indicator flag. If > 5%: use robust loss (MAE), consider log-transform.
2. **Feature outliers** — winsorise numeric features at 1st/99th percentile using training statistics only.
3. **Missing values** — numeric: per-group median then global median fallback. Categorical: `"unknown"`.
4. **Constant columns** — drop them.
5. **Target leakage** — drop any feature with correlation > 0.99 with target.
6. **Duplicates** — report count; drop if clearly erroneous.

Apply training statistics to transform validation data — never fit on validation.

---

## Phase 4 — Feature Engineering

Always apply, adapted to what the data actually contains:

- **Encode categoricals** — label-encode all low-cardinality string columns
- **Encode time** — convert time column to ordinal integer (sorted chronologically)
- **Encode text** — TF-IDF (max 50 features) on any free-text columns, fit on train only
- **Group statistics** — per (entity, grouping-column) compute mean/median/std of target from training data; merge onto both train and val
- **Lag features** — if time + entity exist: lag_1, lag_2, rolling_mean_3 of target per (entity, grouping-column), shifted 1 period to prevent leakage. For val rows: use last known training value.
- **Cross-group features** — if a grouping column exists: pivot other groups' last known target values as features per (entity, time), lag-shifted.
- **Interactions** — multiply related numeric pairs where domain logic suggests it (e.g. rate × population).

---

## Phase 5 — Establish Baselines

Before training any ML model, compute naive baselines using the same cross-validation folds you will use for models.

For regression: global mean, global median, per-(entity, group) mean.
For classification: majority class, stratified random.
For clustering: random assignment.

**Record baseline scores. Any model that does not beat the best baseline is rejected — use the baseline for submission instead.**

---

## Phase 6 — Train and Compare Models

Use the **same CV folds** for every model so scores are directly comparable.

CV strategy:
- Time-series: `TimeSeriesSplit` — never shuffle time data
- Classification: `StratifiedKFold`
- Regression without time: `KFold(shuffle=True)`

Train all applicable candidates:

**Regression:** LightGBM (MAE loss), LightGBM (MSE loss), RandomForest, ExtraTrees, Ridge, HuberRegressor
**Classification:** LightGBM, RandomForest, GradientBoosting, LogisticRegression
**Clustering:** KMeans (k=2..8 with elbow/silhouette), AgglomerativeClustering, DBSCAN

**If a grouping column exists:** train separate models per group. This is mandatory — a single global model cannot capture group-specific patterns. Each group gets its own fit/predict cycle with group-specific lag features.

Print a leaderboard of all models with their CV scores. Select the winner. If top-2 models are within 2% of each other, blend them and verify the ensemble improves the OOF score before using it.

---

## Phase 7 — Verify Predictions Before Submitting

Run these checks on your predictions. **If any fail, diagnose and fix before generating submission.csv.**

**For all task types:**
- No NaN predictions
- Prediction count matches submission template exactly
- Predictions are finite numbers

**For regression:**
- Predictions are non-negative (if target was non-negative in training)
- Prediction mean is within 2× of training target mean
- Prediction std is within 3× of training target std

**For tasks with a grouping column (critical):**
- Compute predictions grouped by the grouping column. Each group should have a distinct mean and distribution.
- Compute pairwise correlation between groups' predictions (pivoted by entity × time). If any pair exceeds 0.95 correlation, the model is not learning group-specific patterns — you must retrain with separate per-group models.

**For classification:**
- All predicted classes exist in the training set
- Class distribution in predictions is not wildly different from training

**For clustering:**
- No cluster has fewer than 2% of rows (degenerate clustering)
- Silhouette score is positive

If any check fails, return to Phase 6, diagnose the issue, fix it, and re-verify.

---

## Phase 8 — Generate submission.csv

Read the submission template file to learn the **exact expected format** — never assume it.

```python
template = pd.read_csv('data/<submission_template_file>')
print(template.columns.tolist())   # this defines exactly what to output
print(template.shape)
print(template.head(3))
```

Mirror the template exactly: same columns, same column order, same row count, same row_id values. Fill in your predictions for the target column(s). Write to `submission.csv` at the repo root.

Final assertion:
```python
original = pd.read_csv('data/<submission_template_file>')
result   = pd.read_csv('submission.csv')
assert list(result.columns) == list(original.columns)
assert len(result) == len(original)
assert result.isnull().sum().sum() == 0
```

---

## Phase 9 — Generate report.pdf

Install reportlab if needed. Write `report.pdf` to the repo root containing:

1. Task type and reasoning
2. Data quality findings and actions taken
3. Feature engineering summary
4. Model leaderboard — all candidates with CV scores vs baseline
5. Winner selection rationale
6. Feature importance (regression/classification) or cluster profiles (clustering)
7. Prediction distribution vs training target
8. Conclusion

---

## Error Recovery

For every error: read the full traceback → fix the specific cause → retry.

After 2 failed attempts at any step, fall back to a simpler approach:
- Complex features failing → use only numeric features with median imputation
- Per-group models failing → single global model (but flag it in the report)
- LightGBM unavailable → `pip install lightgbm -q`
- Any package unavailable → `pip install <package> -q`
- All models fail → submit the best baseline

Minimum acceptable output is `submission.csv` with non-NaN predictions. Even a baseline submission is better than no submission.

---

## Constraints

- Token budget: ~1,000,000 total. Never print full dataframes — use `.head(5)` and `.describe()`.
- Time budget: 2 hours. Skip EDA plots if running long; a minimal report is acceptable.
- No external data downloads.
- All code must run in the standard Python environment (install packages with pip as needed).

---

## Done

Print: `"Pipeline complete. Task: <type>. Winner: <model>. CV: <score> vs Baseline: <score>. submission.csv: <N> rows. report.pdf: generated."`

---

## Scoring Alignment — Critical

The organizers always evaluate using **block-averaged MAE**:

```
final_score = mean over each group of ( mean absolute error within that group )
```

Each group contributes **equally** to the final score regardless of how many rows it has. A group with 10 rows counts the same as a group with 1000 rows.

**This must be your CV metric too.** Do not use weighted MAE, RMSE, AUC, or F1 as the selection criterion — even for classification tasks where the submission values will be scored as continuous numbers.

Implement it like this:

```python
def block_averaged_mae(y_true, y_pred, groups):
    """
    groups : array-like of group labels (e.g. category column values)
             If no grouping column exists, pass None and this reduces to plain MAE.
    """
    import numpy as np
    if groups is None:
        return np.mean(np.abs(y_true - y_pred))
    unique_groups = np.unique(groups)
    group_maes = []
    for g in unique_groups:
        mask = groups == g
        if mask.sum() == 0:
            continue
        group_maes.append(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    return np.mean(group_maes)   # equal weight per group
```

Use this function to:
1. Compute baseline scores (Phase 5)
2. Evaluate every candidate model in CV (Phase 6)
3. Select the winner — lowest block-averaged MAE wins
4. Report the final CV score in the pipeline completion message

If no grouping column exists, this reduces to plain MAE — still correct.