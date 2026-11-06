# Adult Income (>50K): Five-Model Classification

## a. Problem statement

Predict whether a person's annual income exceeds $50K using demographic and employment-related attributes from the UCI Adult Census Income dataset. Five classification algorithms are trained and evaluated on the same official training/test split.

The positive class is `>50K` and the negative class is `<=50K`.

## b. Dataset description

- **Dataset:** Adult (Census Income)
- **Source:** [UCI Machine Learning Repository – Adult](https://archive.ics.uci.edu/dataset/2/adult)
- **Training data:** 32,561 rows (`adult.data`)
- **Test data:** 16,281 rows (`adult.test`)
- **Total instances:** 48,842
- **Input features:** 14 (6 numerical and 8 categorical)
- **Target:** `income` (binary classification)
- **Positive class:** `>50K` (encoded as 1)
- **Negative class:** `<=50K` (encoded as 0)

The dataset exceeds the assignment minimum of 500 instances and 12 input features. The official UCI train/test split is used. The test split is held out from model fitting.

Categorical `?` placeholders are converted to `Unknown`; the trailing period in labels in `adult.test` is removed. No rows are dropped for these placeholders.

**Dataset citation:** Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Preprocessing methodology

1. The official training and test files are loaded with the 15 original column names.
2. Whitespace is stripped from categorical values and target labels. `?` is replaced with `Unknown`, and the period at the end of test-set income labels is removed.
3. The target is encoded as 1 for `>50K` and 0 for `<=50K`.
4. Numerical features are transformed using `StandardScaler`.
5. Categorical features are transformed using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
6. The transformations are placed inside a `ColumnTransformer`, which is included in each model's scikit-learn `Pipeline`. Each pipeline is fitted only on the training split.

The notebook reports 108 transformed columns. Keeping preprocessing inside each pipeline ensures that the saved pipeline can accept the original 14 input columns.

## c. GitHub Repository

[MLAssignment repository](https://github.com/2026aj05285-lang/MLAssignment)

**Live Streamlit app:** Add the deployed application URL after deployment.

## d. Models used

All five models use the same training data, test data and preprocessing approach.

| # | Model | Configuration |
|---|---|---|
| 1 | Logistic Regression | `max_iter=2000` |
| 2 | Decision Tree | `random_state=42` |
| 3 | K-Nearest Neighbors | `n_neighbors=5` |
| 4 | Naive Bayes | `GaussianNB()` |
| 5 | Support Vector Machine | `LinearSVC(C=1.0, max_iter=3000, random_state=42)` |

### Comparison table

The following are the values recorded in the executed notebook, evaluated on the 16,281-row official test split. Values are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8530 | 0.9055 | 0.7306 | 0.5985 | 0.6580 | 0.5701 |
| Decision Tree | 0.8082 | 0.7405 | 0.5909 | 0.6121 | 0.6013 | 0.4752 |
| KNN | 0.8332 | 0.8579 | 0.6622 | 0.5998 | 0.6295 | 0.5232 |
| Naive Bayes | 0.5532 | 0.7639 | 0.3389 | 0.9376 | 0.4979 | 0.3320 |
| Support Vector Machine (LinearSVC) | 0.8540 | 0.9048 | 0.7387 | 0.5910 | 0.6567 | 0.5711 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves the highest AUC (0.9055) and highest F1 score (0.6580). Its precision (0.7306) is higher than its recall (0.5985), indicating that it is comparatively conservative when predicting the positive income class. |
| Decision Tree | Has the lowest AUC (0.7405) and lowest MCC (0.4752). Its precision (0.5909) and recall (0.6121) are relatively close, but its overall performance is below the other main models on this test split. |
| KNN | Produces middle-range results, with accuracy 0.8332, AUC 0.8579 and MCC 0.5232. Its precision is 0.6622 and recall is 0.5998. |
| Naive Bayes | Has the highest recall (0.9376), but its precision is only 0.3389. Its accuracy (0.5532) is below the majority-class baseline accuracy of 0.7638. It identifies most positive cases but also produces many false-positive predictions. |
| Support Vector Machine (LinearSVC) | Achieves the highest accuracy (0.8540), precision (0.7387) and MCC (0.5711). Its AUC (0.9048) and F1 (0.6567) are slightly below Logistic Regression. |
| **Overall winner for this dataset** | **Support Vector Machine (LinearSVC)**, based on its highest MCC and its lead in accuracy and precision. Logistic Regression is a close alternative and leads in AUC and F1. Naive Bayes is preferable only if prioritising recall despite its low precision and accuracy. |

The majority-class baseline accuracy on the test set is 0.7638. Accuracy should therefore not be interpreted alone; AUC, precision, recall, F1 and MCC provide additional information about model behaviour.

## Model export and verification

The notebook creates a `model/` directory and exports each fitted preprocessing-plus-classifier pipeline using `joblib`:

- `model/logistic_regression.joblib`
- `model/decision_tree.joblib`
- `model/knn.joblib`
- `model/naive_bayes.joblib`
- `model/svm.joblib`

It also exports `test_data.csv`, containing the held-out test features and their true `income` labels.

The notebook reloads all five saved pipelines and evaluates them on the exported test data. Its recorded output reports a largest metric difference of `0.0` between the in-memory and reloaded models. This is the notebook's saved execution result; rerun the export and verification cells in the current environment to recreate and verify the files there.

## Repository layout

The completed project should contain:

```text
MLAssignment/
├── ML_Assignment_21.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── adult.data
│   └── adult.test
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

The notebook defines its shared constants and helper functions in the notebook itself; a separate `common.py` file is not required by the notebook as currently written.

The Streamlit application file (`app.py`) and live application URL can be added when that part of the assignment is completed.

## Run the notebook

1. Install the packages listed in `requirements.txt`.
2. Place the UCI `adult.data` and `adult.test` files in the `data/` directory.
3. Open `ML_Assignment_21.ipynb` from the project root.
4. Run the notebook from the beginning, including the final model-export and saved-model verification cells.
5. Confirm that the `model/` directory contains all five `.joblib` files and that `test_data.csv` is generated.

