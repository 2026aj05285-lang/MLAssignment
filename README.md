# Adult Income Classification: Five-Model Comparison

## 1. Project Overview

This project uses the **UCI Adult Census Income dataset** to predict whether a person's annual income is greater than `$50K`.

It trains and compares five classification algorithms using the same training/test split and preprocessing pipeline. The trained preprocessing-plus-model pipelines are saved as Joblib files and can be evaluated through a Streamlit application.

- **Target column:** `income`
- **Positive class:** `>50K` (encoded as `1`)
- **Negative class:** `<=50K` (encoded as `0`)
- **Input features:** 14
- **Models:** Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine

The Streamlit application loads the saved models and evaluates them on a labelled CSV file. **The application does not train or retrain models.**

## 2. Dataset

**Dataset:** Adult (Census Income)  
**Source:** [UCI Machine Learning Repository – Adult](https://archive.ics.uci.edu/dataset/2/adult)

The repository uses the official UCI training and test files:

| File | Rows |
|---|---:|
| `data/adult.data` | 32,561 |
| `data/adult.test` | 16,281 |
| **Total** | **48,842** |

The dataset contains 14 predictor features and one target column (`income`).

### Feature columns

**Numerical features (6):**
- `age`
- `fnlwgt`
- `education_num`
- `capital_gain`
- `capital_loss`
- `hours_per_week`

**Categorical features (8):**
- `workclass`
- `education`
- `marital_status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native_country`

### Data preparation

The notebook applies the following preparation steps:

1. Loads the official training and test files using the original column names.
2. Strips leading and trailing whitespace from categorical values and target labels.
3. Replaces categorical `?` placeholders with `Unknown`.
4. Removes the trailing period from test-set labels such as `>50K.`.
5. Encodes `>50K` as `1` and `<=50K` as `0`.
6. Standardizes numerical features using `StandardScaler`.
7. One-hot encodes categorical features using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
8. Uses a `ColumnTransformer` inside each scikit-learn `Pipeline`.

The preprocessing steps are fitted on the training data only. The held-out test data is used for evaluation, not model fitting.

**Dataset reference:** Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## 3. Models

All five models use the same training split and preprocessing approach.

| Model | Classifier configuration |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=2000)` |
| Decision Tree | `DecisionTreeClassifier(random_state=42)` |
| KNN | `KNeighborsClassifier(n_neighbors=5)` |
| Naive Bayes | `GaussianNB()` |
| SVM | `LinearSVC(C=1.0, max_iter=3000, random_state=42)` |

Each saved `.joblib` file contains the complete fitted pipeline, including preprocessing and the classifier.

## 4. Evaluation Metrics

The models are evaluated using six metrics:

| Metric | Description |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| AUC | Area under the ROC curve, calculated from positive-class scores |
| Precision | Proportion of predicted positive cases that are actually positive |
| Recall | Proportion of actual positive cases correctly identified |
| F1 | Harmonic mean of precision and recall |
| MCC | Matthews Correlation Coefficient, a balanced measure of binary classification quality |

For models that support `predict_proba()`, AUC uses the predicted probability for the positive class. For `LinearSVC`, which does not provide `predict_proba()`, AUC uses `decision_function()` scores.

### Results on the official test split

The following values are recorded in the executed notebook. Values are rounded to four decimal places.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8530 | 0.9055 | 0.7306 | 0.5985 | 0.6580 | 0.5701 |
| Decision Tree | 0.8082 | 0.7405 | 0.5909 | 0.6121 | 0.6013 | 0.4752 |
| KNN | 0.8332 | 0.8579 | 0.6622 | 0.5998 | 0.6295 | 0.5232 |
| Naive Bayes | 0.5532 | 0.7639 | 0.3389 | 0.9376 | 0.4979 | 0.3320 |
| SVM (LinearSVC) | 0.8540 | 0.9048 | 0.7387 | 0.5910 | 0.6567 | 0.5711 |

### Observations

- **SVM (LinearSVC)** has the highest accuracy, precision, and MCC in this recorded evaluation.
- **Logistic Regression** has the highest AUC and F1 score.
- **Naive Bayes** has the highest recall, but its precision and accuracy are substantially lower, indicating many false-positive predictions.
- The majority-class baseline accuracy on the test split is approximately `0.7638`. Accuracy should therefore be considered alongside AUC, precision, recall, F1, and MCC.

These results describe this particular held-out test split; they do not guarantee the same performance on other data.

## 5. Repository Structure

```text
MLAssignment/
├── ML_Assignment_2.ipynb
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
│
├── data/
│   ├── adult.data
│   └── adult.test
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

### Main files and folders

- `ML_Assignment_2.ipynb` — data preparation, model training, evaluation, model export, and saved-model verification.
- `app.py` — Streamlit application for evaluating the saved models on labelled CSV data.
- `requirements.txt` — Python package dependencies.
- `data/` — original UCI Adult training and test data files.
- `model/` — five saved fitted pipelines.
- `test_data.csv` — exported labelled test data used by the application and available as a sample download.

The notebook contains its own configuration, constants, and helper functions. A separate `common.py` file is not required by the current project.

## 6. Setup

Use Python in a virtual environment if possible.

### Windows PowerShell

Open a terminal in the repository root:

```powershell
cd C:\Users\91949\MLAssignment
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can install and run the project using the `python` executable without activating the environment.

## 7. Run the Notebook and Generate Model Files

The repository includes the five saved model files and `test_data.csv`. If you need to recreate them:

1. Confirm that `data/adult.data` and `data/adult.test` are present.
2. Open `ML_Assignment_2.ipynb` from the repository root.
3. Run the notebook cells in order.
4. Run the model export section.

The export section creates the `model/` directory if needed, saves the five fitted pipelines with `joblib.dump()`, and exports the labelled test data to `test_data.csv`.

The notebook then reloads the saved pipelines and checks that their evaluation metrics match the in-memory model results.

Expected model files:

```text
model/logistic_regression.joblib
model/decision_tree.joblib
model/knn.joblib
model/naive_bayes.joblib
model/svm.joblib
```

## 8. Run the Streamlit Application

From the repository root, run:

```powershell
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

### Application workflow

1. Choose the bundled `test_data.csv` or upload another CSV.
2. The application checks that all 14 feature columns and the `income` target column are present.
3. The target labels are validated and normalized.
4. The application separates the predictors (`X`) and true labels (`y`).
5. Select one of the five saved models, or choose **All models** for a comparison.
6. The selected fitted pipeline is loaded from the `model/` directory using `joblib.load()`.
7. Predictions and positive-class scores are generated.
8. The application calculates Accuracy, AUC, Precision, Recall, F1, and MCC.
9. For an individual model, the application displays six metric values, a confusion matrix with counts and row percentages, and a classification report.
10. For **All models**, the application displays a six-metric comparison table.

### CSV format required by the application

An uploaded CSV must include these 15 columns:

```text
age
fnlwgt
education_num
capital_gain
capital_loss
hours_per_week
workclass
education
marital_status
occupation
relationship
race
sex
native_country
income
```

- Numerical columns must contain numeric values.
- Categorical columns should contain text values.
- The target column must contain `<=50K` or `>50K`. A trailing period, as in `>50K.`, is accepted.
- Categorical `?` values are converted to `Unknown`.
- Extra columns are ignored.
- The uploaded CSV must include the true target labels because the application calculates evaluation metrics; this is not an unlabeled prediction-only workflow.

### No retraining in the application

The Streamlit application only loads the already trained `.joblib` pipelines and evaluates them. It does not call `fit()` and does not retrain or update any saved model. Uploaded data is used for evaluation only.

## 9. Requirements

The project dependencies are listed in `requirements.txt`. Install them with:

```powershell
python -m pip install -r requirements.txt
```

The application and notebook use Python libraries including Streamlit, pandas, NumPy, scikit-learn, Joblib, Matplotlib, and Seaborn.

## 10. Troubleshooting

**`FileNotFoundError` for a model**

- Confirm that the five `.joblib` files exist in the `model/` directory.
- Run the notebook's model export section if the files are missing.
- Start Streamlit from the repository root.

**`test_data.csv` not found**

- Confirm that `test_data.csv` is beside `app.py`.
- Run the notebook's final export section to recreate it.

**Missing required columns**

- Check the uploaded CSV header against the 15 required column names listed above.
- Ensure the target column is named `income`.

**Invalid numeric values**

- Check the six numerical columns for text values or blank cells.

**Package or version errors**

- Activate the intended Python environment.
- Reinstall dependencies using `python -m pip install -r requirements.txt`.

## 11. GitHub Repository

Repository: https://github.com/2026aj05285-lang/MLAssignment

A deployed Streamlit URL can be added here after the application is deployed.

---

**Project summary:** This project trains and compares five classification pipelines on the UCI Adult Income dataset, saves the fitted pipelines using Joblib, verifies that the saved models reproduce the notebook evaluation results, and provides a Streamlit interface to evaluate those saved models on labelled CSV data.
