from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PROJECT CONFIGURATION
# These values and functions are copied from the notebook's
# configuration/helper-functions section (no common.py needed).
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "income"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RAW_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    TARGET,
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

METRIC_NAMES = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]


def normalize_labels(df):
    """Match the notebook's label and categorical-value normalization."""
    df = df.copy()

    for column in CATEGORICAL_FEATURES + [TARGET]:
        df[column] = df[column].astype(str).str.strip()

    # The UCI test data uses labels such as '>50K.'.
    df[TARGET] = df[TARGET].str.rstrip(".")

    # Match the notebook: '?' in categorical predictors becomes 'Unknown'.
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].replace(
        "?", "Unknown"
    )

    return df


def positive_scores(model, X):
    """Use probabilities when available, otherwise decision scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return model.decision_function(X)


def evaluate_model(model, X, y_true):
    """Calculate the same six metrics as the notebook."""
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# Model descriptions reflect the classifiers in notebook cell 21.
MODEL_NOTES = {
    "Logistic Regression": (
        "LogisticRegression(max_iter=2000), using StandardScaler for numeric "
        "features and OneHotEncoder(handle_unknown='ignore', "
        "sparse_output=False) for categorical features."
    ),
    "Decision Tree": (
        "DecisionTreeClassifier(random_state=42), using the same preprocessing "
        "pipeline as the other classifiers."
    ),
    "KNN": (
        "KNeighborsClassifier(n_neighbors=5), using the same preprocessing "
        "pipeline. The saved model stores the training data."
    ),
    "Naive Bayes": (
        "GaussianNB, using the same preprocessing pipeline as the other "
        "classifiers."
    ),
    "SVM": (
        "LinearSVC(C=1.0, max_iter=3000, random_state=42), using the same "
        "preprocessing pipeline. AUC uses decision_function scores because "
        "LinearSVC does not provide predict_proba."
    ),
}


# ============================================================
# LOAD NOTEBOOK OUTPUTS
# ============================================================

@st.cache_resource
def load_models():
    """Load the five trained pipelines saved by the notebook."""
    missing_files = [
        filename
        for filename in MODEL_FILES.values()
        if not (MODEL_DIR / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing trained model file(s) in the model/ folder: "
            + ", ".join(missing_files)
            + ". Run the model-saving cell in ML_Assignment_2.ipynb first."
        )

    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


@st.cache_data
def load_bundled_test():
    if not TEST_CSV.is_file():
        raise FileNotFoundError(
            "test_data.csv was not found beside app.py. Run the notebook's "
            "final export cell to generate it."
        )

    return pd.read_csv(TEST_CSV, skipinitialspace=True)


@st.cache_data
def bundled_test_bytes():
    return TEST_CSV.read_bytes()


# ============================================================
# INPUT VALIDATION
# ============================================================

def prepare(raw):
    """Validate the CSV and prepare X/y in the notebook's feature order."""
    required_columns = [TARGET, *FEATURES]
    missing_columns = [
        column for column in required_columns if column not in raw.columns
    ]

    if missing_columns:
        return None, (
            "Missing required columns: " + ", ".join(missing_columns)
        )

    # Ignore additional columns and preserve the notebook's feature order.
    df = normalize_labels(raw[required_columns])

    valid_labels = {NEGATIVE_LABEL, POSITIVE_LABEL}
    found_labels = set(df[TARGET].dropna())
    unknown_labels = sorted(found_labels - valid_labels)

    if unknown_labels:
        return None, (
            f"'{TARGET}' must contain only '{NEGATIVE_LABEL}' or "
            f"'{POSITIVE_LABEL}'. Found: {', '.join(unknown_labels)}"
        )

    if df[TARGET].isna().any():
        return None, f"The '{TARGET}' column contains missing values."

    X = df[FEATURES].copy()

    try:
        X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(
            pd.to_numeric, errors="raise"
        )
    except (ValueError, TypeError):
        return None, (
            "Numeric feature columns must contain numbers only: "
            + ", ".join(NUMERIC_FEATURES)
        )

    if X[NUMERIC_FEATURES].isna().any().any():
        return None, "Numeric feature columns must not contain missing values."

    if X.isna().any().any():
        return None, (
            "Feature columns contain missing values. For unknown categorical "
            "values, use '?' so they can be converted to 'Unknown'."
        )

    y = (df[TARGET] == POSITIVE_LABEL).astype(int)

    if y.nunique() < 2:
        return None, (
            f"The test data must contain both '{NEGATIVE_LABEL}' and "
            f"'{POSITIVE_LABEL}' rows to calculate AUC."
        )

    return (X, y), None


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.set_page_config(
    page_title="Adult Income Classifiers",
    page_icon="📊",
    layout="wide",
)

st.title("Adult Income (>50K): Classifier Comparison")

st.info(
    "This app evaluates the five trained pipelines saved from "
    "ML_Assignment_2.ipynb. Models are loaded from the model/ folder; "
    "they are not retrained in the app."
)

with st.expander("Expected CSV format"):
    st.write(
        f"Each row must contain the target column `{TARGET}` "
        f"(`{NEGATIVE_LABEL}` or `{POSITIVE_LABEL}`) and all "
        f"{len(FEATURES)} predictor columns."
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Column": FEATURES,
                "Type": [
                    "Numeric" if column in NUMERIC_FEATURES else "Categorical"
                    for column in FEATURES
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Extra columns are ignored. Numeric columns must contain valid "
        "numbers and cannot be missing. A '?' in a categorical column is "
        "converted to 'Unknown'. A trailing period in a target label, "
        "such as '>50K.', is accepted."
    )

    if TEST_CSV.is_file():
        st.download_button(
            "Download sample test_data.csv",
            data=bundled_test_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "Bundled test_data.csv is not present. Generate it using the "
            "final export cell in the notebook."
        )


data_source = st.radio(
    "Test data",
    ["Bundled test_data.csv", "Upload CSV"],
    horizontal=True,
)

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader(
        f"Upload a CSV containing '{TARGET}' and the {len(FEATURES)} "
        "feature columns.",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to see the model results.")
        st.stop()

    try:
        raw_data = pd.read_csv(uploaded_file, skipinitialspace=True)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    try:
        raw_data = load_bundled_test()
    except Exception as exc:
        st.error(str(exc))
        st.stop()


prepared_data, validation_error = prepare(raw_data)

if validation_error:
    st.error(validation_error)
    st.stop()

X, y = prepared_data

try:
    models = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()


st.caption(
    f"Evaluating {len(y):,} test rows: "
    f"{(y == 0).sum():,} '{NEGATIVE_LABEL}' and "
    f"{(y == 1).sum():,} '{POSITIVE_LABEL}'."
)

selected_model = st.selectbox(
    "Model",
    ["All models", *MODEL_FILES.keys()],
)


# ============================================================
# ALL-MODEL COMPARISON
# ============================================================

if selected_model == "All models":
    results = pd.DataFrame(
        {
            name: evaluate_model(model, X, y)
            for name, model in models.items()
        }
    ).T

    st.subheader(f"Metrics on {len(y):,} test rows")
    st.dataframe(
        results[METRIC_NAMES].round(4),
        use_container_width=True,
    )

    st.caption(
        "The six metrics use the same definitions as the notebook. "
        "AUC uses predicted probabilities where available and decision "
        "scores for LinearSVC."
    )


# ============================================================
# INDIVIDUAL-MODEL DETAILS
# ============================================================

else:
    model = models[selected_model]

    st.subheader(selected_model)
    st.caption(MODEL_NOTES[selected_model])

    metrics = evaluate_model(model, X, y)

    metric_columns = st.columns(len(METRIC_NAMES))
    for column, metric_name in zip(metric_columns, METRIC_NAMES):
        column.metric(metric_name, f"{metrics[metric_name]:.4f}")

    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    row_totals = cm.sum(axis=1, keepdims=True)
    row_percentages = (cm / row_totals.clip(min=1)) * 100

    matrix_cells = [
        [
            f"{cm[row, col]:,} ({row_percentages[row, col]:.1f}%)"
            for col in range(2)
        ]
        for row in range(2)
    ]

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Confusion matrix")
        st.dataframe(
            pd.DataFrame(
                matrix_cells,
                index=[
                    f"Actual {NEGATIVE_LABEL}",
                    f"Actual {POSITIVE_LABEL}",
                ],
                columns=[
                    f"Predicted {NEGATIVE_LABEL}",
                    f"Predicted {POSITIVE_LABEL}",
                ],
            ),
            use_container_width=True,
        )
        st.caption(
            "Each cell shows the count and percentage of its actual-class row."
        )

    with right_column:
        st.subheader("Classification report")
        report = classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
            digits=4,
            zero_division=0,
        )
        st.code(report, language="text")
