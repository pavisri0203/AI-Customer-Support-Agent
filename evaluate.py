import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------
# Load Golden Set
# ---------------------------------------

df = pd.read_csv("golden_set_final.csv")

# Keep only our 5 final intents
valid_intents = [
    "battery_issue",
    "update_problem",
    "app_feature_problem",
    "device_hardware_issue",
    "general_issue"
]

df = df[df["human_intent"].isin(valid_intents)].copy()

X = df["message"].fillna("")
y = df["human_intent"]

print("Total evaluation examples:", len(df))

# ---------------------------------------
# Split data
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# BASELINE 1
# Majority Class
# ---------------------------------------

majority_class = y_train.value_counts().idxmax()

majority_predictions = [
    majority_class
    for _ in range(len(y_test))
]

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions
)

majority_precision = precision_score(
    y_test,
    majority_predictions,
    average="macro",
    zero_division=0
)

majority_recall = recall_score(
    y_test,
    majority_predictions,
    average="macro",
    zero_division=0
)

majority_f1 = f1_score(
    y_test,
    majority_predictions,
    average="macro",
    zero_division=0
)

# ---------------------------------------
# BASELINE 2
# TF-IDF + Logistic Regression
# ---------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

tfidf_predictions = model.predict(
    X_test_tfidf
)

tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_predictions
)

tfidf_precision = precision_score(
    y_test,
    tfidf_predictions,
    average="macro",
    zero_division=0
)

tfidf_recall = recall_score(
    y_test,
    tfidf_predictions,
    average="macro",
    zero_division=0
)

tfidf_f1 = f1_score(
    y_test,
    tfidf_predictions,
    average="macro",
    zero_division=0
)

# ---------------------------------------
# RESULTS
# ---------------------------------------

results = pd.DataFrame({
    "Model": [
        "Majority Baseline",
        "TF-IDF + Logistic Regression"
    ],
    "Accuracy": [
        majority_accuracy,
        tfidf_accuracy
    ],
    "Macro Precision": [
        majority_precision,
        tfidf_precision
    ],
    "Macro Recall": [
        majority_recall,
        tfidf_recall
    ],
    "Macro F1": [
        majority_f1,
        tfidf_f1
    ]
})

print("\n==============================")
print("       EVALUATION RESULTS")
print("==============================")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

# ---------------------------------------
# TF-IDF Detailed Report
# ---------------------------------------

print("\n==============================")
print(" TF-IDF CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        tfidf_predictions,
        zero_division=0
    )
)

# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

print("\n==============================")
print(" CONFUSION MATRIX")
print("==============================")

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    tfidf_predictions,
    labels=labels
)

print("Labels:")
print(labels)

print("\nMatrix:")
print(cm)

# ---------------------------------------
# Save results
# ---------------------------------------

results.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\n✅ Results saved to evaluation_results.csv")