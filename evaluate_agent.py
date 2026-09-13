import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load files
golden = pd.read_csv("golden_set_final.csv")
candidates = pd.read_csv("local_golden_candidates.csv")

# Keep only prediction columns from candidates
predictions = candidates[
    ["tweet_id", "suggested_intent", "suggested_decision"]
].copy()

# Merge
df = golden.merge(
    predictions,
    on="tweet_id",
    how="left",
    suffixes=("_human", "_agent")
)

# Valid 5 intents
valid_intents = [
    "battery_issue",
    "update_problem",
    "app_feature_problem",
    "device_hardware_issue",
    "general_issue"
]

df = df[df["human_intent"].isin(valid_intents)].copy()

print("================================")
print("      AI AGENT EVALUATION")
print("================================")

print("\nTotal examples:", len(df))

# =============================
# INTENT EVALUATION
# =============================

y_true = df["human_intent"]
y_pred = df["suggested_intent_agent"]

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n========== INTENT RESULTS ==========")

print(f"Accuracy        : {accuracy:.3f}")
print(f"Macro Precision : {precision:.3f}")
print(f"Macro Recall    : {recall:.3f}")
print(f"Macro F1        : {f1:.3f}")

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

# =============================
# CONFUSION MATRIX
# =============================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=valid_intents
)

print("\n========== CONFUSION MATRIX ==========")

print("Labels:")
print(valid_intents)

print("\nMatrix:")
print(cm)

# =============================
# ESCALATION EVALUATION
# =============================

decision_true = df["human_decision"]
decision_pred = df["suggested_decision_agent"]

decision_accuracy = accuracy_score(
    decision_true,
    decision_pred
)

decision_f1 = f1_score(
    decision_true,
    decision_pred,
    average="macro",
    zero_division=0
)

print("\n========== ESCALATION RESULTS ==========")

print(f"Decision Accuracy : {decision_accuracy:.3f}")
print(f"Decision Macro F1 : {decision_f1:.3f}")

# =============================
# SAVE RESULTS
# =============================

results = pd.DataFrame({
    "Metric": [
        "Intent Accuracy",
        "Intent Macro Precision",
        "Intent Macro Recall",
        "Intent Macro F1",
        "Decision Accuracy",
        "Decision Macro F1"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        decision_accuracy,
        decision_f1
    ]
})

results.to_csv(
    "agent_evaluation_results.csv",
    index=False
)

print("\n✅ Results saved to agent_evaluation_results.csv")