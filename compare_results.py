import pandas as pd

# Load baseline results
baseline = pd.read_csv("evaluation_results.csv")

# Load AI agent results
agent = pd.read_csv("agent_evaluation_results.csv")

# Get values
majority = baseline[baseline["Model"] == "Majority Baseline"].iloc[0]
tfidf = baseline[baseline["Model"] == "TF-IDF + Logistic Regression"].iloc[0]

agent_intent = agent[agent["Metric"] == "Intent Accuracy"]["Score"].iloc[0]
agent_f1 = agent[agent["Metric"] == "Intent Macro F1"]["Score"].iloc[0]
decision_acc = agent[agent["Metric"] == "Decision Accuracy"]["Score"].iloc[0]

results = pd.DataFrame({
    "System": [
        "Majority Baseline",
        "TF-IDF + Logistic Regression",
        "Our AI Agent"
    ],
    "Intent Accuracy": [
        majority["Accuracy"],
        tfidf["Accuracy"],
        agent_intent
    ],
    "Macro F1": [
        majority["Macro F1"],
        tfidf["Macro F1"],
        agent_f1
    ],
    "Decision Accuracy": [
        "-",
        "-",
        decision_acc
    ]
})

print("\n==========================================")
print("        SYSTEM COMPARISON")
print("==========================================")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

results.to_csv(
    "system_comparison.csv",
    index=False
)

print("\n✅ Saved to system_comparison.csv")