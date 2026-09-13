import pandas as pd

print("\n==============================================")
print("       AI CUSTOMER SUPPORT AGENT")
print("          FINAL EVALUATION RESULTS")
print("==============================================")

print("\n1. DATASET")
print("----------------------------------------------")
print("Golden Set Size        : 150")
print("Intent Categories      : 5")
print("Observed Intent Errors : 1")
print("Observed Error Rate    : 0.67%")

print("\n2. INTENT DISTRIBUTION")
print("----------------------------------------------")

intent_counts = {
    "general_issue": 55,
    "update_problem": 53,
    "battery_issue": 16,
    "app_feature_problem": 15,
    "device_hardware_issue": 11
}

for intent, count in intent_counts.items():
    percentage = (count / 150) * 100
    print(f"{intent:25} {count:3} ({percentage:.1f}%)")

print("\n3. MODEL COMPARISON")
print("----------------------------------------------")

comparison = pd.read_csv("system_comparison.csv")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\n4. AI AGENT PERFORMANCE")
print("----------------------------------------------")
print("Intent Accuracy        : 99.3%")
print("Macro Precision        : 99.6%")
print("Macro Recall           : 98.8%")
print("Macro F1               : 99.2%")
print("Decision Accuracy      : 99.3%")
print("Decision Macro F1      : 99.3%")

print("\n5. FAILURE ANALYSIS")
print("----------------------------------------------")
print("Observed failures     : 1")
print("Failure type          : Image/link-only message")
print("Human intent          : battery_issue")
print("Agent intent          : general_issue")

print("\n6. IMPORTANT LIMITATION")
print("----------------------------------------------")
print(
    "The 99.3% result is based on a reviewed 150-example "
    "golden set. Candidate generation was AI-assisted, "
    "so this should be treated as an optimistic evaluation "
    "rather than an unbiased production estimate."
)

print("\n7. REPLY EVALUATION")
print("----------------------------------------------")
print(
    "Full LLM-as-judge evaluation was not completed because "
    "the API request-per-day limit was reached."
)

print("\n==============================================")
print("             EVALUATION COMPLETE")
print("==============================================")