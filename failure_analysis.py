import pandas as pd

golden = pd.read_csv("golden_set_final.csv")
candidates = pd.read_csv("local_golden_candidates.csv")

# Take only the agent prediction columns
predictions = candidates[
    ["tweet_id", "suggested_intent", "suggested_decision"]
].copy()

# Remove duplicate prediction columns from golden if present
for col in ["suggested_intent", "suggested_decision"]:
    if col in golden.columns:
        golden = golden.drop(columns=[col])

# Merge
df = golden.merge(
    predictions,
    on="tweet_id",
    how="left"
)

# Find intent mistakes
failures = df[
    df["human_intent"] != df["suggested_intent"]
].copy()

print("\n================================")
print("       FAILURE ANALYSIS")
print("================================")

print("\nTotal intent failures:", len(failures))

if len(failures) > 0:

    for i, (_, row) in enumerate(
        failures.head(5).iterrows(), 1
    ):

        print(f"\n--- Failure {i} ---")

        print("\nCustomer:")
        print(row["message"])

        print("\nHuman Intent:")
        print(row["human_intent"])

        print("\nAgent Intent:")
        print(row["suggested_intent"])

        print("\nHistorical Reply:")
        print(row["historical_reply"])

else:
    print("No intent failures found.")

failures.to_csv(
    "failure_analysis.csv",
    index=False
)

print("\n✅ Saved to failure_analysis.csv")