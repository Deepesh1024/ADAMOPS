import pandas as pd

# Load dataset
df = pd.read_csv("data/titanic.csv")

# Identify all columns containing alphabetic data (object or string types)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Keep only 'Sex' column among categorical ones
cols_to_drop = [col for col in categorical_cols if col != 'Sex']

# Drop unwanted categorical columns
df.drop(columns=cols_to_drop, inplace=True)

# Save permanent change to the same file
df.to_csv("data/titanic.csv", index=False)

print("Removed columns:", cols_to_drop)
print("Remaining columns:", df.columns.tolist())
