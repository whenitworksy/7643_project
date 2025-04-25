import pandas as pd
import json

# Replace with your actual path
with open("data/recipe1m_images/det_ingrs.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Show structure
print(df.columns)
print(df.head())