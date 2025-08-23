import pandas as pd
import json
from pathlib import Path

# Load version metadata
with open("version_meta.json") as f:
    meta = json.load(f)

raw_data = meta["raw_data"]
version = meta["version_tag"]

# Load raw data
df = pd.read_csv(raw_data)

# Sample 100 rows for batch input
df_sample = df.sample(n=100, random_state=42)

# Save input CSV
batch_input_dir = Path("data/batch")
batch_input_dir.mkdir(parents=True, exist_ok=True)
batch_input_file = batch_input_dir / f"input_batch_{version}.csv"
df_sample.to_csv(batch_input_file, index=False)

print(f"Batch input generated: {batch_input_file}")
