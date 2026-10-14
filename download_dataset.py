from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing

out = Path("data")
out.mkdir(exist_ok=True)
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
df["MedHouseVal_USD"] = df["MedHouseVal"] * 100_000
df.to_csv(out / "california_housing.csv", index=False)
print(f"Saved {len(df):,} records to data/california_housing.csv")
print("Source: California Housing (1990 U.S. Census-derived census block groups).")
print("Not current individual property listings; use USD and explain this limitation.")
