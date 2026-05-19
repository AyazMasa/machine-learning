import pandas as pd
from pathlib import Path

# Load the dataset
csv_file_path = Path(__file__).resolve().parent / "booksData.csv"
df = pd.read_csv(csv_file_path)

# Display first few rows
print(df.head())
