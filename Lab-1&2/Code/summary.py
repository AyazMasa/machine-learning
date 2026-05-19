import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the dataset
file_path = Path(__file__).resolve().parent / "booksData.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# 🔹 Fixing the Price column
df['Price'] = df['Price'].astype(str)  # Convert to string to avoid errors
df['Price'] = df['Price'].str.replace(r'[^\d.]', '', regex=True)  # Remove all non-numeric characters
df['Price'] = df['Price'].astype(float)  # Convert to float

#  Correcting the Rating column
df.rename(columns={"Rating (1-5)": "Rating"}, inplace=True)

# Check if rating is now correct
print("\nFirst 5 Rows After Fixing Rating:\n", df[['Rating']].head())

# Summary statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# Availability counts
availability_counts = df['Availability'].value_counts()
print("\nAvailability Counts:\n", availability_counts)

# Rating distribution
rating_counts = df['Rating'].value_counts()
print("\nRating Distribution:\n", rating_counts)

#  Plot price distribution (Histogram)
plt.figure(figsize=(8, 5))
df['Price'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Book Prices')
plt.xlabel('Price (£)')
plt.ylabel('Frequency')
plt.show()

#  Plot rating distribution (Bar Chart)
if not df['Rating'].isna().all():
    plt.figure(figsize=(8, 5))
    df['Rating'].dropna().astype(int).value_counts().sort_index().plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Book Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

#  Create a box plot for book prices
plt.figure(figsize=(8, 5))
sns.boxplot(y=df['Price'], color='skyblue')
plt.title('Box Plot of Book Prices')
plt.ylabel('Price (£)')
plt.show()

#  Scatter Plot of Price vs Rating
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Rating'], y=df['Price'], color='orange', edgecolor='black')
plt.title('Scatter Plot of Price vs Rating')
plt.xlabel('Rating')
plt.ylabel('Price (£)')
plt.show()
