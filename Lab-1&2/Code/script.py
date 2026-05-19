from bs4 import BeautifulSoup
import requests
import csv

# Base URL for book pages
base_url = 'https://books.toscrape.com/catalogue/page-{}.html'

# Star rating conversion dictionary
rating_conversion = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

# Initialize empty list to store book data
data = []
page_number = 1  
min_data_count = 300  # Collect at least 300 books

while len(data) < min_data_count:
    url = base_url.format(page_number)  # Format URL for pagination
    response = requests.get(url)

    # If the page does not exist, break the loop
    if response.status_code != 200:
        break

    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all book elements
    books = soup.find_all('article', class_='product_pod')

    # If no more books are found, stop the loop
    if not books:
        break

    # Extract book details
    for book in books:
        title = book.h3.a['title']
        price = book.find('p', class_='price_color').text.strip().replace('£', '')  # Remove £ symbol
        instock_availability = book.find('p', class_='instock availability').text.strip()

        # Extract and convert star rating
        rating_class = book.p['class'][1]  # Example: "star-rating Three"
        rating = rating_conversion.get(rating_class, 0)  # Default to 0 if missing

        product_link = "https://books.toscrape.com/catalogue/" + book.h3.a['href']

        # Append book details to list
        data.append([title, price, instock_availability, rating, product_link])

        # Stop when we reach the required number of books
        if len(data) >= min_data_count:
            break

    page_number += 1  # Move to next page

# Save data to a CSV file
csv_filename = "booksData.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Price", "Availability", "Rating (1-5)", "Product Link"])
    writer.writerows(data)

print(f"✅ Data successfully scraped and saved to {csv_filename}")

