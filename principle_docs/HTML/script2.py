import csv
import re
import os
from glob import glob

# Function to extract notations and remaining content
def extract_notations(text):
    # Extract numerical notations like (1), (2), etc.
    numerical_notation = re.search(r'^\(\d+\)', text)
    numerical_notation = numerical_notation.group(0)[1:-1] if numerical_notation else ''

    # Extract letter notations like (a), (b), etc.
    letter_notation = re.search(r'^\([a-z]\)', text)
    letter_notation = letter_notation.group(0)[1:-1] if letter_notation else ''

    # Extract Roman numeral notations like (i), (ii), etc.
    roman_notation = re.search(r'^\([ivx]+\)', text, re.IGNORECASE)
    roman_notation = roman_notation.group(0)[1:-1] if roman_notation else ''

    # Remove the extracted notations from the text
    remaining_content = re.sub(r'^\(\d+\)|^\([a-z]\)|^\([ivx]+\)', '', text, flags=re.IGNORECASE).strip()

    return numerical_notation, letter_notation, roman_notation, remaining_content

# Function to process a single CSV file
def process_csv_file(csv_file):
    rows = []
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header
        # Update the header to include only c1, c2, c3, c4, c5, c6
        header = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
        rows.append(header)

        # Variables to store the last seen numerical and letter notations
        last_numerical_notation = ''
        last_letter_notation = ''

        for row in reader:
            # Take only the first three columns (c1, c2, c3)
            c1, c2, c3 = row[:3]  # Slice the row to ensure only three values are unpacked
            numerical_notation, letter_notation, roman_notation, remaining_content = extract_notations(c3)

            # Update c3 and c4 based on the current and previous notations
            if numerical_notation:
                last_numerical_notation = numerical_notation
                last_letter_notation = ''  # Reset letter notation when a new numerical notation is found
            if letter_notation:
                last_letter_notation = letter_notation

            # Fill c3 and c4 with the last seen numerical and letter notations
            c3_cleaned = last_numerical_notation
            c4_cleaned = last_letter_notation

            # Append the cleaned data to the row
            rows.append([c1, c2, c3_cleaned, c4_cleaned, roman_notation, remaining_content])

    # Write the updated data back to the same CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print(f"Updated CSV file: {csv_file}")

# Get all CSV files in the current directory
csv_files = glob('*.csv')

# Process each CSV file
for csv_file in csv_files:
    process_csv_file(csv_file)

print("All CSV files processed and updated.")