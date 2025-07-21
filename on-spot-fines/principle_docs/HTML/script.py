import csv
import os
from bs4 import BeautifulSoup
from glob import glob

# Function to process a single HTML file and save as CSV
def process_html_to_csv(html_file):
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Select all relevant elements in the order they appear
    elements = soup.select('font.sectionshorttitle, a[dynamicanimation="fpAnimformatRolloverFP1"], font.subsectioncontent')

    current_c1 = None
    current_c2 = None
    current_c3s = []
    rows = []

    for element in elements:
        if element.name == 'font' and 'sectionshorttitle' in element.get('class', []):
            # Handle the previous section if exists
            if current_c1 is not None and current_c2 is not None:
                for c3 in current_c3s:
                    rows.append([current_c1, current_c2, c3])
            # Start new section
            current_c1 = element.get_text(strip=True)
            current_c2 = None
            current_c3s = []
        elif element.name == 'a' and element.get('dynamicanimation') == 'fpAnimformatRolloverFP1':
            current_c2 = element.get_text(strip=True)
        elif element.name == 'font' and 'subsectioncontent' in element.get('class', []):
            if current_c1 is not None and current_c2 is not None:
                current_c3s.append(element.get_text(strip=True))

    # Add the last section
    if current_c1 is not None and current_c2 is not None and current_c3s:
        for c3 in current_c3s:
            rows.append([current_c1, current_c2, c3])

    # Write to CSV with the same name as the HTML file
    csv_file = os.path.splitext(html_file)[0] + '.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['c1', 'c2', 'c3'])
        writer.writerows(rows)

    print(f"CSV file created successfully: {csv_file}")

# Get all HTML files in the current directory
html_files = glob('*.html')

# Process each HTML file
for html_file in html_files:
    process_html_to_csv(html_file)

print("All HTML files processed.")