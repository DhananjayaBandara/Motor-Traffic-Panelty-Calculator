import csv
import re  # Add import for regex

input_file = "test.txt"
output_file = "output.csv"

rows = []
with open(input_file, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
while i < len(lines):
    # Title is the line with number and name
    title = lines[i]
    # Remove leading number and dot (e.g., "1. ")
    title = re.sub(r'^\d+\.\s*', '', title)
    # Description is the next line
    if i + 1 < len(lines):
        description = lines[i + 1]
    else:
        description = ""
    rows.append([title, description])
    i += 2

with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Title", "Description"])
    writer.writerows(rows)
