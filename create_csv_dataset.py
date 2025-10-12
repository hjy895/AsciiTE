#!/usr/bin/env python3
"""
Create AsciiTE.csv dataset file for GitHub repository
Similar to ELCo.csv format
"""
import json
import csv

# Read the dataset
with open('data/asciite_elco_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create CSV file
with open('AsciiTE.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header (similar to ELCo.csv format)
    writer.writerow(['EN', 'ASCII', 'Description', 'Compositional_strategy', 'Attribute', 'Label'])
    
    # Write data rows
    for row in data:
        description = f"ASCII '{row['ascii']}' {'represents' if row['label']==1 else 'does NOT represent'} '{row['phrase']}'"
        writer.writerow([
            row['phrase'],
            row['ascii'],
            description,
            row['strategy'],
            row['attribute'],
            row['label']
        ])

print(f"Created AsciiTE.csv with {len(data)} entries")
print(f"Columns: EN, ASCII, Description, Compositional_strategy, Attribute, Label")

