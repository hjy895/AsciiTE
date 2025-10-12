#!/usr/bin/env python3
"""
Create final clean AsciiTE.csv with ONLY correct mappings
Clean descriptions following ELCo format exactly
"""
import json
import csv

# Read the original dataset
with open('data/asciite_elco_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter ONLY positive examples (correct mappings)
correct_mappings = [row for row in data if row['label'] == 1]

print(f"Total entries in original: {len(data)}")
print(f"Correct mappings (label=1): {len(correct_mappings)}")

# Create clean CSV file
with open('AsciiTE.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Header matching ELCo format
    writer.writerow(['EN', 'ASCII', 'Description', 'Compositional_strategy', 'Attribute'])
    
    # Write only correct mappings with clean descriptions
    for row in correct_mappings:
        # Clean description: just the ASCII in brackets like ELCo
        description = f"['{row['ascii']}']"
        
        writer.writerow([
            row['phrase'],
            row['ascii'],
            description,
            row['strategy'],
            row['attribute']
        ])

print(f"\nCreated clean AsciiTE.csv with {len(correct_mappings)} correct mappings")
print("All descriptions are clean - no 'does NOT represent' entries")
print("Format matches ELCo exactly")

