#!/usr/bin/env python3
"""
Create AsciiTE.csv dataset file following ELCo format EXACTLY
Only include correct ASCII-phrase mappings (no labels, no negative examples)
"""
import json
import csv

# Read the dataset
with open('data/asciite_elco_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter only positive examples (correct mappings)
correct_mappings = [row for row in data if row['label'] == 1]

# Create CSV file (following ELCo.csv format)
with open('AsciiTE.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header (matching ELCo.csv columns)
    # ELCo has: EN, EM, Description, Compositional strategy, Attribute
    # AsciiTE: EN, ASCII, Description, Compositional_strategy, Attribute
    writer.writerow(['EN', 'ASCII', 'Description', 'Compositional_strategy', 'Attribute'])
    
    # Write only correct mappings
    for row in correct_mappings:
        # Create description of the ASCII art (like ELCo's description)
        description = f"['{row['ascii']}']"
        
        writer.writerow([
            row['phrase'],
            row['ascii'],
            description,
            row['strategy'],
            row['attribute']
        ])

print(f"Created AsciiTE.csv with {len(correct_mappings)} correct ASCII-phrase mappings")
print(f"Columns: EN, ASCII, Description, Compositional_strategy, Attribute")
print(f"\nNote: Like ELCo, negative examples are generated during experiments,")
print(f"not stored in the main dataset CSV.")

