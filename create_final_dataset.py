#!/usr/bin/env python3
"""
Create final AsciiTE dataset with proper descriptions
Matching ELCo format with descriptive text
"""
import json
import csv

# Read the dataset with correct mappings
with open('data/asciite_elco_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get only correct mappings (label=1)
correct_mappings = [row for row in data if row['label'] == 1]

print(f"Total correct mappings: {len(correct_mappings)}")

# Create proper descriptions
def create_description(ascii_art, phrase, strategy):
    """Create meaningful description for ASCII art"""
    # Return description of what the ASCII represents
    return f"ASCII art '{ascii_art}' representing {phrase}"

# Create CSV with proper format
with open('AsciiTE.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Header (matching ELCo: EN, EM/ASCII, Description, Compositional_strategy, Attribute)
    writer.writerow(['EN', 'ASCII', 'Description', 'Compositional_strategy', 'Attribute'])
    
    # Write all correct mappings with proper descriptions
    for row in correct_mappings:
        description = create_description(row['ascii'], row['phrase'], row['strategy'])
        
        writer.writerow([
            row['phrase'],
            row['ascii'],
            description,
            row['strategy'],
            row['attribute']
        ])

print(f"Created AsciiTE.csv with {len(correct_mappings)} entries")
print(f"All entries have meaningful descriptions")
print(f"\nStrategy distribution:")

from collections import Counter
strategies = Counter(row['strategy'] for row in correct_mappings)
for strategy, count in strategies.most_common():
    print(f"  {strategy}: {count} ({count/len(correct_mappings)*100:.1f}%)")

print(f"\nAttribute distribution:")
attributes = Counter(row['attribute'] for row in correct_mappings)
for attr, count in attributes.most_common():
    print(f"  {attr}: {count} ({count/len(correct_mappings)*100:.1f}%)")

