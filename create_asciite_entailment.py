"""
Create AsciiTE Textual Entailment dataset in ELCo format
"""
import pandas as pd
import random
import numpy as np

# Read the AsciiTE dataset
df = pd.read_csv('AsciiTE.csv')

# Map compositional strategies to numbers (like ELCo)
strategy_map = {
    'Direct': 0,
    'Metaphorical': 1,
    'Semantic List': 2,
    'Reduplication': 3,
    'Single': 4
}

# Convert strategy to numeric
df['strategy_num'] = df['Compositional_strategy'].map(strategy_map)

# Create positive examples (entailment = 1)
positive_examples = []
for _, row in df.iterrows():
    sent1 = f"This is {row['ASCII']}."
    sent2 = f"This is {row['EN']}."
    positive_examples.append({
        'sent1': sent1,
        'sent2': sent2,
        'label': 1,
        'strategy': row['strategy_num']
    })

print(f"Created {len(positive_examples)} positive examples")

# Create negative examples (non-entailment = 0)
# For each ASCII, pair it with a WRONG English phrase
negative_examples = []
all_phrases = df['EN'].tolist()

for _, row in df.iterrows():
    # Get all phrases EXCEPT the correct one
    incorrect_phrases = [p for p in all_phrases if p != row['EN']]
    
    # Randomly select a wrong phrase
    wrong_phrase = random.choice(incorrect_phrases)
    
    sent1 = f"This is {row['ASCII']}."
    sent2 = f"This is {wrong_phrase}."
    negative_examples.append({
        'sent1': sent1,
        'sent2': sent2,
        'label': 0,
        'strategy': 6  # ELCo uses 6 for non-entailment
    })

print(f"Created {len(negative_examples)} negative examples")

# Combine positive and negative examples
all_examples = positive_examples + negative_examples
print(f"Total examples: {len(all_examples)}")

# Convert to dataframe
entailment_df = pd.DataFrame(all_examples)

# Shuffle the dataset
entailment_df = entailment_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train/val/test (70/15/15 split)
n = len(entailment_df)
train_size = int(0.70 * n)
val_size = int(0.15 * n)

train_df = entailment_df[:train_size]
val_df = entailment_df[train_size:train_size+val_size]
test_df = entailment_df[train_size+val_size:]

print(f"\nDataset splits:")
print(f"Train: {len(train_df)} ({len(train_df)/len(entailment_df)*100:.1f}%)")
print(f"Val: {len(val_df)} ({len(val_df)/len(entailment_df)*100:.1f}%)")
print(f"Test: {len(test_df)} ({len(test_df)/len(entailment_df)*100:.1f}%)")

# Count positive/negative in each split
print(f"\nTrain - Positive: {(train_df['label']==1).sum()}, Negative: {(train_df['label']==0).sum()}")
print(f"Val - Positive: {(val_df['label']==1).sum()}, Negative: {(val_df['label']==0).sum()}")
print(f"Test - Positive: {(test_df['label']==1).sum()}, Negative: {(test_df['label']==0).sum()}")

# Save to CSV files
train_df.to_csv('benchmark_data/ascii-textual-entailment/train.csv', index=False)
val_df.to_csv('benchmark_data/ascii-textual-entailment/val.csv', index=False)
test_df.to_csv('benchmark_data/ascii-textual-entailment/test.csv', index=False)

print("\nFiles saved successfully!")
print("- benchmark_data/ascii-textual-entailment/train.csv")
print("- benchmark_data/ascii-textual-entailment/val.csv")
print("- benchmark_data/ascii-textual-entailment/test.csv")

# Show some examples
print("\n=== Sample Examples ===")
print("\nPositive (Entailment) examples:")
print(train_df[train_df['label']==1].head(3).to_string(index=False))
print("\nNegative (Non-entailment) examples:")
print(train_df[train_df['label']==0].head(3).to_string(index=False))

