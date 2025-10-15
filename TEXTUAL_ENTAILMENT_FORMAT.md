# AsciiTE Textual Entailment Format

## Overview

The AsciiTE benchmark data has been transformed to follow the **textual entailment format** used by ELCo (Emoji-based Lexical Composition). This format is designed to evaluate how well models understand the compositional semantics of ASCII art.

## Task: ASCII-based Textual Entailment (AsciiTE)

Following the EmoTE (Emoji-based Textual Entailment) paradigm, we frame ASCII art comprehension as a binary entailment task.

### Format

Each example consists of:
- **sent1** (Premise): "This is [ASCII art]."
- **sent2** (Hypothesis): "This is [English phrase]."
- **label**: 1 (entailment) or 0 (non-entailment)
- **strategy**: Compositional strategy code (0-4) or 6 for non-entailment

### Entailment Definition

An English phrase EN is **entailed** by an ASCII sequence ASCII if the sequence captures the phrase's meaning.

**Example - Entailment (label=1):**
- Premise: "This is (๑•̀ㅂ•́)و✧."
- Hypothesis: "This is determined spirit."
- The ASCII art `(๑•̀ㅂ•́)و✧` metaphorically represents "determined spirit" → **Entailment**

**Example - Non-Entailment (label=0):**
- Premise: "This is ***."
- Hypothesis: "This is determined spirit."
- The ASCII art `***` (star pattern) does not represent "determined spirit" → **Non-Entailment**

## Dataset Statistics

### Original Dataset
- **1,503 ASCII-phrase mappings** (all correct/positive examples)

### Textual Entailment Benchmark
- **3,006 total entailment pairs**
  - 1,503 positive examples (entailment, label=1)
  - 1,503 negative examples (non-entailment, label=0)

### Data Splits
| Split | Total | Positive | Negative | Percentage |
|-------|-------|----------|----------|------------|
| Train | 2,104 | 1,055 | 1,049 | 70% |
| Val | 450 | 226 | 224 | 15% |
| Test | 452 | 222 | 230 | 15% |

## Strategy Encoding

Following ELCo's approach, compositional strategies are encoded as integers:

| Strategy | Code | Description | Example |
|----------|------|-------------|---------|
| Direct | 0 | Clear visual-linguistic mapping | `:)` → "happy face" |
| Metaphorical | 1 | Abstract symbolic meaning | `(╯°□°）╯︵ ┻━┻` → "extreme frustration" |
| Semantic List | 2 | Multiple elements | `??? ? ?` → "growing confusion" |
| Reduplication | 3 | Repeated patterns | `zzzzzz` → "deep sleep" |
| Single | 4 | Single Unicode character | `♥` → "love symbol" |
| Non-entailment | 6 | Incorrect pairing | Any mismatched pair |

## Files

```
benchmark_data/ascii-textual-entailment/
├── train.csv    # 2,104 examples
├── val.csv      # 450 examples
└── test.csv     # 452 examples
```

### CSV Format

```csv
sent1,sent2,label,strategy
This is (๑•̀ㅂ•́)و✧.,This is determined spirit.,1,1
This is ***.,This is determined spirit.,0,6
This is ಥ_ಥ.,This is tears of joy.,1,1
This is :D.,This is sad face.,0,6
```

## Negative Example Generation

For each positive example (correct ASCII-phrase pair), one negative example was created by:
1. Taking the same ASCII art
2. Pairing it with a **randomly selected incorrect** English phrase
3. Labeling it as non-entailment (label=0, strategy=6)

This ensures:
- **Balanced dataset**: Equal positive and negative examples
- **Realistic evaluation**: Models must distinguish correct from incorrect compositions
- **Comparable to ELCo**: Same binary classification task structure

## Comparison to ELCo

### Similarities
✓ Same task format (premise-hypothesis pairs)
✓ Binary classification (entailment vs non-entailment)
✓ Prefix format: "This is X."
✓ Strategy encoding (numerical)
✓ Balanced positive/negative examples
✓ 70/15/15 train/val/test split

### Differences
- **ELCo**: Emoji sequences with [EM] separator (e.g., "sparkler [EM] crystal_ball")
- **AsciiTE**: Direct ASCII art (e.g., "(๑•̀ㅂ•́)و✧")
- **Focus**: AsciiTE emphasizes ASCII art composition vs ELCo's emoji composition

## Usage

### Loading Data

```python
import pandas as pd

# Load train set
train_df = pd.read_csv('benchmark_data/ascii-textual-entailment/train.csv')

# Filter positive examples
positive = train_df[train_df['label'] == 1]

# Filter negative examples
negative = train_df[train_df['label'] == 0]
```

### Task Evaluation

Models are evaluated on their ability to:
1. **Identify entailment**: Recognize when ASCII art correctly represents a phrase
2. **Identify non-entailment**: Detect when ASCII art does NOT represent a phrase
3. **Understand composition**: Leverage compositional strategies for better comprehension

### Metrics
- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Per-class performance
- **Strategy-wise accuracy**: Performance by compositional strategy type

## Research Motivation

This format enables research in:
- **Compositional understanding**: How models interpret ASCII art composition
- **Symbol grounding**: Mapping visual symbols to linguistic concepts
- **Cross-modal reasoning**: Understanding relationships between ASCII and natural language
- **Lexical semantics**: How ASCII art combines to create meaning

## Citation

If you use this dataset format, please cite:

```bibtex
@inproceedings{AsciiTEDataset2024,
    title = "The AsciiTE Dataset: ASCII Art and Lexical Composition for Textual Entailment",
    author = {Author Names},
    booktitle = "Proceedings of Conference",
    year = "2024"
}
```

---

**Generated**: October 2025  
**Format Version**: 1.0 (ELCo-compatible)  
**Total Pairs**: 3,006 (1,503 positive + 1,503 negative)

