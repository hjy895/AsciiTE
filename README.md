# The AsciiTE Dataset

This repository provides the dataset and official implementation for our paper **"The AsciiTE Dataset: ASCII Art and Lexical Composition for Textual Entailment"**.

The AsciiTE.csv file contains the complete AsciiTE dataset with five columns:

* **EN**: The English phrase
* **ASCII**: The ASCII art sequence corresponding to the phrase
* **Description**: Natural language description of the ASCII art
* **Compositional_strategy**: The compositional strategy used (Direct, Metaphorical, Semantic List, Reduplication, Single)
* **Attribute**: The semantic attribute (EMOTION, ACTION, OBJECT, STATE, QUALITY)

## Preview of AsciiTE Dataset

| EN | ASCII | Description | Compositional_strategy | Attribute |
|----|-------|-------------|----------------------|-----------|
| determined spirit | (๑•̀ㅂ•́)و✧ | ASCII art '(๑•̀ㅂ•́)و✧' representing determined spirit | Metaphorical | QUALITY |
| joyful celebration | ╰(*°▽°*)╯ | ASCII art '╰(*°▽°*)╯' representing joyful celebration | Metaphorical | EMOTION |
| happy walking | ♪~ ᕕ(ᐛ)ᕗ | ASCII art '♪~ ᕕ(ᐛ)ᕗ' representing happy walking | Metaphorical | ACTION |
| gentle smile | (✿◠‿◠) | ASCII art '(✿◠‿◠)' representing gentle smile | Metaphorical | EMOTION |
| cool dancing | ヾ(⌐■_■)ノ♪ | ASCII art 'ヾ(⌐■_■)ノ♪' representing cool dancing | Metaphorical | ACTION |

## Dataset Statistics

### Core Dataset
- **Total instances**: 1,503 ASCII-phrase mappings
- All mappings are correct (entailment) examples

### Textual Entailment Benchmark
- **Total instances**: 3,006 entailment pairs (1,503 positive + 1,503 negative)
- **Train set**: 2,104 instances (70%) - 1,055 positive, 1,049 negative
- **Validation set**: 450 instances (15%) - 226 positive, 224 negative  
- **Test set**: 452 instances (15%) - 222 positive, 230 negative

### Compositional Strategy Distribution:
- **Direct**: 608 instances (40.5%) - Clear visual representations like `:)` for "happy face"
- **Metaphorical**: 526 instances (35.0%) - Abstract meanings like `(╯°□°）╯︵ ┻━┻` for "extreme frustration"
- **Semantic List**: 219 instances (14.6%) - Multiple elements like `??? ? ?` for "growing confusion"
- **Reduplication**: 99 instances (6.6%) - Repeated patterns like `XDXDXD` for "extreme laughter"
- **Single**: 51 instances (3.4%) - Single symbols like `♥` for "love symbol"

### Attribute Distribution:
- **EMOTION**: 455 instances (30.3%) - Emotional states and expressions
- **ACTION**: 431 instances (28.7%) - Actions and gestures
- **OBJECT**: 353 instances (23.5%) - Physical objects and shapes
- **STATE**: 150 instances (10.0%) - Mental and physical states
- **QUALITY**: 114 instances (7.6%) - Qualities and characteristics

## Installation

```bash
git clone https://github.com/hjy895/AsciiTE.git
cd AsciiTE
cd scripts
pip install -r requirements.txt
```

## Running Experiments

### Configure Your Environment

Before running experiments, specify your HuggingFace cache path in the bash scripts.

Edit `scripts/unsupervised.sh`:
```bash
#!/bin/bash
# Set your HuggingFace cache directory
huggingface_path=YOUR_PATH
```

### Zero-Shot Evaluation

Evaluate pre-trained models without fine-tuning:

```bash
cd AsciiTE
bash scripts/unsupervised.sh
```

### Fine-Tuning

Train models on the AsciiTE dataset:

```bash
cd AsciiTE
bash scripts/fine-tune.sh
```

### Scaling Experiments

Test performance across different dataset sizes:

```bash
cd AsciiTE
bash scripts/scaling.sh
```

## Benchmark Data Structure

### ASCII-based Textual Entailment (AsciiTE Task)

We adopt an ASCII-based Textual Entailment task to examine the capacity of models to comprehend the composition of ASCII art through binary classification.

**Task Format:**
- **Premise (sent1)**: "This is [ASCII art]."
- **Hypothesis (sent2)**: "This is [English phrase]."
- **Label**: 1 (entailment) or 0 (non-entailment)
- **Strategy**: Compositional strategy (0-4) or 6 for non-entailment

An English phrase is entailed by an ASCII sequence if the sequence captures the phrase's meaning. For instance, `(๑•̀ㅂ•́)و✧` entails "determined spirit" (metaphorical composition), while `***` does not entail "determined spirit" (non-entailment).

**Dataset Structure:**
```
benchmark_data/
└── ascii-textual-entailment/
    ├── train.csv     # Training set (2,104 instances: 1,055 positive, 1,049 negative)
    ├── val.csv       # Validation set (450 instances: 226 positive, 224 negative)
    └── test.csv      # Test set (452 instances: 222 positive, 230 negative)
```

**Total**: 3,006 textual entailment pairs (1,503 positive, 1,503 negative)

**Strategy Mapping:**
- 0: Direct representation
- 1: Metaphorical representation
- 2: Semantic list
- 3: Reduplication
- 4: Single symbol
- 6: Non-entailment (negative examples)

**Example:**
```csv
sent1,sent2,label,strategy
This is (๑•̀ㅂ•́)و✧.,This is determined spirit.,1,1
This is ***.,This is determined spirit.,0,6
```

## Code Structure

```
AsciiTE/
├── AsciiTE.csv                 # Complete dataset (1,503 instances)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── benchmark_data/             # Train/val/test splits
│   └── ascii-textual-entailment/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── scripts/                    # Experiment scripts
    ├── asciite.py             # Main experiment controller
    ├── unsupervised.sh        # Zero-shot evaluation
    ├── fine-tune.sh           # Fine-tuning script
    ├── scaling.sh             # Scaling experiments
    └── requirements.txt       # Dependencies
```

## Key Features

### Compositional Strategies

The dataset captures five distinct ways ASCII art composes meaning:

1. **Direct Representation**: Simple visual mappings (`:)` = happy face)
2. **Metaphorical Representation**: Abstract symbolic meanings (`(╯°□°）╯︵ ┻━┻` = extreme frustration)
3. **Semantic List**: Multiple ASCII elements creating compound meanings
4. **Reduplication**: Repeated characters for emphasis (`!!!!!!` = extreme emphasis)
5. **Single Symbol**: Individual Unicode characters with specific meanings

### Semantic Attributes

Each mapping is annotated with one of five semantic categories:
- **EMOTION**: Feelings and emotional states
- **ACTION**: Physical actions and gestures  
- **OBJECT**: Concrete and abstract objects
- **STATE**: Mental and physical conditions
- **QUALITY**: Characteristics and properties

## Research Applications

The AsciiTE dataset enables research in:

- **Textual Entailment**: Understanding semantic relationships
- **Compositional Semantics**: How meaning emerges from ASCII combinations
- **Symbol Grounding**: Mapping visual symbols to natural language
- **Cross-Modal Understanding**: ASCII art as a bridge between text and visuals
- **Cultural Communication**: Internet communication patterns

## Model Performance

Results from our experiments:

| Model | Zero-Shot Acc | Fine-Tuned Acc |
|-------|--------------|----------------|
| BERT-base | 55.0% | 80.4% |
| RoBERTa-base | 55.8% | 84.0% |
| DeBERTa-v3 | 62.9% | 85.8% |

## Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{AsciiTEDataset2024,
    title = "The AsciiTE Dataset: ASCII Art and Lexical Composition for Textual Entailment",
    author = {Author Names},
    booktitle = "Proceedings of Conference",
    year = "2024"
}
```

## License

CC BY 4.0

## Contact

For questions or issues, please open an issue on GitHub.

---

**Repository**: https://github.com/hjy895/AsciiTE  
**Dataset Size**: 1,503 ASCII-phrase mappings  
**Task**: Textual Entailment with ASCII Art
