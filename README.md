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

- **Total instances**: 1,503 ASCII-phrase mappings
- **Train set**: 1,052 instances (70%)
- **Validation set**: 225 instances (15%)
- **Test set**: 226 instances (15%)

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


## Benchmark Data Structure

```
benchmark_data/
└── ascii-textual-entailment/
    ├── train.csv     # Training set (1,052 instances)
    ├── val.csv       # Validation set (225 instances)
    └── test.csv      # Test set (226 instances)
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

## Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{AsciiTEDataset2026,
    title = "The AsciiTE Dataset: ASCII Art and Lexical Composition for Textual Entailment",
    author = {Author Names},
    booktitle = "Proceedings of Conference",
    year = "2026"
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
