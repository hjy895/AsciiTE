# AsciiTE Dataset Documentation

## Overview

The AsciiTE (ASCII Art Textual Entailment) dataset is a comprehensive collection of 1,503 ASCII art-phrase pairs designed for textual entailment research. This dataset explores how ASCII art composes with natural language to convey meaning through various compositional strategies.

## Dataset Composition

### Dataset Statistics

- **Total Instances**: 1,503 ASCII-phrase mappings
- **Training Set**: 1,052 instances (70%)
- **Validation Set**: 225 instances (15%)
- **Test Set**: 226 instances (15%)

### Compositional Strategies

The dataset employs five distinct compositional strategies:

1. **Direct Representation (40.5%)**: Clear visual-linguistic mappings
   - Example: `:)` → "happy face"
   - 608 instances

2. **Metaphorical Representation (35.0%)**: Abstract symbolic meanings
   - Example: `(╯°□°）╯︵ ┻━┻` → "extreme frustration"
   - 526 instances

3. **Semantic List (14.6%)**: Multiple ASCII elements creating compound meanings
   - Example: `??? ? ?` → "growing confusion"
   - 219 instances

4. **Reduplication (6.6%)**: Repeated patterns for emphasis
   - Example: `XDXDXD` → "extreme laughter"
   - 99 instances

5. **Single Symbol (3.4%)**: Individual Unicode characters
   - Example: `♥` → "love symbol"
   - 51 instances

### Semantic Attributes

Each mapping is annotated with one of five semantic categories:

- **EMOTION (30.3%)**: 455 instances - Feelings and emotional states
- **ACTION (28.7%)**: 431 instances - Physical actions and gestures
- **OBJECT (23.5%)**: 353 instances - Physical objects and shapes
- **STATE (10.0%)**: 150 instances - Mental and physical conditions
- **QUALITY (7.6%)**: 114 instances - Characteristics and properties

## File Structure

### Main Dataset File: `AsciiTE.csv`

The dataset is stored in CSV format with five columns:

| Column | Description |
|--------|-------------|
| **EN** | The English phrase describing the meaning |
| **ASCII** | The ASCII art sequence |
| **Description** | Natural language description of the ASCII art |
| **Compositional_strategy** | The strategy used (Direct, Metaphorical, Semantic List, Reduplication, Single) |
| **Attribute** | The semantic category (EMOTION, ACTION, OBJECT, STATE, QUALITY) |

### Benchmark Splits

Located in `benchmark_data/ascii-textual-entailment/`:

- `train.csv`: 1,052 training instances
- `val.csv`: 225 validation instances  
- `test.csv`: 226 test instances

## Research Applications

The AsciiTE dataset supports research in:

- **Textual Entailment**: Understanding semantic relationships between symbolic and linguistic representations
- **Compositional Semantics**: How meaning emerges from ASCII character combinations
- **Symbol Grounding**: Mapping visual symbols to natural language concepts
- **Cross-Modal Understanding**: ASCII art as a bridge between text and visual modalities
- **Digital Communication**: Internet communication patterns and conventions

## Dataset Examples

### Direct Representation
- `:)` → "happy face"
- `<3` → "heart shape"
- `o/` → "waving hand"

### Metaphorical Representation
- `(๑•̀ㅂ•́)و✧` → "determined spirit"
- `╰(*°▽°*)╯` → "joyful celebration"
- `♪~ ᕕ(ᐛ)ᕗ` → "happy walking"

### Semantic List
- `<3 <3 <3` → "multiple hearts"
- `??? ? ?` → "growing confusion"
- `>>> <<<` → "back and forth"

### Reduplication
- `XDXDXD` → "extreme laughter"
- `!!!!!` → "extreme emphasis"
- `zzzzzz` → "deep sleep"

### Single Symbol
- `♥` → "love symbol"
- `♪` → "music note"
- `★` → "star symbol"

## Usage Guidelines

### Loading the Dataset

```python
import pandas as pd

# Load main dataset
df = pd.read_csv('AsciiTE.csv')

# Load benchmark splits
train = pd.read_csv('benchmark_data/ascii-textual-entailment/train.csv')
val = pd.read_csv('benchmark_data/ascii-textual-entailment/val.csv')
test = pd.read_csv('benchmark_data/ascii-textual-entailment/test.csv')
```

### Data Format

Each row represents a correct ASCII-phrase mapping. For textual entailment tasks, negative examples (non-entailment pairs) should be generated programmatically during training/evaluation by pairing ASCII art with incorrect phrase descriptions.

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

This dataset is released under CC BY 4.0 License.

## Contact

For questions or issues regarding the dataset, please open an issue on the GitHub repository.

