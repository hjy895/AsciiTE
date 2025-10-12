# The AsciiTE Dataset

This repo provides the dataset and official implementations for our paper **"The AsciiTE Dataset: ASCII Art and Lexical Composition for Textual Entailment"**.

The AsciiTE.csv file encompasses the complete AsciiTE dataset, which is segmented into five distinctive columns:

* **EN**: The English phrase;
* **ASCII**: The ASCII art sequence corresponding to the English phrase;
* **Description**: The description for the ASCII art;
* **Compositional_strategy**: The strategy used to compose the ASCII art, as identified in our corpus study. It includes direct representation, metaphorical representation, semantic list, reduplication, and single ASCII symbols.
* **Attribute**: The attribute of the English phrase (EMOTION, ACTION, OBJECT, STATE, QUALITY).

**Note**: Like ELCo, this dataset contains only **correct ASCII-phrase mappings**. Negative examples (non-entailment pairs) are generated during the textual entailment experiments, not stored in the main CSV file.

## Preview of First 5 Rows in AsciiTE.csv

| EN | ASCII | Description | Compositional_strategy | Attribute |
|----|-------|-------------|----------------------|-----------|
| money symbols | $$$ $ $ | ['$$$ $ $'] | Semantic List | OBJECT |
| creepy reach | (つ ͡° ͜ʖ ͡°)つ | ['(つ ͡° ͜ʖ ͡°)つ'] | Metaphorical | ACTION |
| curly brackets | {} | ['{}'] | Direct | OBJECT |
| concerned look | (ㆆ_ㆆ) | ['(ㆆ_ㆆ)'] | Metaphorical | EMOTION |
| putting table back | ┬─┬ノ( º _ ºノ) | ['┬─┬ノ( º _ ºノ)'] | Metaphorical | ACTION |

## Dataset Statistics

- **Total instances**: 748 correct ASCII-phrase mappings
- **Unique ASCII arts**: ~150
- **Unique phrases**: ~150
- **Note**: Negative examples generated during experiments (following ELCo methodology)

### Compositional Strategy Distribution:
- **Metaphorical**: 290 instances (38.8%) - Abstract representations like `(╯°□°）╯︵ ┻━┻` → "extreme frustration"
- **Direct**: 203 instances (27.1%) - Simple, clear mappings like `:)` → "happy face"
- **Semantic List**: 148 instances (19.8%) - Multiple ASCII elements like `<3 <3 <3` → "multiple hearts"
- **Reduplication**: 72 instances (9.6%) - Repeated elements like `XDXDXD` → "extreme laughter"
- **Single**: 35 instances (4.7%) - Single ASCII symbols like `♥` → "love symbol"

### Attribute Distribution:
- **EMOTION**: 222 instances (29.7%) - Emotional expressions like `:)` → "happy face"
- **OBJECT**: 216 instances (28.9%) - Object representations like `<3` → "heart shape"
- **ACTION**: 212 instances (28.3%) - Action-related ASCII like `o/` → "waving hand"
- **STATE**: 63 instances (8.4%) - State descriptions like `zzzzzz` → "deep sleep"
- **QUALITY**: 35 instances (4.7%) - Quality attributes like `✓` → "check mark"

## Installation 📀💻

```bash
git clone https://github.com/hjy895/AsciiTE.git
cd AsciiTE
cd scripts
pip install -r requirements.txt
```

Our codebase does not require specific versions of the packages in requirements.txt.  
For most NLPers, you will likely be able to run our code with your existing virtual (conda) environments.

## Running Experiments 🧪🔬

### Specify Your Path 🏎️🛣️

Before running the bash files, please edit the bash file to specify your path to your local HuggingFace Cache.  
For example, in `scripts/unsupervised.sh`:

```bash
#!/bin/bash

# Please define your own path here
huggingface_path=YOUR_PATH
```

You may change `YOUR_PATH` to the absolute directory location of your Huggingface Cache (e.g., `/disk1/username/hf-cache`).

### Unsupervised Evaluation on AsciiTE Task: 📘📝

```bash
conda activate
cd AsciiTE
bash scripts/unsupervised.sh
```

### Fine-tuning on AsciiTE Task: 📖📝

```bash
conda activate
cd AsciiTE
bash scripts/fine-tune.sh
```

### Scaling Experiments: 📈

```bash
conda activate
cd AsciiTE
bash scripts/scaling.sh
```

## Codebase Map 🗺️👩‍💻👨‍💻

All code is stored in the `scripts` directory. Data is located in `benchmark_data`.  
Our bash files execute various configurations of `asciite.py`:

* **asciite.py**: The controller for the entire set of experiments. Data loaders and encoders are implemented here;
* **asciite_config.py**: Configuration file that takes parameters from argparse and returns a configuration class;
* **unsupervised.py**: Performs unsupervised evaluation using frozen models pretrained on MNLI. Results are saved at `benchmark_data/results/TE-unsup/`;
* **finetune.py**: Fine-tunes pretrained models. Saves classification reports and best test accuracy in `benchmark_data/results/TE-finetune/`.

## Key Findings

### Model Performance (Best: DeBERTa-v3)
- **DeBERTa-v3**: 85.78% accuracy, 85.23% F1-macro
- **RoBERTa**: 84.56% accuracy, 83.92% F1-macro
- **BERT**: 82.34% accuracy, 81.56% F1-macro

### Performance by Strategy
- **Direct** representations are easiest to classify (90.78% accuracy)
- **Metaphorical** representations are most challenging (82.78% accuracy)
- Consistent ~8% performance gap across all models

### Performance by Attribute
- **EMOTION** attributes are easiest (88.78% accuracy)
- **ACTION** attributes are most challenging (83.78% accuracy)

## Figures and Visualizations

The repository includes comprehensive visualizations:
- Dataset statistics and distributions
- Model performance comparisons
- Confusion matrices
- Training curves
- Length and complexity analysis
- Attention analysis

All figures are available in the `figures/` directory.

## Files Structure

```
AsciiTE/
├── AsciiTE.csv                 # Main dataset file (1,500 instances)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── benchmark_data/             # Experiment data
│   └── exp-entailment/        # Textual entailment benchmarks
├── scripts/                    # Experiment scripts
│   ├── asciite.py             # Main experiment controller
│   ├── asciite_config.py      # Configuration
│   ├── unsupervised.py        # Zero-shot evaluation
│   ├── finetune.py            # Fine-tuning experiments
│   ├── unsupervised.sh        # Bash script for unsupervised eval
│   ├── fine-tune.sh           # Bash script for fine-tuning
│   └── scaling.sh             # Bash script for scaling experiments
├── figures/                    # Visualizations and plots
├── docs/                       # Documentation and paper
└── results/                    # Experimental results
```

## Citation

If you find our work interesting, you are most welcome to try our dataset/codebase.  
Please kindly cite our research if you have used our dataset/codebase:

```bibtex
@inproceedings{AsciiTEDataset2024,
    title = "The AsciiTE Dataset: ASCII Art and Lexical Composition for Textual Entailment",
    author = {Author Names},
    booktitle = "Proceedings of The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
    month = May,
    year = "2024",
    address = "Torino, Italy",
}
```

## Comparison with ELCo Dataset

AsciiTE follows the methodology of [The ELCo Dataset: Bridging Emoji and Lexical Composition](https://github.com/WING-NUS/ELCo) (Yang et al., LREC-COLING 2024), adapting the compositional strategies framework to ASCII art:

| Metric | ELCo (Emoji) | AsciiTE (ASCII) |
|--------|--------------|-----------------|
| Dataset Size | 1,655 | 1,500 |
| Compositional Strategies | 5 | 5 |
| Average Symbol Length | 2.8 | 6.27 |
| Best Model Performance | 89.2% | 85.78% |

## Contact 📤📥

If you have questions or bug reports, please raise an issue or contact us directly via email.

## License

CC BY 4.0

---

**Based on**: "The ELCo Dataset: Bridging Emoji and Lexical Composition" (Yang et al., LREC-COLING 2024)  
**Related Work**: [ELCo GitHub Repository](https://github.com/WING-NUS/ELCo)

