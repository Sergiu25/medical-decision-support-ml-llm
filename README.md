# Medical Decision Support ML/LLM

Bachelor thesis project focused on medical text classification in a decision support context.
The project compares classical machine learning models, a neural network based on TF-IDF features, and fine-tuned BERT-based models for multiclass diagnosis prediction.

## Project overview

The purpose of this repository is to provide the source code used for the experimental part of the bachelor thesis. The experiments evaluate several approaches for classifying medical cases into diagnostic categories:

* classical machine learning models using TF-IDF features;
* metadata-only baseline using age and gender;
* a feed-forward neural network trained on TF-IDF representations;
* BERT-based models fine-tuned for medical text classification;
* comparison of all evaluated approaches.

The repository contains the code required to reproduce the main experimental workflow.
The dataset and trained model checkpoints are not included in this repository.

## Repository structure

```text
medical-decision-support-ml-llm/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── scripts/
│   ├── 00_create_fixed_splits.py
│   ├── 01_text_tfidf_baseline.py
│   ├── 02_metadata_age_gender_fixed_splits.py
│   ├── 03_compare_experiments.py
│   ├── 04_tfidf_mlp_neural_network.py
│   └── 05_compare_experiments_with_mlp.py
│
└── notebooks/
    ├── bert_512_baseline.ipynb
    ├── bert_head_tail.ipynb
    └── bert_relevant_chunks.ipynb
```

## Dataset

The dataset used in the experiments is not included in this repository.

To reproduce the experiments, the dataset must be obtained from its original source and placed locally in the project directory. The scripts expect a cleaned dataset file with the following name:

```text
multicare_clean_for_ml_noleak.csv
```

Recommended local structure:

```text
medical-decision-support-ml-llm/
│
├── multicare_clean_for_ml_noleak.csv
├── scripts/
└── notebooks/
```

If the dataset is placed in another location, the input path must be updated inside the scripts.

The dataset is excluded from the repository in order to avoid redistributing data that may be subject to its own license, terms of use or access restrictions.

## Experimental workflow

The experiments were organized in several steps.

### 1. Create fixed train/validation/test splits

```bash
python scripts/00_create_fixed_splits.py
```

This script creates fixed stratified splits used by the following experiments.
The same splits are used across the classical ML, metadata, MLP and BERT experiments in order to make the results comparable.

Expected output: generated split files inside a local output folder.

### 2. Run classical TF-IDF machine learning baselines

```bash
python scripts/01_text_tfidf_baseline.py
```

This script trains and evaluates classical machine learning models using TF-IDF representations of the medical text.

The evaluated models include classical text classification approaches such as linear models and support vector machines.

### 3. Run metadata-only baseline

```bash
python scripts/02_metadata_age_gender_fixed_splits.py
```

This script evaluates models trained only on structured metadata, such as age and gender.
The purpose of this experiment is to verify whether metadata alone contains enough predictive information for diagnosis classification.

### 4. Compare classical experiments

```bash
python scripts/03_compare_experiments.py
```

This script compares the results obtained from the text-based classical models and the metadata-only baseline.

### 5. Run TF-IDF + MLP experiment

```bash
python scripts/04_tfidf_mlp_neural_network.py
```

This script trains multiple feed-forward neural network configurations using TF-IDF vectors as input.

The evaluated MLP configurations include:

* one hidden layer with 256 neurons and dropout;
* two hidden-layer architectures;
* a no-hidden-layer linear neural baseline.

The best model is selected based on validation performance and then evaluated on the test set.

### 6. Compare experiments including MLP

```bash
python scripts/05_compare_experiments_with_mlp.py
```

This script adds the MLP results to the previous comparison and generates summary tables/figures for the classical and neural-network-based experiments.

## BERT experiments

The BERT experiments are provided as Jupyter notebooks:

```text
notebooks/bert_512_baseline.ipynb
notebooks/bert_head_tail.ipynb
notebooks/bert_relevant_chunks.ipynb
```

These notebooks were designed to be run in Google Colab or another environment with GPU support.

The evaluated BERT strategies are:

1. **BERT 512 baseline**
   The input text is truncated to the first 512 tokens.

2. **BERT Head+Tail**
   The input representation combines the beginning and the end of the medical text, preserving both the initial symptoms and the final clinical information.

3. **BERT Relevant Chunks**
   The text is divided into chunks and the most relevant chunks are selected for classification.

Recommended hardware for running the BERT notebooks:

* GPU runtime;
* NVIDIA T4 or equivalent;
* sufficient disk space for model checkpoints and generated outputs.

The BERT notebooks fine-tune `bert-base-uncased` for multiclass classification.
The internal BERT layers are not frozen; both the base BERT model and the final classification layer are trained.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sergiu25/medical-decision-support-ml-llm.git
cd medical-decision-support-ml-llm
```


### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If PyTorch with GPU support is needed, install the correct PyTorch version for the local CUDA environment by following the official PyTorch installation instructions.

For Google Colab, most dependencies are already available, but additional packages can be installed inside the notebooks when needed.

## Main dependencies

The project uses the following main Python libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* scipy
* joblib
* torch
* transformers
* datasets
* evaluate
* accelerate
* tqdm

## Reproducibility notes

The experiments use a fixed random seed where applicable, generally:

```python
RANDOM_STATE = 42
```

The dataset is split using stratified train/validation/test splits.
This helps preserve the class distribution across all subsets and allows fair comparison between models.

The TF-IDF vectorizer is fitted only on the training set.
The validation and test sets are transformed using the same fitted vectorizer, preventing information leakage from evaluation data into the training process.

Generated outputs, model checkpoints and large result folders are not included in the repository.
They can be regenerated by running the scripts and notebooks.

## Outputs

The scripts generate local output folders containing files such as:

* validation metrics;
* test metrics;
* classification reports;
* confusion matrices;
* training history files;
* comparison tables;
* generated figures.

These files are intentionally not all included in the repository because they can be regenerated and may significantly increase repository size.

## License

The code in this repository is released under the MIT License.

The dataset and pretrained models used in the experiments are not included in this repository and remain subject to their original licenses and terms of use.

## Author

Sergiu-Cătălin Petriș
Bachelor thesis project
Faculty of Economics and Business Administration
Business Informatics
