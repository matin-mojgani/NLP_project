# ELKP: Knowledge-Powered Rumor Detection 🕵️‍♂️🔍

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-BERT-yellow)
![SpaCy](https://img.shields.io/badge/SpaCy-3.7+-09A3D5?logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Social media rumors spread faster than truth. Traditional AI models often fail to detect them because they only analyze *writing style* rather than *facts*.

This project implements a minimal **ELKP (Entity-Link Knowledge-Powered)** framework that transforms a standard BERT model into a "Fact-Checker" by:

1. **Extracting Entities** from a tweet (e.g., "Eiffel Tower")
2. **Retrieving Knowledge** from Wikipedia
3. **Injecting Context** into the model to detect contradictions

### 🔄 Pipeline Flow

```
Input Tweet → NER (SpaCy) → Wiki Search → Knowledge Injection → BERT Classification → Rumor/Non-Rumor
```

---

## 📂 Project Structure

The project follows **OOP (Object-Oriented Programming)** principles for modularity.

```
rumor-detection-elkp/
│
├── .env                        # Configuration (Project Root Path)
├── main(oop).py                # Master Script: Train, Evaluate & Save Model
├── requirements.txt            # Project Dependencies
├── README.md                   # Project Documentation
│
├── data/
│   ├── raw/                    # Original PHEME dataset (archive/)
│   └── processed/              # Cleaned, Augmented, and Test CSVs
│       ├── dataset.csv         # Extracted tweets from raw data
│       ├── augmented_dataset.csv  # Knowledge-enhanced tweets
│       └── test_dataset.csv    # Held-out test set for demo
│
├── src/                        # Source Modules
│   ├── __init__.py
│   ├── prepare_data(oop).py    # Parses raw JSON dataset to CSV
│   ├── preprocess(oop).py      # The "Brain" - Injects Wikipedia Knowledge
│   └── check_install.py        # Utility for checking dependencies
│
├── models/                     # Saved trained models (created after training)
│   └── rumor_model/            # Fine-tuned BERT model
│
├── results/                    # Training outputs and checkpoints
│
└── demo/
    └── demo_notebook.ipynb     # Interactive presentation notebook
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- (Optional) CUDA-enabled GPU for faster training

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rumor-detection-elkp
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download SpaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment

Create a `.env` file in the project root:

```ini
PROJECT_ROOT=C:\Users\YourName\path\to\rumor-detection-elkp
```

> ⚠️ **Important:** Replace the path with your actual project directory.

---

## 🛠️ Usage Pipeline

Follow these steps **in order** to go from raw data to a working AI model.

### Step 1: Data Parsing 📁

Convert the PHEME dataset's complex directory structure (JSONs) into a clean CSV file.

```bash
python "src/prepare_data(oop).py"
```

**Output:** `data/processed/dataset.csv`

### Step 2: Knowledge Injection 🧠

This is the **core innovation**. The script:
- Scans tweets for named entities using SpaCy NER
- Retrieves Wikipedia summaries for detected entities
- Creates knowledge-augmented text in the format: `Knowledge: [Wiki Info] [SEP] Tweet: [Original Text]`

```bash
python "src/preprocess(oop).py"
```

**Output:** `data/processed/augmented_dataset.csv`

> ⏳ **Note:** This process queries Wikipedia for every tweet and may take several minutes.

### Step 3: Training & Evaluation 🎓

Fine-tune the BERT model on the knowledge-augmented data.

```bash
python "main(oop).py"
```

**This script:**
- Splits data into 80% training / 20% test sets
- Saves the test set separately for the demo (`test_dataset.csv`)
- Downloads and fine-tunes `bert-base-uncased`
- Trains for 3 epochs with reproducible seeds
- Evaluates and reports Accuracy & F1-Score
- Saves the best model to `models/rumor_model/`

### Step 4: Interactive Demo 🎮

Launch the Jupyter Notebook to test the model interactively.

```bash
jupyter notebook demo/demo_notebook.ipynb
```

**Features:**
- Load the trained model
- Test on unseen data from `test_dataset.csv`
- Enter custom rumors to check in real-time

---

## 📊 Model Performance

Results using `bert-base-uncased` with 3 epochs of fine-tuning on the PHEME dataset:

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | ~85.14% | Correctly classifies 85% of claims |
| **F1-Score** | ~0.84 | Balanced precision and recall |

---

## 🧐 How It Works (Example)

**Input Tweet:**
> "The Eiffel Tower has been sold to a private company."

**Step-by-step processing:**

1. **NER:** Detects entity → `Eiffel Tower`
2. **Wiki Search:** Retrieves info → *"The Eiffel Tower is a wrought-iron lattice tower... owned by the City of Paris."*
3. **Conflict Detection:** Model sees "Owned by City of Paris" vs "Sold to private company"
4. **Prediction:** `RUMOR (Fake)` ❌

---

## 📁 Dataset

This project uses the **PHEME dataset** for rumor detection, which contains:
- Twitter conversation threads about breaking news events
- 9 different events (Charlie Hebdo, Ferguson, Germanwings crash, etc.)
- Labeled as `rumour` or `non-rumour`

**Dataset structure:**
```
data/raw/archive/all-rnr-annotated-threads_1/
├── charliehebdo-all-rnr-threads/
│   ├── rumours/
│   └── non-rumours/
├── ferguson-all-rnr-threads/
│   ├── rumours/
│   └── non-rumours/
└── ... (other events)
```

---

## ⚙️ Configuration

### Training Hyperparameters

Defined in `main(oop).py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_NAME` | `bert-base-uncased` | Pre-trained language model |
| `BATCH_SIZE` | 4 | Training batch size |
| `EPOCHS` | 3 | Number of training epochs |
| `MAX_LENGTH` | 128 | Maximum token sequence length |
| `SEED` | 42 | Random seed for reproducibility |
| `LEARNING_RATE` | 2e-5 | Adam optimizer learning rate |

---

## 🔧 Troubleshooting

### Common Issues

**1. `PROJECT_ROOT not found in .env file`**
- Ensure `.env` file exists in the project root
- Check the path format matches your OS

**2. `Dataset not found`**
- Run `prepare_data(oop).py` first to generate `dataset.csv`
- Verify the PHEME dataset is in `data/raw/archive/`

**3. `SpaCy model not found`**
- Run: `python -m spacy download en_core_web_sm`

**4. `CUDA out of memory`**
- Reduce `BATCH_SIZE` in `main(oop).py`
- Or set `use_cpu=True` in training arguments

---

## 📦 Dependencies

```
torch>=2.2.0
transformers>=4.38.0
datasets>=2.18.0
pandas>=2.2.0
scikit-learn>=1.4.0
spacy>=3.7.0
wikipedia-api>=0.6.0
python-dotenv>=1.0.0
tqdm>=4.66.0
jupyter>=1.0.0
accelerate>=0.27.0
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ELKP Framework                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Tweet   │ → │  SpaCy   │ → │Wikipedia │ → │   BERT   │  │
│  │  Input   │    │   NER    │    │   API    │    │Classifier│  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │              │               │               │          │
│       │         Extract          Retrieve        Classify       │
│       │         Entities         Knowledge      Rumor/Not       │
│       │              │               │               │          │
│       └──────────────┴───────────────┴───────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📜 Quick Start Checklist

```bash
# 1. Set up environment
echo "PROJECT_ROOT=your/project/path" > .env

# 2. Parse raw data
python "src/prepare_data(oop).py"

# 3. Inject knowledge
python "src/preprocess(oop).py"

# 4. Train model
python "main(oop).py"

# 5. Run demo
jupyter notebook demo/demo_notebook.ipynb
```

---

## 🙏 Acknowledgments

- **HuggingFace Transformers** - Pre-trained BERT model and training utilities
- **SpaCy** - Named Entity Recognition
- **Wikipedia-API** - Knowledge retrieval
- **PHEME Dataset** - Labeled rumor detection dataset

---

## 📄 License

This project is for educational purposes as part of an NLP course project.

---

## 👥 Authors

- University NLP Course Project

---

<p align="center">
  Made with ❤️ for fighting misinformation
</p>