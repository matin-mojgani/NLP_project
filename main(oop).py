import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    set_seed
)
from datasets import Dataset
from dotenv import load_dotenv

# --- Configuration Constants ---
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 4
EPOCHS = 3           
MAX_LENGTH = 128
SEED = 42            

class RumorClassifierTrainer:
    """
    Manages the full lifecycle of training the Rumor Detection model:
    Data loading, Preprocessing, Training, Evaluation, and Model Persistence.
    """

    def __init__(self):
        # Initialize environment and paths
        load_dotenv()
        self.project_root = os.getenv("PROJECT_ROOT")
        
        if not self.project_root:
            self.project_root = os.path.dirname(os.path.abspath(__file__))

        self.data_path = os.path.join(self.project_root, "data", "processed", "augmented_dataset.csv")
        self.output_dir = os.path.join(self.project_root, "results")
        self.model_save_path = os.path.join(self.project_root, "models", "rumor_model")
        
        # Ensure reproducibility across runs
        self._set_deterministic(SEED)
        
        # Initialize tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _set_deterministic(self, seed: int):
        """Sets seeds for all random number generators to ensure reproducible results."""
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _compute_metrics(self, eval_pred):
        """Calculates Accuracy and F1-Score for the Trainer."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': acc,
            'f1': f1
        }

    def run(self):
        """Main execution flow."""
        if not os.path.exists(self.data_path):
            print(f"❌ Error: Dataset not found at {self.data_path}")
            return

        print("1. Loading and splitting data...")
        # Loads data and crucially saves the test set for the demo
        train_dataset, test_dataset = self._load_and_prepare_data()

        print("2. Tokenizing data...")
        tokenized_train = train_dataset.map(self._tokenize_function, batched=True)
        tokenized_test = test_dataset.map(self._tokenize_function, batched=True)

        print("3. Initializing Model...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        # Configure training parameters
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",      # Evaluate at the end of every epoch
            save_strategy="epoch",      # Save checkpoint at the end of every epoch
            learning_rate=2e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            load_best_model_at_end=True, # Ensure we keep the best model, not the last one
            metric_for_best_model="accuracy",
            use_cpu=not torch.cuda.is_available(),
            seed=SEED
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=self._compute_metrics
        )

        # Baseline evaluation (Before training)
        print("\n🧐 Evaluating Baseline (Before Training)...")
        initial_metrics = trainer.evaluate()
        print(f"   >> Initial Accuracy: {initial_metrics['eval_accuracy']:.2%}")

        # Start Training
        print("\n🚀 Starting Training...")
        trainer.train()
        
        # Final evaluation (After training)
        print("\n🎓 Evaluating Trained Model (After Training)...")
        final_metrics = trainer.evaluate()
        print(f"   >> Final Accuracy: {final_metrics['eval_accuracy']:.2%}")

        # Calculate and display improvement
        improvement = final_metrics['eval_accuracy'] - initial_metrics['eval_accuracy']
        print(f"\n📈 Improvement: +{improvement:.2%}")

        self._save_model(model)

    def _load_and_prepare_data(self):
        """
        Reads the augmented CSV, performs train/test split, 
        and saves the test set separately for the Demo notebook.
        """
        print("   -> Reading augmented dataset...")
        df = pd.read_csv(self.data_path).dropna(subset=['text', 'label'])
        
        # Split data using a fixed seed
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
        
        # --- Crucial Step: Persist Test Data ---
        test_save_path = os.path.join(self.project_root, "data", "processed", "test_dataset.csv")
        test_df.to_csv(test_save_path, index=False)
        print(f"   -> ✅ Test data saved separately to: {test_save_path}")
        # ---------------------------------------

        return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

    def _tokenize_function(self, examples):
        """Hugging Face compatible tokenizer function."""
        return self.tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LENGTH
        )

    def _save_model(self, model):
        """Saves the fine-tuned model and tokenizer to disk."""
        print(f"💾 Saving best model to {self.model_save_path}")
        model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)

if __name__ == "__main__":
    try:
        trainer = RumorClassifierTrainer()
        trainer.run()
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()