import os
import sys
import pandas as pd
import spacy
import wikipediaapi
from tqdm import tqdm
from dotenv import load_dotenv

class KnowledgeEngine:
    """Handles Named Entity Recognition and Knowledge Retrieval."""
    
    MODEL_NAME = "en_core_web_sm"
    USER_AGENT = "RumorDetectionProject/1.0 (student@university.edu)"

    def __init__(self):
        self._load_spacy()
        self.wiki = wikipediaapi.Wikipedia(language='en', user_agent=self.USER_AGENT)

    def _load_spacy(self):
        """Loads SpaCy model, downloading it if necessary."""
        try:
            self.nlp = spacy.load(self.MODEL_NAME)
        except OSError:
            print(f"Downloading {self.MODEL_NAME}...")
            os.system(f"python -m spacy download {self.MODEL_NAME}")
            self.nlp = spacy.load(self.MODEL_NAME)

    def get_augmented_text(self, text):
        """
        Extracts entity, fetches knowledge, and formats the prompt.
        Logic based on ELKP framework[cite: 207, 307].
        """
        doc = self.nlp(text)
        
        # 1. Named Entity Recognition (NER) [cite: 207]
        if not doc.ents:
            return f"Tweet: {text}"

        entity_name = doc.ents[0].text
        knowledge_snippet = self._fetch_wiki_summary(entity_name)

        # 2. Construct Knowledge-Powered Prompt 
        if knowledge_snippet:
            return f"Knowledge: {knowledge_snippet} [SEP] Tweet: {text}"
        
        return f"Tweet: {text}"

    def _fetch_wiki_summary(self, entity_name):
        """Queries Wikipedia and returns a short summary[cite: 219]."""
        try:
            page = self.wiki.page(entity_name)
            if page.exists():
                # Limit to 200 chars to reduce noise
                return f"Entity: {entity_name} | Info: {page.summary[:200]}..."
        except Exception:
            return None
        return None


class PreprocessingPipeline:
    """Manages file I/O and data transformation."""

    def __init__(self, project_root):
        self.project_root = project_root
        self.input_path = os.path.join(project_root, "data", "processed", "dataset.csv")
        self.output_path = os.path.join(project_root, "data", "processed", "augmented_dataset.csv")
        self.engine = KnowledgeEngine()

    def run(self):
        if not os.path.exists(self.input_path):
            print(f"❌ Error: Input file not found at {self.input_path}")
            return

        print(f"Reading data from: {self.input_path}")
        df = pd.read_csv(self.input_path)
        
        # Note: To test quickly, append .head(100) to the line above
        
        print("🧠 Injecting knowledge (this may take time)...")
        tqdm.pandas()
        
        # Apply the augmentation logic to the 'text' column
        augmented_texts = [self.engine.get_augmented_text(text) for text in tqdm(df['text'])]
        
        df['text'] = augmented_texts
        self._save(df)

    def _save(self, df):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"✅ Knowledge injection complete! Saved to: {self.output_path}")


if __name__ == "__main__":
    load_dotenv()
    root_dir = os.getenv("PROJECT_ROOT")

    if not root_dir:
        print("❌ Error: PROJECT_ROOT not found in .env")
    else:
        pipeline = PreprocessingPipeline(root_dir)
        pipeline.run()