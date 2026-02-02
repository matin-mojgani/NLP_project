import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

class PhemeDatasetBuilder:
    # Constants for dataset structure
    LABEL_MAP = {'rumours': 1, 'non-rumours': 0}
    SOURCE_DIR_NAME = 'source-tweets'

    def __init__(self, root_path, output_path):
        self.root_path = root_path
        self.output_path = output_path
        self.records = []

    def run(self):
        """Main execution method to traverse directories and save data."""
        if not os.path.exists(self.root_path):
            print(f"❌ Error: Root directory not found: {self.root_path}")
            return

        # Get valid event directories (ignoring system files)
        events = [d for d in os.listdir(self.root_path) 
                  if os.path.isdir(os.path.join(self.root_path, d)) and not d.startswith('.')]
        
        print(f"Found {len(events)} events. Starting extraction...")

        for event in tqdm(events, desc="Processing Events"):
            self._process_single_event(event)

        self._save_to_csv()

    def _process_single_event(self, event_name):
        """Iterates through rumor/non-rumor folders for a specific event."""
        event_path = os.path.join(self.root_path, event_name)

        for folder_name, label_code in self.LABEL_MAP.items():
            category_path = os.path.join(event_path, folder_name)
            
            if os.path.exists(category_path):
                self._extract_threads(category_path, label_code, event_name)

    def _extract_threads(self, category_path, label, event_name):
        """Loops through conversation threads within a category."""
        # Filter valid thread directories
        threads = [t for t in os.listdir(category_path) 
                   if os.path.isdir(os.path.join(category_path, t)) and not t.startswith('.')]

        for thread_id in threads:
            thread_path = os.path.join(category_path, thread_id)
            source_dir = os.path.join(thread_path, self.SOURCE_DIR_NAME)
            
            tweet_text = self._read_source_tweet(source_dir)
            if tweet_text:
                self.records.append({
                    'text': tweet_text,
                    'label': label,
                    'event': event_name
                })

    def _read_source_tweet(self, directory):
        """Finds and reads the valid JSON file in the source directory."""
        if not os.path.exists(directory):
            return None

        try:
            # Filter for valid JSONs, ignoring MacOS hidden files (._)
            valid_files = [
                f for f in os.listdir(directory) 
                if f.endswith('.json') and not f.startswith('._')
            ]

            if not valid_files:
                return None

            # Pick the first valid file
            file_path = os.path.join(directory, valid_files[0])
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('text', '').strip()

        except Exception as e:
            # Log error briefly and continue
            print(f"⚠️ Error reading file in {directory}: {e}")
            return None

    def _save_to_csv(self):
        """Saves the accumulated records to a CSV file."""
        if not self.records:
            print("❌ No tweets found. Please check dataset path.")
            return

        df = pd.DataFrame(self.records)
        print(f"\n✅ Extraction Complete! Total Tweets: {len(df)}")
        print(f"Distribution:\n{df['label'].value_counts()}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"💾 Saved processed data to: {self.output_path}")


if __name__ == "__main__":
    load_dotenv()
    
    # Load project root from environment variable
    project_root = os.getenv("PROJECT_ROOT")
    
    if not project_root:
        print("❌ PROJECT_ROOT not found in .env file.")
    else:
        # Construct paths
        input_dir = os.path.join(project_root, "data", "raw", "archive", "all-rnr-annotated-threads_1")
        output_file = os.path.join(project_root, "data", "processed", "dataset.csv")
        
        # Run processor
        builder = PhemeDatasetBuilder(input_dir, output_file)
        builder.run()