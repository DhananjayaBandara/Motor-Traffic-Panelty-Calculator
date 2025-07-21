import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import pickle

class EmbeddingCreator:
    def __init__(self, csv_file_path):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load and process the CSV data
        self.df = pd.read_csv(csv_file_path)
        self.processed_data = self._process_data()
    
    def _process_data(self):
        """Process the CSV data and create searchable content"""
        processed_data = []
        
        for index, row in self.df.iterrows():
            # Combine c1 and c6 for searchable content
            c1_content = str(row['c1']) if pd.notna(row['c1']) else ""
            c6_content = str(row['c6']) if pd.notna(row['c6']) else ""
            
            # Create searchable text by combining c1 and c6
            searchable_text = f"{c1_content} {c6_content}".strip()
            
            # Skip rows with no searchable content
            if searchable_text and searchable_text != "nan nan":
                processed_data.append({
                    'index': index,
                    'searchable_text': searchable_text,
                    'c1': c1_content,
                    'c2': str(row['c2']) if pd.notna(row['c2']) else "",
                    'c3': str(row['c3']) if pd.notna(row['c3']) else "",
                    'c4': str(row['c4']) if pd.notna(row['c4']) else "",
                    'c5': str(row['c5']) if pd.notna(row['c5']) else "",
                    'c6': c6_content
                })
        
        return processed_data
    
    def _get_embedding(self, text):
        """Get embedding for a given text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def create_embeddings(self):
        """Create embeddings for all searchable content"""
        embeddings = []
        total_items = len(self.processed_data)
        print(f"Creating embeddings for {total_items} Motor Traffic Act sections...")
        print("This process will create the embeddings cache file...")
        
        for i, item in enumerate(self.processed_data):
            # Progress indicator
            progress = (i + 1) / total_items * 100
            print(f"Progress: {progress:.1f}% ({i+1}/{total_items})", end='\r')
            
            embedding = self._get_embedding(item['searchable_text'])
            if embedding:
                embeddings.append(embedding)
            else:
                embeddings.append([0] * 1536)  # Default embedding size for text-embedding-3-small
        
        print("\nEmbeddings created successfully!")
        return np.array(embeddings)
    
    def save_embeddings(self, embeddings, cache_file="motor_traffic_embeddings.pkl"):
        """Save embeddings and processed data to cache file"""
        try:
            cache_data = {
                'embeddings': embeddings.tolist(),
                'processed_data': self.processed_data,
                'data_length': len(self.processed_data)
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Embeddings and data cached successfully to: {cache_file}")
            return True
        except Exception as e:
            print(f"Error saving embeddings cache: {e}")
            return False

def main():
    """Main function to create embeddings"""
    csv_path = "motor_traffic_act_utf8.csv"
    cache_file = "motor_traffic_embeddings.pkl"
    
    # Check if the UTF-8 file exists
    if not os.path.exists(csv_path):
        print("Error: UTF-8 encoded CSV file not found!")
        print("Please run the csv_converter.py script first to convert the CSV file to UTF-8 encoding.")
        print("Command: python csv_converter.py")
        return
    
    # Check if embeddings already exist
    if os.path.exists(cache_file):
        overwrite = input(f"Embeddings file '{cache_file}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Operation cancelled.")
            return
    
    try:
        print("=== Motor Traffic Act Embeddings Creator ===")
        print("Initializing embedding creator...")
        
        creator = EmbeddingCreator(csv_path)
        print(f"Loaded {len(creator.processed_data)} sections from CSV")
        
        # Create embeddings
        embeddings = creator.create_embeddings()
        
        # Save to cache
        if creator.save_embeddings(embeddings, cache_file):
            print("\n=== Embeddings Creation Complete ===")
            print(f"Embeddings saved to: {cache_file}")
            print("You can now use the RAG application with these embeddings.")
        else:
            print("\n=== Embeddings Creation Failed ===")
            print("Please check the error messages above.")
            
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")

if __name__ == "__main__":
    main()
