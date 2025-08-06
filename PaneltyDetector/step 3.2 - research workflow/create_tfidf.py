import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def preprocess_text(text):
    """Preprocess text for better TF-IDF performance"""
    if not text or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def create_tfidf_vectors(input_file="motor_traffic_processed.csv", 
                        output_file="motor_traffic_tfidf.pkl"):
    """Create TF-IDF vectors from processed motor traffic data"""
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Please run 'python csv_converter.py' first to create the processed data.")
        return
    
    print("Loading processed data...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    # Filter out rows where c6 (text content) is empty
    df = df.dropna(subset=['c6'])
    df = df[df['c6'].str.strip() != '']
    print(f"Found {len(df)} rows with text content")
    
    if len(df) == 0:
        print("No text content found in the data!")
        return
    
    # Prepare data for TF-IDF
    print("Preprocessing text content...")
    processed_data = []
    documents = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
        text_content = preprocess_text(row['c6'])
        
        if text_content:  # Only include non-empty texts
            processed_data.append({
                'c2': row['c2'] if pd.notna(row['c2']) else '',
                'c3': row['c3'] if pd.notna(row['c3']) else '',
                'c4': row['c4'] if pd.notna(row['c4']) else '',
                'c5': row['c5'] if pd.notna(row['c5']) else '',
                'c6': text_content
            })
            documents.append(text_content)
    
    print(f"Prepared {len(documents)} documents for TF-IDF vectorization")
    
    if len(documents) == 0:
        print("No valid documents found for vectorization!")
        return
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit vocabulary size
        min_df=2,           # Ignore terms that appear in less than 2 documents
        max_df=0.8,         # Ignore terms that appear in more than 80% of documents
        stop_words='english',  # Remove English stop words
        ngram_range=(1, 2),    # Use both unigrams and bigrams
        sublinear_tf=True      # Apply sublinear tf scaling
    )
    
    try:
        # Fit and transform documents
        tfidf_vectors = vectorizer.fit_transform(documents)
        
        print(f"Created TF-IDF vectors with shape: {tfidf_vectors.shape}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Save to cache file
        cache_data = {
            'tfidf_vectors': tfidf_vectors,
            'processed_data': processed_data,
            'vectorizer': vectorizer,
            'vocabulary_size': len(vectorizer.vocabulary_),
            'num_documents': len(documents)
        }
        
        print(f"Saving TF-IDF data to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print("✓ TF-IDF vectors created and saved successfully!")
        print(f"Cache file: {output_file}")
        print(f"Documents: {len(documents)}")
        print(f"Features: {tfidf_vectors.shape[1]}")
        print(f"Sparsity: {1 - tfidf_vectors.nnz / (tfidf_vectors.shape[0] * tfidf_vectors.shape[1]):.3f}")
        
        # Show some sample vocabulary
        vocab_items = list(vectorizer.vocabulary_.items())[:20]
        print(f"\nSample vocabulary: {[word for word, _ in vocab_items]}")
        
    except Exception as e:
        print(f"Error creating TF-IDF vectors: {e}")

def main():
    """Main function"""
    print("TF-IDF Vector Creation Tool")
    print("=" * 40)
    
    create_tfidf_vectors()

if __name__ == "__main__":
    main()
