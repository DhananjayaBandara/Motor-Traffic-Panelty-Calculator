import pandas as pd
import numpy as np
import os
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import csv
from tqdm import tqdm
import time
from domain_synonyms import DOMAIN_SYNONYMS

nltk.download('punkt', quiet=True)

class MotorTrafficActRAG_TinyLLM:
    def __init__(self, embeddings_cache_file="motor_traffic_embeddings_tinyllm.pkl", csv_file="motor_traffic_act_utf8.csv"):
        self.embeddings_cache_file = embeddings_cache_file
        self.csv_file = csv_file
        self.vectorizer = None
        self.tfidf_matrix = None
        self.processed_data = None
        self.vocabulary = set()
        self._load_or_create_embeddings()

    def _preprocess(self, text):
        """Simple preprocessing using NLTK tokenization"""
        if not text or pd.isna(text):
            return ""
        try:
            tokens = word_tokenize(str(text).lower())
            # Filter out very short tokens and numbers-only tokens
            tokens = [token for token in tokens if len(token) >= 2 and not token.isdigit()]
            return " ".join(tokens)
        except:
            # Fallback to basic preprocessing if NLTK fails
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    def _create_enhanced_searchable_text(self, c1, c6):
        """Create searchable text with title emphasis"""
        c1_clean = self._preprocess(c1)
        c6_clean = self._preprocess(c6)
        if c1_clean and c6_clean:
            return f"{c1_clean} {c1_clean} {c6_clean}"  # Emphasize title
        elif c1_clean:
            return f"{c1_clean} {c1_clean}"
        elif c6_clean:
            return c6_clean
        return ""

    def _process_data(self, df):
        processed_data = []
        for index, row in df.iterrows():
            c1_content = str(row['c1']) if pd.notna(row['c1']) else ""
            c6_content = str(row['c6']) if pd.notna(row['c6']) else ""
            searchable_text = self._create_enhanced_searchable_text(c1_content, c6_content)
            
            if searchable_text and len(searchable_text.strip()) > 10:
                processed_item = {
                    'index': index,
                    'searchable_text': searchable_text,
                    'c1': c1_content,
                    'c2': str(row['c2']) if pd.notna(row['c2']) else "",
                    'c3': str(row['c3']) if pd.notna(row['c3']) else "",
                    'c4': str(row['c4']) if pd.notna(row['c4']) else "",
                    'c5': str(row['c5']) if pd.notna(row['c5']) else "",
                    'c6': c6_content,
                    'content_length': len(c6_content),
                    'has_title': bool(c1_content.strip()),
                    'has_section_number': bool(str(row['c2']).strip()) if pd.notna(row['c2']) else False
                }
                processed_data.append(processed_item)
                
                # Build vocabulary for query expansion
                words = searchable_text.split()
                self.vocabulary.update(words)
                
        return processed_data

    def _load_or_create_embeddings(self):
        if os.path.exists(self.embeddings_cache_file):
            with open(self.embeddings_cache_file, 'rb') as f:
                cache = pickle.load(f)
            self.vectorizer = cache['vectorizer']
            self.tfidf_matrix = cache['tfidf_matrix']
            self.processed_data = cache['processed_data']
            self.vocabulary = cache.get('vocabulary', set())
            
            # Fix missing keys in old cache format
            for item in self.processed_data:
                if 'has_title' not in item:
                    item['has_title'] = bool(item.get('c1', '').strip())
                if 'has_section_number' not in item:
                    item['has_section_number'] = bool(item.get('c2', '').strip())
                if 'content_length' not in item:
                    item['content_length'] = len(item.get('c6', ''))
            
            print(f"Loaded Tiny LLM embeddings from {self.embeddings_cache_file}")
        else:
            if not os.path.exists(self.csv_file):
                raise FileNotFoundError(f"CSV file '{self.csv_file}' not found!")
            print("Processing data and creating Tiny LLM embeddings...")
            df = pd.read_csv(self.csv_file)
            self.processed_data = self._process_data(df)
            
            # Create simple TF-IDF vectorizer (Tiny LLM approach)
            texts = [item['searchable_text'] for item in self.processed_data]
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=5000,  # Smaller feature set for "tiny" approach
                ngram_range=(1, 2)  # Simpler n-grams
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'tfidf_matrix': self.tfidf_matrix,
                    'processed_data': self.processed_data,
                    'vocabulary': self.vocabulary
                }, f)
            print(f"Created and cached Tiny LLM embeddings to {self.embeddings_cache_file}")

    def _expand_query(self, query):
        """Simple query expansion using domain synonyms and vocabulary"""
        if not query:
            return ""
        
        query_processed = self._preprocess(query)
        query_words = query_processed.split()
        expanded_words = query_words.copy()
        
        for word in query_words:
            # Use DOMAIN_SYNONYMS for expansion
            if word in DOMAIN_SYNONYMS:
                expanded_words.extend(DOMAIN_SYNONYMS[word])
            else:
                # Add simple vocabulary matches
                matches = [vocab_word for vocab_word in self.vocabulary if word in vocab_word and len(vocab_word) > len(word)]
                expanded_words.extend(matches[:1])  # Limit to 1 match for "tiny" approach
        
        return ' '.join(set(expanded_words))  # Remove duplicates

    def _calculate_content_boost(self, item, base_score):
        boost_factor = 1.0
        if item['has_title']:
            boost_factor *= 1.1  # Smaller boost for tiny approach
        if item['has_section_number']:
            boost_factor *= 1.05
        if item['content_length'] > 100:
            boost_factor *= 1.02
        return base_score * boost_factor

    def find_multiple_matches(self, scenario_text, min_similarity_threshold=0.1, max_matches=3):
        if not scenario_text:
            return []
        
        expanded_query = self._expand_query(scenario_text)
        query_vec = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Apply content boosting
        boosted_scores = []
        for i, score in enumerate(similarities):
            if i < len(self.processed_data):
                boosted_score = self._calculate_content_boost(self.processed_data[i], score)
                boosted_scores.append(boosted_score)
            else:
                boosted_scores.append(score)
        
        boosted_scores = np.array(boosted_scores)
        valid_indices = np.where(boosted_scores >= min_similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        sorted_indices = valid_indices[np.argsort(boosted_scores[valid_indices])[::-1]]
        
        # Group by section
        section_groups = {}
        for idx in sorted_indices[:max_matches*3]:
            item = self.processed_data[idx]
            section_number = item['c2'] if item['c2'] else 'Unknown'
            similarity_score = boosted_scores[idx]
            
            if section_number not in section_groups:
                section_groups[section_number] = []
            
            section_groups[section_number].append({
                'similarity_score': float(similarity_score),
                'section_number': item['c2'] if item['c2'] else '',
                'subsection': item['c3'] if item['c3'] else '',
                'paragraph': item['c4'] if item['c4'] else '',
                'subparagraph': item['c5'] if item['c5'] else ''
            })
        
        # Sort sections by best score
        sorted_sections = sorted(section_groups.items(),
                                 key=lambda x: max(result['similarity_score'] for result in x[1]),
                                 reverse=True)[:max_matches]
        
        matches = []
        for section_number, section_results in sorted_sections:
            section_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            matches.append(section_results[0])
        
        return matches

    def analyze_testdata(self, input_file="testdata.csv", output_file="analyzed_testdata_tinyllm.csv"):
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found!")
            print("Please ensure the testdata.csv file exists in the current directory.")
            return
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
            print(f"Loaded {len(df)} test scenarios from {input_file}")
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            return
        if 'Title' not in df.columns or 'Description' not in df.columns:
            print("Error: CSV file must contain 'Title' and 'Description' columns")
            return
        for i in range(1, 4):
            df[f'Section_Number_{i}'] = ''
            df[f'Subsection_{i}'] = ''
            df[f'Paragraph_{i}'] = ''
            df[f'Subparagraph_{i}'] = ''
            df[f'Similarity_Score_{i}'] = 0.0
        df['Total_Matches'] = 0
        df['Match_Status'] = ''
        print("\nStarting analysis process...")
        print("=" * 60)
        
        matched_count = 0
        no_match_count = 0
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing scenarios"):
            scenario_text = f"{row['Title']} {row['Description']}"
            print(f"\nProcessing {index + 1}/{len(df)}: {row['Title'][:50]}...")
            
            match_results = self.find_multiple_matches(scenario_text, max_matches=3)
            
            if match_results:
                df.at[index, 'Total_Matches'] = len(match_results)
                df.at[index, 'Match_Status'] = 'MATCHED'
                matched_count += 1
                
                for i, match_result in enumerate(match_results, 1):
                    df.at[index, f'Section_Number_{i}'] = match_result['section_number']
                    df.at[index, f'Subsection_{i}'] = match_result['subsection']
                    df.at[index, f'Paragraph_{i}'] = match_result['paragraph']
                    df.at[index, f'Subparagraph_{i}'] = match_result['subparagraph']
                    df.at[index, f'Similarity_Score_{i}'] = match_result['similarity_score']
                
                sections = [m['section_number'] for m in match_results if m['section_number']]
                scores = [f"{m['similarity_score']:.3f}" for m in match_results]
                print(f"  ✓ Matched to {len(match_results)} section(s): {', '.join(sections)} (Scores: {', '.join(scores)})")
            else:
                df.at[index, 'Match_Status'] = 'NO_MATCH'
                no_match_count += 1
                print(f"  ✗ No sufficient match found")
            
            if index < len(df) - 1:
                time.sleep(0.3)  # Shorter delay for Tiny LLM
        
        try:
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n" + "=" * 60)
            print("ANALYSIS COMPLETE!")
            print(f"Results saved to: {output_file}")
            print(f"\nSUMMARY:")
            print(f"  Total scenarios processed: {len(df)}")
            print(f"  Successfully matched: {matched_count}")
            print(f"  No match found: {no_match_count}")
            print(f"  Match rate: {(matched_count/len(df)*100):.1f}%")
            # Show section distribution
            section_cols = [col for col in df.columns if col.startswith('Section_Number_')]
            all_sections = pd.concat([df[col] for col in section_cols])
            section_counts = all_sections[all_sections != ''].value_counts()
            if not section_counts.empty:
                print(f"\nSECTION DISTRIBUTION:")
                for section, count in section_counts.head(10).items():
                    print(f"  Section {section}: {count} scenario(s)")
        except Exception as e:
            print(f"Error saving results: {e}")

    def search(self, query, top_k=5, min_similarity=0.01):
        if not query:
            return []
        
        expanded_query = self._expand_query(query)
        query_vec = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Apply content boosting
        boosted_scores = []
        for i, score in enumerate(similarities):
            if i < len(self.processed_data):
                boosted_score = self._calculate_content_boost(self.processed_data[i], score)
                boosted_scores.append(boosted_score)
            else:
                boosted_scores.append(score)
        
        boosted_scores = np.array(boosted_scores)
        top_indices = np.argsort(boosted_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.processed_data) and boosted_scores[idx] >= min_similarity:
                item = self.processed_data[idx]
                results.append({
                    'similarity_score': float(boosted_scores[idx]),
                    'original_score': float(similarities[idx]),
                    'c1': item['c1'],
                    'c2': item['c2'],
                    'c3': item['c3'],
                    'c4': item['c4'],
                    'c5': item['c5'],
                    'c6': item['c6'],
                    'searchable_text': item['searchable_text']
                })
        
        return results

    def format_results(self, results):
        if not results:
            return "No relevant sections found."
        
        formatted_output = "=== MOTOR TRAFFIC ACT - SEARCH RESULTS (Tiny LLM/TF-IDF) ===\n\n"
        for i, result in enumerate(results, 1):
            formatted_output += f"Result {i} (Enhanced Score: {result['similarity_score']:.4f}, Base: {result['original_score']:.4f}):\n"
            formatted_output += f"Section Title: {result['c1']}\n"
            if result['c2']:
                formatted_output += f"Section Number: {result['c2']}\n"
            if result['c3']:
                formatted_output += f"Subsection: {result['c3']}\n"
            if result['c4']:
                formatted_output += f"Paragraph: {result['c4']}\n"
            if result['c5']:
                formatted_output += f"Subparagraph: {result['c5']}\n"
            formatted_output += f"Content: {result['c6']}\n"
            formatted_output += "-" * 80 + "\n\n"
        return formatted_output

def main():
    print("Initializing Motor Traffic Act RAG System (Tiny LLM)...")
    print("=" * 60)
    try:
        analyzer = MotorTrafficActRAG_TinyLLM()
        print("Tiny LLM RAG System ready!")
        print("Features: Simple TF-IDF, NLTK tokenization, query expansion, content boosting")
        
        input_file = "testdata.csv"
        if not os.path.exists(input_file):
            print(f"\nCreating sample {input_file} file...")
            sample_data = [
                ["The Speeding Motorcyclist", "A motorcyclist was caught speeding at 80 km/h in a 50 km/h zone during peak hours."],
                ["The Phone Call Driver", "A driver was observed talking on a mobile phone while driving through a busy intersection."],
                ["The Drunk Driver", "A driver was arrested with a blood alcohol content of 0.08% after being stopped at a checkpoint."],
                ["The Reckless Overtaking", "A car overtook multiple vehicles on a double yellow line causing other drivers to brake suddenly."],
                ["The Parking Violation", "A vehicle was parked in a disabled parking space without proper authorization."]
            ]
            with open(input_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Title', 'Description'])
                writer.writerows(sample_data)
            print(f"Sample {input_file} created with {len(sample_data)} scenarios")
        
        analyzer.analyze_testdata()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo get started:")
        print("1. Ensure your CSV file is in the correct location.")
        print("2. The system will create the Tiny LLM model cache automatically.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()