import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import csv
from tqdm import tqdm
import time  # Add import for time delay

class TestDataAnalyzer:
    def __init__(self, embeddings_cache_file="motor_traffic_embeddings.pkl"):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load embeddings and processed data from cache
        self.embeddings_cache_file = embeddings_cache_file
        self.embeddings, self.processed_data = self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings and processed data from cache file"""
        if not os.path.exists(self.embeddings_cache_file):
            raise FileNotFoundError(
                f"Embeddings cache file '{self.embeddings_cache_file}' not found! "
                f"Please run 'python create_embeddings.py' first to create the embeddings."
            )
        
        try:
            print("Loading embeddings from cache...")
            with open(self.embeddings_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            embeddings = np.array(cached_data['embeddings'])
            processed_data = cached_data['processed_data']
            
            print(f"Loaded {len(processed_data)} sections with embeddings!")
            return embeddings, processed_data
            
        except Exception as e:
            raise Exception(f"Error loading embeddings cache: {e}")
    
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
    
    def find_best_match(self, scenario_text, min_similarity_threshold=0.2):
        """Find the best matching section for a given scenario"""
        # Get embedding for the scenario
        query_embedding = self._get_embedding(scenario_text)
        if not query_embedding:
            return None
        
        # Calculate cosine similarity
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Find the best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity < min_similarity_threshold:
            return None
        
        item = self.processed_data[best_idx]
        return {
            'similarity_score': float(best_similarity),
            'section_number': item['c2'] if item['c2'] else '',
            'subsection': item['c3'] if item['c3'] else '',
            'paragraph': item['c4'] if item['c4'] else '',
            'subparagraph': item['c5'] if item['c5'] else ''
        }
    
    def find_multiple_matches(self, scenario_text, min_similarity_threshold=0.2, max_matches=3):
        """Find multiple matching sections for a given scenario"""
        # Get embedding for the scenario
        query_embedding = self._get_embedding(scenario_text)
        if not query_embedding:
            return []
        
        # Calculate cosine similarity
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Filter by minimum similarity threshold
        valid_indices = np.where(similarities >= min_similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Sort by similarity scores
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # Group results by section number and keep only top 3 sections
        section_groups = {}
        
        for idx in sorted_indices[:max_matches*3]:  # Get more candidates for grouping
            item = self.processed_data[idx]
            section_number = item['c2'] if item['c2'] else 'Unknown'
            similarity_score = similarities[idx]
            
            if section_number not in section_groups:
                section_groups[section_number] = []
            
            section_groups[section_number].append({
                'similarity_score': float(similarity_score),
                'section_number': item['c2'] if item['c2'] else '',
                'subsection': item['c3'] if item['c3'] else '',
                'paragraph': item['c4'] if item['c4'] else '',
                'subparagraph': item['c5'] if item['c5'] else ''
            })
        
        # Sort sections by their highest similarity score and take top sections
        sorted_sections = sorted(section_groups.items(), 
                               key=lambda x: max(result['similarity_score'] for result in x[1]), 
                               reverse=True)[:max_matches]
        
        # Collect results from top sections
        matches = []
        for section_number, section_results in sorted_sections:
            # Sort results within each section by similarity
            section_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            # Add the best result from this section
            matches.append(section_results[0])
        
        return matches
    
    def analyze_testdata(self, input_file="testdata.csv", output_file="analyzed_testdata.csv"):
        """Analyze test data and add section information"""
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found!")
            print("Please ensure the testdata.csv file exists in the current directory.")
            return
        
        # Read the CSV file
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
            print(f"Loaded {len(df)} test scenarios from {input_file}")
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            return
        
        # Check required columns
        if 'Title' not in df.columns or 'Description' not in df.columns:
            print("Error: CSV file must contain 'Title' and 'Description' columns")
            return
        
        # Initialize new columns for multiple matches
        for i in range(1, 4):  # Support up to 3 matches
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
        
        # Process each row with progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing scenarios"):
            scenario_text = f"{row['Title']} {row['Description']}"
            
            print(f"\nProcessing {index + 1}/{len(df)}: {row['Title'][:50]}...")
            
            # Find multiple matches
            match_results = self.find_multiple_matches(scenario_text, max_matches=3)
            
            if match_results:
                df.at[index, 'Total_Matches'] = len(match_results)
                df.at[index, 'Match_Status'] = 'MATCHED'
                matched_count += 1
                
                # Store each match
                for i, match_result in enumerate(match_results, 1):
                    df.at[index, f'Section_Number_{i}'] = match_result['section_number']
                    df.at[index, f'Subsection_{i}'] = match_result['subsection']
                    df.at[index, f'Paragraph_{i}'] = match_result['paragraph']
                    df.at[index, f'Subparagraph_{i}'] = match_result['subparagraph']
                    df.at[index, f'Similarity_Score_{i}'] = match_result['similarity_score']
                
                # Display matches
                sections = [m['section_number'] for m in match_results if m['section_number']]
                scores = [f"{m['similarity_score']:.3f}" for m in match_results]
                print(f"  ✓ Matched to {len(match_results)} section(s): {', '.join(sections)} (Scores: {', '.join(scores)})")
            else:
                df.at[index, 'Match_Status'] = 'NO_MATCH'
                no_match_count += 1
                print(f"  ✗ No sufficient match found")
            
            # Add 10-second delay between requests (except for the last item)
            if index < len(df) - 1:
                time.sleep(10)
        
        # Save results
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
            section_counts = df[df['Section_Number'] != '']['Section_Number'].value_counts()
            if not section_counts.empty:
                print(f"\nSECTION DISTRIBUTION:")
                for section, count in section_counts.head(10).items():
                    print(f"  Section {section}: {count} scenario(s)")
            
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """Main function"""
    print("Initializing Test Data Analyzer...")
    print("=" * 60)
    
    try:
        analyzer = TestDataAnalyzer()
        print("Analyzer ready!")
        
        # Check if testdata.csv exists
        input_file = "testdata.csv"
        if not os.path.exists(input_file):
            print(f"\nCreating sample {input_file} file...")
            # Create sample data if file doesn't exist
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
        
        # Run analysis
        analyzer.analyze_testdata()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo get started:")
        print("1. Run: python csv_converter.py")
        print("2. Run: python create_embeddings.py")
        print("3. Run: python analyze_testdata.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
