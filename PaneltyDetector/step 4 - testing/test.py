import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class MotorTrafficActRAG:
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
    
    def search(self, query, min_similarity_threshold=0.3, max_results=10):
        """Search for relevant sections with intelligent similarity filtering"""
        # Get embedding for the query
        query_embedding = self._get_embedding(query)
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
        
        for idx in sorted_indices[:max_results]:
            item = self.processed_data[idx]
            section_number = item['c2'] if item['c2'] else 'Unknown'
            similarity_score = similarities[idx]
            
            if section_number not in section_groups:
                section_groups[section_number] = []
            
            section_groups[section_number].append({
                'similarity_score': float(similarity_score),
                'c1': item['c1'],
                'c2': item['c2'],
                'c3': item['c3'],
                'c4': item['c4'],
                'c5': item['c5'],
                'c6': item['c6'],
                'searchable_text': item['searchable_text']
            })
        
        # Sort sections by their highest similarity score and take top 3
        sorted_sections = sorted(section_groups.items(), 
                               key=lambda x: max(result['similarity_score'] for result in x[1]), 
                               reverse=True)[:3]
        
        # Collect results from top 3 sections
        results = []
        for section_number, section_results in sorted_sections:
            # Sort results within each section by similarity
            section_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Add the best result from this section
            results.append(section_results[0])
        
        return results
    
    def format_results(self, results):
        """Format the search results with enhanced similarity information"""
        if not results:
            return "No relevant sections found for your scenario."
        
        formatted_output = "=== MOTOR TRAFFIC ACT - SCENARIO ANALYSIS ===\n\n"
        formatted_output += f"Found {len(results)} relevant section(s) for your scenario:\n\n"
        
        for i, result in enumerate(results, 1):
            similarity_percentage = result['similarity_score'] * 100
            
            # Determine relevance level
            if similarity_percentage >= 70:
                relevance = "HIGHLY RELEVANT"
            elif similarity_percentage >= 50:
                relevance = "MODERATELY RELEVANT"
            else:
                relevance = "POSSIBLY RELEVANT"
            
            formatted_output += f"Result {i} - {relevance} ({similarity_percentage:.1f}% match):\n"
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
            formatted_output += "=" * 80 + "\n\n"
        
        return formatted_output

def main():
    """Main function to demonstrate the RAG system"""
    try:
        print("Initializing Motor Traffic Act RAG System...")
        rag = MotorTrafficActRAG()
        print("Motor Traffic Act RAG System ready!")
        print("You can now search for scenarios in the Motor Traffic Act.")
        print("Describe your traffic scenario and get the most relevant legal provisions.")
        print("Type 'quit' to exit.\n")
        
        while True:
            # Get user input
            user_query = input("Describe your traffic scenario: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using the Motor Traffic Act RAG System!")
                break
            
            if not user_query:
                print("Please describe a valid traffic scenario.\n")
                continue
            
            # Search for relevant sections
            print("\nAnalyzing scenario...")
            results = rag.search(user_query, min_similarity_threshold=0.2, max_results=8)
            
            # Display results
            formatted_results = rag.format_results(results)
            print(formatted_results)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo get started:")
        print("1. Run: python csv_converter.py")
        print("2. Run: python create_embeddings.py")
        print("3. Run: python RAG.py")
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")


if __name__ == "__main__":
    main()
