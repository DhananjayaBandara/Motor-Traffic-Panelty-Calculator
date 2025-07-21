import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class MotorTrafficActRAG:
    def __init__(self, embeddings_cache_file="motor_traffic_embeddings.pkl", fine_ranges_file="fine_ranges.csv"):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load embeddings and processed data from cache
        self.embeddings_cache_file = embeddings_cache_file
        self.fine_ranges_file = fine_ranges_file
        self.embeddings, self.processed_data = self._load_embeddings()
        self.fine_ranges_data = self._load_fine_ranges()
    
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
    
    def _load_fine_ranges(self):
        """Load fine ranges data from CSV file"""
        try:
            if os.path.exists(self.fine_ranges_file):
                df = pd.read_csv(self.fine_ranges_file)
                print(f"Loaded fine ranges data with {len(df)} entries!")
                return df
            else:
                print(f"Fine ranges file '{self.fine_ranges_file}' not found. Continuing without fine information.")
                return None
        except Exception as e:
            print(f"Error loading fine ranges: {e}")
            return None
    
    def _get_fine_info(self, c2, c3, c4, c5):
        """Get fine information for given section identifiers"""
        if self.fine_ranges_data is None:
            return None
        
        # Convert inputs to strings and handle empty values
        c2_str = str(c2) if c2 and str(c2).strip() else ''
        c3_str = str(c3) if c3 and str(c3).strip() else ''
        c4_str = str(c4) if c4 and str(c4).strip() else ''
        c5_str = str(c5) if c5 and str(c5).strip() else ''
        
        # Search for matching fine information
        for _, row in self.fine_ranges_data.iterrows():
            row_c2 = str(row['c2']) if pd.notna(row['c2']) else ''
            row_c3 = str(row['c3']) if pd.notna(row['c3']) else ''
            row_c4 = str(row['c4']) if pd.notna(row['c4']) else ''
            row_c5 = str(row['c5']) if pd.notna(row['c5']) else ''
            
            # Check for exact match
            if (row_c2 == c2_str and row_c3 == c3_str and 
                row_c4 == c4_str and row_c5 == c5_str):
                return row.to_dict()
        
        return None
    
    def search(self, query, top_k=5):
        """Search for relevant sections based on the query"""
        # Get embedding for the query
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate cosine similarity
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar items
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.processed_data[idx]
            results.append({
                'similarity_score': float(similarities[idx]),
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
        """Format the search results for better readability"""
        if not results:
            return "No relevant sections found."
        
        formatted_output = "=== MOTOR TRAFFIC ACT - SEARCH RESULTS ===\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_output += f"Result {i} (Similarity: {result['similarity_score']:.3f}):\n"
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
            
            # Add fine information if available
            fine_info = self._get_fine_info(result['c2'], result['c3'], result['c4'], result['c5'])
            if fine_info:
                formatted_output += "\n--- PENALTY INFORMATION ---\n"
                
                # First conviction
                if pd.notna(fine_info.get('first_conviction_minimum_fine')) or pd.notna(fine_info.get('first_conviction_maximum_fine')):
                    formatted_output += "First Conviction:\n"
                    if pd.notna(fine_info.get('first_conviction_minimum_fine')) and pd.notna(fine_info.get('first_conviction_maximum_fine')):
                        formatted_output += f"  Fine: Rs. {fine_info['first_conviction_minimum_fine']:,.0f} - Rs. {fine_info['first_conviction_maximum_fine']:,.0f}\n"
                    if pd.notna(fine_info.get('first_conviction_imprisionment_months')):
                        formatted_output += f"  Imprisonment: {fine_info['first_conviction_imprisionment_months']} months\n"
                    if pd.notna(fine_info.get('first_conviction_other')) and str(fine_info['first_conviction_other']).strip():
                        formatted_output += f"  Other: {fine_info['first_conviction_other']}\n"
                
                # Second conviction
                if pd.notna(fine_info.get('second_conviction_minimum_fine')) or pd.notna(fine_info.get('second_conviction_maximum_fine')):
                    formatted_output += "Second Conviction:\n"
                    if pd.notna(fine_info.get('second_conviction_minimum_fine')) and pd.notna(fine_info.get('second_conviction_maximum_fine')):
                        formatted_output += f"  Fine: Rs. {fine_info['second_conviction_minimum_fine']:,.0f} - Rs. {fine_info['second_conviction_maximum_fine']:,.0f}\n"
                    if pd.notna(fine_info.get('second_conviction_imprisionment_months')):
                        formatted_output += f"  Imprisonment: {fine_info['second_conviction_imprisionment_months']} months\n"
                    if pd.notna(fine_info.get('second_conviction_other')) and str(fine_info['second_conviction_other']).strip():
                        formatted_output += f"  Other: {fine_info['second_conviction_other']}\n"
                
                # Third conviction
                if pd.notna(fine_info.get('third_conviction_minimum_fine')) or pd.notna(fine_info.get('third_conviction_maximum_fine')):
                    formatted_output += "Third Conviction:\n"
                    if pd.notna(fine_info.get('third_conviction_minimum_fine')) and pd.notna(fine_info.get('third_conviction_maximum_fine')):
                        formatted_output += f"  Fine: Rs. {fine_info['third_conviction_minimum_fine']:,.0f} - Rs. {fine_info['third_conviction_maximum_fine']:,.0f}\n"
                    if pd.notna(fine_info.get('third_conviction_imprisionments')):
                        formatted_output += f"  Imprisonment: {fine_info['third_conviction_imprisionments']} months\n"
                    if pd.notna(fine_info.get('third_conviction_other')) and str(fine_info['third_conviction_other']).strip():
                        formatted_output += f"  Other: {fine_info['third_conviction_other']}\n"
            
            formatted_output += "-" * 80 + "\n\n"
        
        return formatted_output
    
    def search_structured(self, query, top_k=1):
        """Search for relevant sections and return structured data"""
        # Get search results
        results = self.search(query, top_k)
        
        if not results:
            return {
                "status": "success",
                "message": "No relevant sections found",
                "data": None
            }
        
        # Process the first result
        result = results[0]
        
        # Get fine information
        fine_info = self._get_fine_info(result['c2'], result['c3'], result['c4'], result['c5'])
        
        # Structure the response
        structured_result = {
            "similarity_score": round(result['similarity_score'], 3),
            "section_info": {
                "section_title": result['c1'] if result['c1'] else None,
                "section_number": result['c2'] if result['c2'] else None,
                "subsection": result['c3'] if result['c3'] else None,
                "paragraph": result['c4'] if result['c4'] else None,
                "subparagraph": result['c5'] if result['c5'] else None,
                "content": result['c6'] if result['c6'] else None
            },
            "penalty_info": self._format_penalty_info(fine_info) if fine_info else None
        }
        
        return {
            "status": "success",
            "message": "Search completed successfully",
            "data": structured_result
        }
    
    def _format_penalty_info(self, fine_info):
        """Format penalty information into structured format"""
        penalty_data = {}
        
        # First conviction
        first_conviction = {}
        if pd.notna(fine_info.get('first_conviction_minimum_fine')) and pd.notna(fine_info.get('first_conviction_maximum_fine')):
            first_conviction["fine_range"] = {
                "minimum": int(fine_info['first_conviction_minimum_fine']),
                "maximum": int(fine_info['first_conviction_maximum_fine']),
                "currency": "LKR"
            }
        if pd.notna(fine_info.get('first_conviction_imprisionment_months')):
            first_conviction["imprisonment_months"] = int(fine_info['first_conviction_imprisionment_months'])
        if pd.notna(fine_info.get('first_conviction_other')) and str(fine_info['first_conviction_other']).strip():
            first_conviction["other_penalties"] = str(fine_info['first_conviction_other'])
        
        if first_conviction:
            penalty_data["first_conviction"] = first_conviction
        
        # Second conviction
        second_conviction = {}
        if pd.notna(fine_info.get('second_conviction_minimum_fine')) and pd.notna(fine_info.get('second_conviction_maximum_fine')):
            second_conviction["fine_range"] = {
                "minimum": int(fine_info['second_conviction_minimum_fine']),
                "maximum": int(fine_info['second_conviction_maximum_fine']),
                "currency": "LKR"
            }
        if pd.notna(fine_info.get('second_conviction_imprisionment_months')):
            second_conviction["imprisonment_months"] = int(fine_info['second_conviction_imprisionment_months'])
        if pd.notna(fine_info.get('second_conviction_other')) and str(fine_info['second_conviction_other']).strip():
            second_conviction["other_penalties"] = str(fine_info['second_conviction_other'])
        
        if second_conviction:
            penalty_data["second_conviction"] = second_conviction
        
        # Third conviction
        third_conviction = {}
        if pd.notna(fine_info.get('third_conviction_minimum_fine')) and pd.notna(fine_info.get('third_conviction_maximum_fine')):
            third_conviction["fine_range"] = {
                "minimum": int(fine_info['third_conviction_minimum_fine']),
                "maximum": int(fine_info['third_conviction_maximum_fine']),
                "currency": "LKR"
            }
        if pd.notna(fine_info.get('third_conviction_imprisionments')):
            third_conviction["imprisonment_months"] = int(fine_info['third_conviction_imprisionments'])
        if pd.notna(fine_info.get('third_conviction_other')) and str(fine_info['third_conviction_other']).strip():
            third_conviction["other_penalties"] = str(fine_info['third_conviction_other'])
        
        if third_conviction:
            penalty_data["third_conviction"] = third_conviction
        
        return penalty_data if penalty_data else None

# Initialize RAG system globally
try:
    rag_system = MotorTrafficActRAG()
    print("Motor Traffic Act RAG System initialized successfully!")
except Exception as e:
    print(f"Error initializing RAG system: {e}")
    rag_system = None

@app.route('/api/search', methods=['POST'])
def search_traffic_act():
    """API endpoint to search Motor Traffic Act"""
    try:
        # Check if RAG system is initialized
        if rag_system is None:
            return jsonify({
                "status": "error",
                "message": "RAG system not initialized. Please check embeddings and fine ranges files.",
                "data": None
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'query' field in request body",
                "data": None
            }), 400
        
        query = data['query'].strip()
        
        if not query:
            return jsonify({
                "status": "error",
                "message": "Query cannot be empty",
                "data": None
            }), 400
        
        # Perform search
        result = rag_system.search_structured(query)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "data": None
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Motor Traffic Act RAG API is running",
        "data": {
            "rag_system_status": "initialized" if rag_system else "not_initialized"
        }
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "Motor Traffic Act RAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/search": "Search Motor Traffic Act sections",
            "GET /api/health": "Health check",
            "GET /": "API documentation"
        },
        "usage": {
            "search_endpoint": {
                "method": "POST",
                "url": "/api/search",
                "content_type": "application/json",
                "body": {
                    "query": "your search query here"
                }
            }
        }
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
