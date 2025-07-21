import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from dotenv import load_dotenv
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from domain_synonyms import DOMAIN_SYNONYMS
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Load environment variables
load_dotenv()
nltk.download(['punkt', 'wordnet'], quiet=True)

def normalize_and_lemmatize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)  # Preserves word meaning better than stemming

def expand_with_domain_synonyms(tokens):
    expanded = set(tokens)
    for token in tokens:
        if token in DOMAIN_SYNONYMS:
            expanded.update(DOMAIN_SYNONYMS[token])
    return expanded

def expand_with_synonyms(text):
    """Expand query with WordNet and domain synonyms."""
    tokens = nltk.word_tokenize(text)
    expanded = set(tokens)
    # WordNet expansion
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    # Domain-specific expansion
    expanded = expand_with_domain_synonyms(expanded)
    return ' '.join(expanded)

class OffenceDetector:
    def __init__(self, ngram_range=(1,3), max_features=5000):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.df = pd.read_csv("offence_updated.csv")
        self.texts = (
            self.df["Offence"].fillna('') + " " +
            self.df["Description of offence"].fillna('') + " " +
            (self.df["Keywords"].fillna('') + " ") * 3
        )
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            max_features=max_features,
            preprocessor=normalize_and_lemmatize
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)
        # Precompute OpenAI embeddings for all offences
        self.embeddings = self._compute_embeddings(self.texts.tolist())

    def _compute_embeddings(self, texts):
        # Batch embedding for efficiency (OpenAI API supports batching)
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            embeddings.extend([d.embedding for d in resp.data])
        return embeddings

    def _get_query_embedding(self, query):
        resp = self.client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        return resp.data[0].embedding

    def find_relevant_offences(self, query, top_k=10, alpha=0.5):
        """
        Hybrid retrieval: combine TF-IDF and embedding similarity.
        alpha: weight for TF-IDF (0.0 = only embedding, 1.0 = only TF-IDF)
        """
        norm_query = normalize_and_lemmatize(query)
        expanded_query = expand_with_synonyms(norm_query)
        # TF-IDF similarity
        query_vec = self.tfidf.transform([expanded_query])
        tfidf_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        # Embedding similarity
        query_emb = self._get_query_embedding(expanded_query)
        import numpy as np
        emb_matrix = np.array(self.embeddings)
        emb_sim = cosine_similarity([query_emb], emb_matrix).flatten()
        # Hybrid score
        hybrid_sim = alpha * tfidf_sim + (1 - alpha) * emb_sim
        top_indices = hybrid_sim.argsort()[-top_k:][::-1]
        result_df = self.df.iloc[top_indices].copy()
        result_df['tfidf_similarity'] = tfidf_sim[top_indices]
        result_df['embedding_similarity'] = emb_sim[top_indices]
        result_df['hybrid_similarity'] = hybrid_sim[top_indices]
        return result_df

    def analyze_scenario(self, scenario):
        relevant_df = self.find_relevant_offences(scenario, top_k=5)
        if relevant_df.empty:
            return "None"
        offences_list = "\n".join(
            f"Index: {row['Index']}\nOffence: {row['Offence']}\n"
            f"Description: {row['Description of offence']}\n"
            f"Keywords: {row['Keywords']}\n{'-'*40}" 
            for _, row in relevant_df.iterrows()
        )
        prompt = f"""
        Analyze this scenario: "{scenario}"
        
        Identify ALL applicable offences from this list:
        {offences_list}
        
        Decision Rules:
        1. Match scenario details to Keywords/Description
        2. Return comma-separated Index codes (e.g., "o6,o7")
        3. If no match, say "None"
        
        Question: Which Index codes apply? 
        Answer ONLY in format: [codes] or [None]
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()

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
            print(f"Embeddings cache file '{self.embeddings_cache_file}' not found! Court fine detection disabled.")
            return None, None
        
        try:
            print("Loading embeddings from cache...")
            with open(self.embeddings_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            embeddings = np.array(cached_data['embeddings'])
            processed_data = cached_data['processed_data']
            
            print(f"Loaded {len(processed_data)} sections with embeddings!")
            return embeddings, processed_data
            
        except Exception as e:
            print(f"Error loading embeddings cache: {e}")
            return None, None
    
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
                print(f"Fine ranges file '{self.fine_ranges_file}' not found.")
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
    
    def search_structured(self, query, top_k=1):
        """Search for relevant sections and return structured data"""
        if self.embeddings is None or self.processed_data is None:
            return {
                "status": "error",
                "message": "Court fine system not available - embeddings not loaded",
                "data": None
            }
        
        # Get embedding for the query
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return {
                "status": "error",
                "message": "Failed to generate query embedding",
                "data": None
            }
        
        # Calculate cosine similarity
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar items
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.processed_data[idx]
            
            # Get fine information
            fine_info = self._get_fine_info(item['c2'], item['c3'], item['c4'], item['c5'])
            
            # Format penalty info
            penalty_info = self._format_penalty_info(fine_info) if fine_info else None
            
            results.append({
                'similarity_score': float(similarities[idx]),
                'section_info': {
                    'section_title': item['c1'] if item['c1'] else None,
                    'section_number': item['c2'] if item['c2'] else None,
                    'subsection': item['c3'] if item['c3'] else None,
                    'paragraph': item['c4'] if item['c4'] else None,
                    'subparagraph': item['c5'] if item['c5'] else None,
                    'content': item['c6'] if item['c6'] else None
                },
                'penalty_info': penalty_info
            })
        
        return {
            "status": "success",
            "message": "Search completed successfully",
            "data": results
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

# --- Flask API setup ---
app = Flask(__name__)
CORS(app)

# Load fines_df and normalize index once at startup
fines_df = pd.read_csv("IndexFines.csv")
fines_df["Index_norm"] = fines_df["Index"].astype(str).str.strip().str.lower()

# Initialize detector once at startup
detector = OffenceDetector(
    ngram_range=(1,3),
    max_features=7000
)

# Initialize RAG system for court fines
try:
    court_rag_system = MotorTrafficActRAG()
    print("Motor Traffic Act RAG System initialized successfully!")
except Exception as e:
    print(f"Error initializing RAG system: {e}")
    court_rag_system = None

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    scenario = data.get("scenario", "")
    fine_type = data.get("fineType", "onspot")
    
    if not scenario:
        return Response("Missing 'scenario' in request.", status=400)
    
    # Handle court fine detection
    if fine_type == "court":
        if court_rag_system is None:
            return jsonify([{
                "Offence": "Court fine system unavailable",
                "Fine": "System not initialized - missing embeddings"
            }])
        
        court_result = court_rag_system.search_structured(scenario)
        
        if court_result["status"] != "success" or not court_result["data"]:
            return jsonify([{
                "Offence": "No court fine information found",
                "Fine": "Unable to determine court penalties"
            }])
        
        # Format court results for frontend
        court_data = court_result["data"][0]
        output = []
        
        section_info = court_data.get("section_info", {})
        penalty_info = court_data.get("penalty_info", {})
        
        offence_title = section_info.get("section_title", "Traffic Violation")
        offence_description = section_info.get("content", "No description available")
        
        # Create a single comprehensive output with section details and all penalties
        section_details = {
            "SectionNumber": section_info.get("section_number") or "-",
            "Subsection": section_info.get("subsection") or "-", 
            "Paragraph": section_info.get("paragraph") or "-",
            "Subparagraph": section_info.get("subparagraph") or "-"
        }
        
        if penalty_info:
            penalties = []
            for conviction_type, penalty_data in penalty_info.items():
                # More descriptive conviction labels
                conviction_labels = {
                    "first_conviction": "First Conviction",
                    "second_conviction": "Second Conviction", 
                    "third_conviction": "Third Conviction"
                }
                conviction_label = conviction_labels.get(conviction_type, conviction_type.replace("_", " ").title())
                
                penalty_details = {}
                
                # Fine amount
                if "fine_range" in penalty_data:
                    fine_range = penalty_data["fine_range"]
                    penalty_details["fine_amount"] = f"Rs. {fine_range['minimum']:,} - Rs. {fine_range['maximum']:,}"
                else:
                    penalty_details["fine_amount"] = "Fine amount not specified"
                
                # Additional penalties
                additional_penalties = []
                if "imprisonment_months" in penalty_data:
                    months = penalty_data['imprisonment_months']
                    if months == 1:
                        additional_penalties.append("1 month imprisonment")
                    else:
                        additional_penalties.append(f"{months} months imprisonment")
                
                if "other_penalties" in penalty_data:
                    other = penalty_data["other_penalties"].strip()
                    if other:
                        additional_penalties.append(other)
                
                penalty_details["additional_penalties"] = additional_penalties
                
                penalties.append({
                    "conviction_type": conviction_label,
                    "penalty_details": penalty_details
                })
            
            output.append({
                "Offence": offence_title,
                "Description": offence_description,
                "SectionDetails": section_details,
                "Penalties": penalties
            })
        else:
            output.append({
                "Offence": offence_title,
                "Description": offence_description,
                "SectionDetails": section_details,
                "Penalties": [{
                    "conviction_type": "General",
                    "penalty_details": {
                        "fine_amount": "Court fine information not available",
                        "additional_penalties": []
                    }
                }]
            })
        
        return jsonify(output)
    
    # Handle on-spot fine detection (existing logic)
    result = detector.analyze_scenario(scenario)
    if result.strip().lower() == "none":
        return jsonify([])
    codes = result.replace("[", "").replace("]", "").replace("None", "").strip()
    codes = codes.replace(" ", "")
    indices = [idx.strip().lower() for idx in codes.split(",") if idx.strip()]
    matched = fines_df[fines_df["Index_norm"].isin(indices)]
    if matched.empty:
        return jsonify([])
    
    output = []
    for _, row in matched.iterrows():
        # Format fine amount properly
        fine_amount = f"Rs. {row['Fine']:,}" if pd.notna(row['Fine']) else "Fine amount not specified"
        
        output.append({
            "Index": row['Index'],
            "Offence": row['Offence'],
            "Fine": fine_amount,
            "Description": row.get('Description', 'No description available')
        })
    return jsonify(output)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Traffic Fine Detection API is running",
        "services": {
            "onspot_detection": "available",
            "court_detection": "available" if court_rag_system else "unavailable"
        }
    }), 200

if __name__ == "__main__":
    app.run(debug=True)

# For further improvement, consider:
# - Expanding domain-specific synonyms.
# - Tuning alpha parameter for hybrid retrieval.
# - Exploring more advanced NLP techniques for offence detection.