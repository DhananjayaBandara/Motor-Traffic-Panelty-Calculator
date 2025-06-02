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

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    scenario = data.get("scenario", "")
    if not scenario:
        return Response("Missing 'scenario' in request.", status=400)
    result = detector.analyze_scenario(scenario)
    if result.strip().lower() == "none":
        return jsonify([])
    codes = result.replace("[", "").replace("]", "").replace("None", "").strip()
    codes = codes.replace(" ", "")  # Remove spaces between codes
    indices = [idx.strip().lower() for idx in codes.split(",") if idx.strip()]
    matched = fines_df[fines_df["Index_norm"].isin(indices)]
    if matched.empty:
        return jsonify([])
    # Build output as a list of dicts
    output = []
    for _, row in matched.iterrows():
        output.append({
            "Index": row['Index'],
            "Offence": row['Offence'],
            "Fine": row['Fine']
        })
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

# For further improvement, consider:
# - Expanding domain-specific synonyms.
# - Tuning alpha parameter for hybrid retrieval.
# - Exploring more advanced NLP techniques for offence detection.