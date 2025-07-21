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
from nltk.tokenize import sent_tokenize


# Load environment variables
load_dotenv()

def normalize_and_lemmatize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)  # Preserves word meaning better than stemming

# Example domain-specific synonyms (expand as needed)
DOMAIN_SYNONYMS = {
    "signal": ["traffic light", "red light", "stoplight", "traffic signal", "green light", "yellow light", "amber light", "signal post"],
    "license": ["licence", "driving license", "permit", "driver's license", "dl", "learner's license", "provisional license"],
    "helmet": ["headgear", "protective helmet", "bike helmet", "motorcycle helmet", "crash helmet", "safety helmet"],
    "speed": ["velocity", "overspeeding", "fast", "speeding", "over speed", "high speed", "speed limit", "exceeding speed"],
    "seatbelt": ["seat belt", "safety belt", "restraint", "belt", "car belt"],
    "insurance": ["vehicle insurance", "motor insurance", "third party insurance", "insurance certificate", "policy", "insurance paper"],
    "registration": ["rc", "registration certificate", "vehicle registration", "rc book", "rc card"],
    "pollution": ["pollution certificate", "puc", "pollution under control", "emission certificate", "pollution paper"],
    "drunk": ["intoxicated", "alcohol", "drunken", "drink and drive", "under influence", "driving under influence", "dui"],
    "mobile": ["cellphone", "cell phone", "phone", "smartphone", "using phone", "mobile phone", "talking on phone"],
    "parking": ["no parking", "illegal parking", "parked", "unauthorized parking", "wrong parking", "parking violation"],
    "red light": ["signal jump", "jumped signal", "signal violation", "crossed red", "ran red light"],
    "overload": ["overloading", "excess load", "over capacity", "overweight", "extra passenger", "overcrowded"],
    "horn": ["honking", "sound horn", "unnecessary horn", "loud horn", "pressure horn", "horn violation"],
    "oneway": ["one way", "wrong way", "opposite direction", "against traffic", "wrong side"],
    "u-turn": ["illegal uturn", "prohibited uturn", "no uturn", "u turn", "u-turn violation"],
    "stop": ["halt", "stopped", "not stopped", "failure to stop", "did not stop"],
    "documents": ["papers", "vehicle papers", "documents missing", "document", "paperwork"],
    "number plate": ["license plate", "registration plate", "number board", "plate", "fancy number", "illegal plate"],
    "bribe": ["corruption", "illegal payment", "offering bribe", "accepting bribe", "bribery"],
    "passenger": ["rider", "traveller", "commuter", "occupant", "person on board"],
    "pedestrian": ["walker", "foot passenger", "person crossing", "jaywalker"],
    "crosswalk": ["zebra crossing", "pedestrian crossing", "crossing line"],
    "lane": ["lane discipline", "lane violation", "wrong lane", "lane cutting", "lane change"],
    "indicator": ["turn signal", "blinker", "indicator light", "signal light"],
    "headlight": ["head lamp", "front light", "high beam", "dipped beam", "head lamp violation"],
    "pollution": ["emission", "smoke", "polluting", "pollution norm", "pollution violation"],
    "fine": ["penalty", "challan", "ticket", "monetary penalty", "spot fine"],
    "stop line": ["signal line", "white line", "crossed stop line", "line violation"],
    "helmetless": ["without helmet", "no helmet", "not wearing helmet"],
    "seatbeltless": ["without seatbelt", "no seatbelt", "not wearing seatbelt"],
    "overspeed": ["over speed", "speeding", "exceeding speed limit"],
    "wrong side": ["opposite direction", "against traffic", "wrong way"],
    "dangerous driving": ["rash driving", "reckless driving", "careless driving", "negligent driving"],
    "hit and run": ["fleeing scene", "ran away", "did not stop after accident", "left scene"],
    "minor": ["underage", "below 18", "child driver", "juvenile"],
    "trip sheet": ["permit paper", "trip permit", "trip document"],
    "tax": ["road tax", "vehicle tax", "tax paid", "tax token"],
    "pollution check": ["emission test", "pollution test", "puc check"],
    "driving test": ["license test", "dl test", "driving examination"],
    "alcohol": ["liquor", "booze", "spirit", "ethanol", "drunk"],
    "road": ["street", "highway", "lane", "avenue", "boulevard"],
    # Add more as needed for your dataset
}

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

if __name__ == "__main__":
    detector = OffenceDetector(
        ngram_range=(1,3),  # Tune as needed
        max_features=7000   # Tune as needed
    )
    print("Offence Detection System")
    while True:
        scenario = input("\nEnter scenario (or 'quit' to exit): ")
        if scenario.lower() == "quit":
            break
        relevant_df = detector.find_relevant_offences(scenario)
        result = detector.analyze_scenario(scenario)
        print("\nPotential Matches:")
        if not relevant_df.empty:
            for _, row in relevant_df.iterrows():
                print(
                    f"{row['Index']}: {row['Offence'][:70]}... "
                    f"(hybrid: {row['hybrid_similarity']:.3f}, "
                    f"tfidf: {row['tfidf_similarity']:.3f}, "
                    f"emb: {row['embedding_similarity']:.3f})"
                )
        print("\nLLM Determination:", result)

# For further improvement, consider:
# - Expanding domain-specific synonyms.
# - Tuning alpha parameter for hybrid retrieval.
# - Exploring more advanced NLP techniques for offence detection.