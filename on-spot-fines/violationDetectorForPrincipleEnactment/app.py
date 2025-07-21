import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FinePredictor:
    def __init__(self):
        self.model = SentenceTransformer('offence_model')
        self.embeddings = np.load("offence_embeddings.npy")
        metadata = joblib.load("metadata.pkl")
        self.offences = metadata['offences']
        self.fines = metadata['fines']
        self.clean_offences = metadata['clean_offences']
        
    def predict(self, user_input, threshold=0.5, top_n=3):
        # Preprocess input
        clean_input = user_input.lower().strip()
        
        # Encode input
        input_embedding = self.model.encode([clean_input])
        
        # Calculate similarities
        similarities = cosine_similarity(input_embedding, self.embeddings)[0]
        
        # Get top matches
        results = []
        for idx in np.argsort(similarities)[::-1]:
            if similarities[idx] < threshold:
                break
            results.append({
                'offence': self.offences[idx],
                'fine': self.fines[idx],
                'similarity': float(similarities[idx])
            })
            if len(results) >= top_n:
                break
        
        return results

if __name__ == "__main__":
    predictor = FinePredictor()
    
    while True:
        user_input = input("\nEnter offence description (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        matches = predictor.predict(user_input)
        
        if matches:
            print("\nPossible matches:")
            for match in matches:
                print(f"- {match['offence']} (Confidence: {match['similarity']:.2f})")
                print(f"  Fine: Rs.{match['fine']}\n")
        else:
            print("No matching offences found.")