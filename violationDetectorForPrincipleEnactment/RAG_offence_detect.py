# RAG_offence_detect.py
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OffenceDetector:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load offence data
        self.df = pd.read_csv("offence_updated.csv")
        
        # Prepare TF-IDF vectors
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.texts = self.df["Offence"] + " " + self.df["Keywords"]
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

    def find_relevant_offences(self, query, threshold=0.3):
        # Convert query to TF-IDF vector
        query_vec = self.tfidf.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        
        # Get indices of relevant offences
        relevant_indices = [
            i for i, score in enumerate(similarities[0]) 
            if score > threshold
        ]
        
        return self.df.iloc[relevant_indices]

    def analyze_scenario(self, scenario):
        # Step 1: Retrieve relevant offences
        relevant_df = self.find_relevant_offences(scenario)
        
        if relevant_df.empty:
            return "None"
        
        # Step 2: LLM validation
        offences_list = "\n".join(
            f"Index: {row['Index']}\nOffence: {row['Offence']}\nKeywords: {row['Keywords']}" 
            for _, row in relevant_df.iterrows()
        )
        
        prompt = f"""Analyze this scenario and identify matching offences from the list below.
        Return ONLY the relevant Index codes (e.g., o1, o2) as a single comma-separated string.
        If none match, return 'None'.
        
        Scenario: {scenario}
        
        Possible Offences:
        {offences_list}
        
        Answer:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.1,
            max_tokens=50
        )
        
        # Extract and format the response content
        result = response.choices[0].message.content.strip()
        return result.replace("\n", ", ")

if __name__ == "__main__":
    detector = OffenceDetector()
    
    print("Offence Detection System")
    while True:
        scenario = input("\nEnter scenario (or 'quit' to exit): ")
        if scenario.lower() == "quit":
            break
            
        # Retrieve relevant indices
        relevant_df = detector.find_relevant_offences(scenario)
        
        # Get LLM analysis
        result = detector.analyze_scenario(scenario)
        
        # Display results
        print("\nPotential Matches:")
        if not relevant_df.empty:
            for _, row in relevant_df.iterrows():
                print(f"{row['Index']}: {row['Offence'][:70]}...")
                
        print("\nLLM Determination:", result)