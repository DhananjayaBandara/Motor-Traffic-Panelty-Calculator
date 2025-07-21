import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env (must contain OPENAI_API_KEY)
load_dotenv()

class RAGMatcher:
    def __init__(self, csv_path="all_parts_combined.csv"):
        # Load the knowledge base
        self.df = pd.read_csv(csv_path)
        # Combine c1 and c6 for semantic search
        self.df["combined"] = self.df["c1"].astype(str) + " " + self.df["c6"].astype(str)
        # Prepare TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined"])
        # OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def find_top_matches(self, scenario, top_n=5):
        # Vectorize the input scenario
        scenario_vec = self.vectorizer.transform([scenario])
        # Compute cosine similarity
        similarities = cosine_similarity(scenario_vec, self.tfidf_matrix)[0]
        # Get indices of top N matches
        top_indices = similarities.argsort()[-top_n:][::-1]
        # Return DataFrame with scores
        top_df = self.df.iloc[top_indices].copy()
        top_df["score"] = similarities[top_indices]
        return top_df

    def rerank_with_llm(self, scenario, candidates):
        # Prepare prompt for LLM
        prompt = (
            "Given the following scenario, select the most relevant part from the list. "
            "Return ONLY the part_name. If none are relevant, return 'None'.\n\n"
            f"Scenario: {scenario}\n\n"
            "Candidates:\n"
        )
        for idx, row in candidates.iterrows():
            prompt += (
                f"part_name: {row['part_name']}\n"
                f"c1: {row['c1']}\n"
                f"c6: {row['c6']}\n"
                f"c2: {row['c2']}\n"
                f"c3: {row['c3']}\n"
                f"c4: {row['c4']}\n"
                f"c5: {row['c5']}\n\n"
            )
        prompt += "Answer:"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )
        answer = response.choices[0].message.content.strip()
        return answer

    def find_best_match(self, scenario):
        candidates = self.find_top_matches(scenario, top_n=5)
        best_part_name = self.rerank_with_llm(scenario, candidates)
        if best_part_name.lower() == "none":
            return None, candidates
        # Find the row with the matching part_name
        row = candidates[candidates["part_name"] == best_part_name]
        if row.empty:
            # fallback to top TF-IDF match
            row = candidates.iloc[[0]]
        row = row.iloc[0]
        return {
            "part_name": row["part_name"],
            "c1": row["c1"],
            "c2": row["c2"],
            "c3": row["c3"],
            "c4": row["c4"],
            "c5": row["c5"],
            "c6": row["c6"]
        }, candidates

if __name__ == "__main__":
    matcher = RAGMatcher()
    print("RAG Knowledge Base Matcher (OpenAI-powered)")
    while True:
        scenario = input("\nEnter scenario (or 'quit' to exit): ")
        if scenario.lower() == "quit":
            break
        result, candidates = matcher.find_best_match(scenario)
        print("\nTop 5 Matches (TF-IDF):")
        for _, row in candidates.iterrows():
            print(
                f"part_name: {row['part_name']}\n"
                f"c1: {row['c1']}\n"
                f"c2: {row['c2']}\n"
                f"c3: {row['c3']}\n"
                f"c4: {row['c4']}\n"
                f"c5: {row['c5']}\n"
                f"c6: {row['c6']}\n"
                f"Score: {row['score']:.3f}\n"
                "---------------------------"
            )
        if result is None:
            print("\nNo relevant match found.")
        else:
            print("\nBest Match:")
            for k, v in result.items():
                print(f"{k}: {v}")
