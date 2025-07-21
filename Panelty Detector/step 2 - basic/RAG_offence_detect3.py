# RAG_offence_detect_perfect.py
import os
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OffenceDetector:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.df = pd.read_csv("offence_updated.csv")
        
        # Convert dataframe to JSON for full context
        self.offence_json = self.df.to_json(orient='records')
        
    def analyze_scenario(self, scenario):
        # Create the perfect prompt with full offence database
        prompt = f"""
        You are an expert traffic violation analyst. Analyze this scenario:
        "{scenario}"
        
        Using the complete offence database below, identify ALL applicable offences:
        {self.offence_json}
        
        Instructions:
        1. Break the scenario into distinct events
        2. For each event, find the BEST matching offence by Index
        3. Return ONLY comma-separated Index codes (e.g., "o6,o10")
        
        Example 1:
        Scenario: "Transporting chemicals without license while speeding"
        → Offences: o6,o10

        Example 2:
        Scenario: "Parked without safety precautions with loud music and no emission certificate"
        → Offences: o42,o23,o34

        Example 3:
        Scenario: "Riding a motorbike without a helmet and not carrying a driving license"
        → Offences: o21,o8

        Example 4:
        Scenario: "Driving special purpose vehicle without a license, no seatbelt, and carrying excess passengers"
        → Offences: o5,o20,o29

        Example 5:
        Scenario: "Ignoring a traffic warden’s signal, running a red light, and excessive noise from vehicle"
        → Offences: o24,o25,o23


        
        Now analyze this scenario:
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "text"}
        )
        
        return response.choices[0].message.content.strip()

if __name__ == "__main__":
    detector = OffenceDetector()
    
    print("Precise Offence Detection System")
    while True:
        scenario = input("\nEnter scenario (or 'quit' to exit): ")
        if scenario.lower() == "quit":
            break
            
        result = detector.analyze_scenario(scenario)
        print("\nLLM Determination:", result)