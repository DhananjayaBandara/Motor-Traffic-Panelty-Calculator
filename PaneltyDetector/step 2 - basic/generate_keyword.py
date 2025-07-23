import pandas as pd
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy's pre-trained model
nlp = spacy.load('en_core_web_sm')

def extract_keywords_tfidf(sentences, top_n=10):
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top keywords for each sentence
    keywords_list = []
    for row in tfidf_matrix:
        row_data = row.toarray().flatten()
        top_indices = row_data.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices if row_data[i] > 0]
        keywords_list.append(', '.join(keywords))
    
    return keywords_list

# Read the CSV file
df = pd.read_csv('offence.csv')

# Combine 'Offence' and 'Description of offence' columns for keyword extraction
combined_text = df['Offence'].astype(str) + ' ' + df['Description of offence'].astype(str)

# Apply TF-IDF keyword extraction to the combined text
df['Keywords'] = extract_keywords_tfidf(combined_text, top_n=10)

# Display the results in the command line
for index, row in df.iterrows():
    print(f"Index: {row['Index']}, Keywords: {row['Keywords']}")

# Save the updated DataFrame to a new CSV file
df.to_csv('offence_updated.csv', index=False)