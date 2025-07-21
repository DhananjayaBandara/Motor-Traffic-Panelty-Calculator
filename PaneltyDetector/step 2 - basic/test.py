import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy's pre-trained model
nlp = spacy.load('en_core_web_sm')

def extract_keywords(sentence):
    # Process the sentence using spaCy NLP model
    doc = nlp(sentence)
    
    # Filter tokens to get only nouns and adjectives, which are more likely to be important keywords
    keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop]
    
    # If we don't have enough keywords, fall back on a simple frequency count of words
    if len(keywords) == 0:
        keywords = [token.text.lower() for token in doc if not token.is_stop]
    
    # Count keyword frequency
    word_counts = Counter(keywords)
    
    # Get the 3 most common keywords
    most_common_keywords = word_counts.most_common(3)
    
    return [word for word, _ in most_common_keywords]

# Example sentence
sentence = "The quick brown fox jumps over the lazy dog near the river."

# Extract the 3 most valuable keywords
top_keywords = extract_keywords(sentence)

print("Top 3 Keywords:", top_keywords)
