# 🚦 Hybrid RAG Traffic Offence Detector

This project implements a **Hybrid Retrieval-Augmented Generation (RAG)** system to detect relevant **traffic offences** based on user-described scenarios. It combines traditional **NLP (TF-IDF)** with modern **LLM-based reasoning (OpenAI Embeddings + GPT-4)** to retrieve and reason about potential offences from a structured offence dataset.

---

## 📌 Purpose

To assist in **identifying applicable traffic law violations** from natural language descriptions by using a hybrid approach that blends statistical and semantic retrieval with LLM-based legal reasoning.

---

## ⚙️ Key Features

- ✅ Text normalization and synonym expansion for better recall.
- ✅ TF-IDF and OpenAI Embedding-based similarity search.
- ✅ Hybrid scoring for robust retrieval.
- ✅ GPT-4-based reasoning to finalize applicable offences.
- ✅ Interactive command-line interface.

---

## 🧠 Functionality Overview

### 1. Text Preprocessing
- **`normalize_and_lemmatize(text)`**: Cleans, tokenizes, and lemmatizes input.
- **`DOMAIN_SYNONYMS`**: Custom dictionary for domain-specific synonym expansion.
- **`expand_with_domain_synonyms()` / `expand_with_synonyms()`**: Enhance the input query using WordNet and domain knowledge.

### 2. OffenceDetector Class

#### Initialization
- Loads offences from `offence_updated.csv`.
- Builds a TF-IDF model with n-grams and lemmatization.
- Precomputes **OpenAI embeddings** for all offences.

#### Core Methods
- **`_compute_embeddings()`**: Batches embedding requests for performance.
- **`_get_query_embedding()`**: Embeds the input query.
- **`find_relevant_offences(query)`**:
  - Applies preprocessing and synonym expansion.
  - Calculates TF-IDF and embedding similarities.
  - Ranks using a **hybrid score**.
- **`analyze_scenario(query)`**:
  - Retrieves top-k offences.
  - Sends context to GPT-4.
  - Returns selected offence indices.

---

## 🔁 Main Workflow

1. **User Input:** Prompt user for a scenario.
2. **Text Processing:** Normalize and expand input.
3. **Retrieval:**
   - Compute TF-IDF and embedding similarity.
   - Combine scores using a hybrid weight.
4. **LLM Reasoning:**
   - Format input and retrieved offences.
   - Send to GPT-4.
   - Return the most applicable offences.
5. **Output:** Print results.

---

## 🛠️ Requirements

- Python 3.8+
- Required packages:
  - `openai`
  - `pandas`
  - `nltk`
  - `scikit-learn`
  - `tqdm`
  - `python-dotenv`

Install dependencies:

```bash
pip install -r requirements.txt
