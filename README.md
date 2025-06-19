# 🧠 NLP Project Collection

This repository contains multiple Jupyter notebooks that demonstrate core concepts and implementations in Natural Language Processing (NLP), including data preprocessing, exploratory analysis, sentiment classification, topic modeling, and text generation.

## 📂 Contents

### 📘 NLP_Data_Extraction_Cleaning.ipynb
- Focuses on loading and cleaning raw textual data.
- Techniques include:
  - Lowercasing
  - Punctuation and special character removal
  - Tokenization
  - Stopword removal
  - Lemmatization/Stemming
- Output: Cleaned and normalized text data for further analysis.

---

### 📘 NLP_EDA.ipynb
- Performs Exploratory Data Analysis (EDA) on the cleaned text.
- Includes:
  - Word frequency distribution
  - Bigram and trigram analysis
  - Word clouds and bar graphs
  - Visual insights into the corpus
- Purpose: Understand textual structure and important terms.

---

### 📘 NLP_Markov_Text_Generator.ipynb
- Builds a Markov Chain model for text generation.
- Learns word transition probabilities from input text.
- Generates new text sequences resembling the original style.
- Useful for creative or mimicry-based text applications.

---

### 📘 NLP_Sentiment_Analysis.ipynb
- Analyzes sentiment polarity of text data.
- Methods may include:
  - VADER
  - TextBlob
  - Machine learning classifiers
- Output: Classification of text into Positive, Negative, or Neutral sentiments.

---

### 📘 NLP_Topic_Modeling.ipynb
- Identifies underlying topics in a collection of documents.
- Implements:
  - Latent Dirichlet Allocation (LDA)
  - Non-negative Matrix Factorization (NMF)
- Outputs:
  - Top words per topic
  - Distribution of topics across documents

---

## 🔧 Requirements

To run these notebooks, make sure the following Python libraries are installed:

- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `matplotlib` – Plotting and visualization
- `seaborn` – Statistical data visualization
- `wordcloud` – Word cloud generation
- `nltk` – Natural Language Toolkit for preprocessing
- `spacy` – Industrial-strength NLP (tokenization, lemmatization)
- `scikit-learn` – Machine learning models and utilities
- `gensim` – Topic modeling (LDA, Word2Vec)
- `textblob` – Sentiment analysis
- `vaderSentiment` – Rule-based sentiment analysis
- `re` – Regular expressions (standard library)
- `string` – String operations (standard library)
- `collections` – Frequency counting and Markov chains

### 📦 Install via pip:

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk spacy scikit-learn gensim textblob vaderSentiment


