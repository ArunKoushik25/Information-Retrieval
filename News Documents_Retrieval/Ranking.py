import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import Levenshtein
import matplotlib.pyplot as plt

# Download the 'words' resource
nltk.download('words')

def preprocess_text(text):
    # Remove HTML tags
    clean_text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters
    clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
    # Convert to lowercase
    clean_text = clean_text.lower()
    return clean_text

def retrieve_posting_list_from_disk(root, term):
    current_node = root
    for char in term:
        current_path = os.path.join(current_node, char)
        if not os.path.exists(current_path):
            return None
        current_node = current_path

    posting_list_file = os.path.join(current_node, "posting_list.json")
    if os.path.exists(posting_list_file):
        with open(posting_list_file, "r") as f:
            return json.load(f)

def load_inverted_index(root, query):
    inverted_index = {}
    
    for term_to_search in query.split():
        posting_list = retrieve_posting_list_from_disk(root, term_to_search)
        if posting_list is not None:
            inverted_index[term_to_search] = posting_list

    return inverted_index

# Calculate TF-IDF scores
def calculate_tfidf(inverted_index):
    doc_term_matrix = defaultdict(lambda: defaultdict(int))
    term_document_count = defaultdict(int)
    total_documents = len(inverted_index)

    for term, postings in inverted_index.items():
        for doc_id, position in postings:
            doc_term_matrix[doc_id][term] = 1  # Using binary term frequency
            term_document_count[term] += 1

    idf = {term: math.log(total_documents / (1 + term_document_count[term])) for term in inverted_index}

    tfidf_matrix = {}
    for doc_id, term_counts in doc_term_matrix.items():
        tfidf_matrix[doc_id] = {term: tf * idf[term] for term, tf in term_counts.items()}

    return tfidf_matrix

# Rank documents based on TF-IDF similarity to the query with feedback
def rank_documents_with_feedback(query, inverted_index, tfidf_matrix, top_k=10):
    query = preprocess_text(query)
    query_vectorizer = TfidfVectorizer()
    query_tfidf = query_vectorizer.fit_transform([query])

    ranked_documents = []

    for doc_id, term_tfidf in list(tfidf_matrix.items())[:top_k]:  # Limit to top_k documents
        doc_tfidf = [term_tfidf.get(term, 0) for term in query_vectorizer.get_feature_names_out()]
        similarity = cosine_similarity(query_tfidf, [doc_tfidf])

        # Load content from the file based on the document ID
        file_path = os.path.join('./Cleaned_Texts', f"{doc_id}.txt")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        print(f"\nDocument ID: {doc_id}, Similarity: {similarity[0, 0]:.4f}")
        print("Content:")
        print(content)

        # Prompt for relevance feedback
        feedback = input("Is this document relevant? (1 for relevant, 0 for non-relevant): ")

        # Append the feedback to the result
        ranked_documents.append({'doc_id': doc_id, 'similarity': similarity[0, 0], 'content': content, 'feedback': int(feedback)})

    # Extract feedback for Precision-Recall calculation
    y_true = np.array([info['feedback'] for info in ranked_documents])
    y_scores = np.array([info['similarity'] for info in ranked_documents])

    # Calculate and print the Precision-Recall curve
    precision_values, recall_values = calculate_pr_curve(y_true, y_scores)
    print("Precision values:", precision_values)
    print("Recall values:", recall_values)

# Calculate and print the Precision-Recall curve
def calculate_pr_curve(y_true, y_scores):
    # Sort predicted probabilities in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Initialize arrays to store precision and recall values
    precision_values = []
    recall_values = []

    # Calculate the total number of positive instances in the dataset
    num_relevant = np.sum(y_true)
    cumulative_relevant = np.cumsum(y_true_sorted)

    # Calculate recall and precision
    recall_values = cumulative_relevant / num_relevant
    precision_values = cumulative_relevant / np.arange(1, len(y_true_sorted) + 1)

    print("Recall values:", recall_values)
    print("Precision values:", precision_values)

    # Interpolate precision values at 11 recall levels
    num_points = 11
    interp_recall_levels = np.linspace(0, 1, num_points)
    interp_precision_levels = np.interp(interp_recall_levels, recall_values[::-1], precision_values[::-1])

    print("Interpolated Precision values:", interp_precision_levels)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, marker='o', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return precision_values, recall_values

# Example usage:
# Get user input for the query
user_query = input("Enter your query: ")

# Load inverted index based on the user query
inverted_index = load_inverted_index('root', user_query)

# Calculate TF-IDF matrix
tfidf_matrix = calculate_tfidf(inverted_index)

# Rank documents based on TF-IDF similarity to the query with feedback
rank_documents_with_feedback(user_query, inverted_index, tfidf_matrix, top_k=10)
