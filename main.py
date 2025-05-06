import csv
import re
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def load_emails_labels(file_path):
    Emails = []
    Labels = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            if len(row) >= 2:
                Labels.append(row[0])
                Emails.append(row[1])
            if len(Emails) == 4:
                break

    return Emails, Labels

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens

def generate_bigrams(words):
    bigrams = []
    for i in range(0, len(words) - 1, 2):
        bigram = ' '.join(words[i:i+2])
        bigrams.append(bigram)
    return bigrams

def build_vocab_and_ngrams(emails):
    all_tokens = []
    token_set = set()

    for email in emails:
        tokens = preprocess(email)
        ngrams = generate_bigrams(tokens)
        all_tokens.append(ngrams)
        for term in ngrams:
            token_set.add(term)

    unique_tokens = list(token_set)

    return all_tokens, unique_tokens

def compute_tf_vector(tokens, unique_tokens):
    tf = [0] * len(unique_tokens)
    total_tokens = len(tokens)

    for token in tokens:
        for i in range(len(unique_tokens)):
            if token == unique_tokens[i]:
                tf[i] += 1

    for i in range(len(tf)):
        tf[i] = tf[i] / total_tokens if total_tokens > 0 else 0

    return tf


def compute_idf(tf_vectors, unique_tokens):
    num_docs = len(tf_vectors)
    idf_counts = [0] * len(unique_tokens)

    for tf in tf_vectors:
        for i, count in enumerate(tf):
            if count > 0:
                idf_counts[i] += 1

    idf = []
    for count in idf_counts:
        if count > 0:
            idf.append(math.log(num_docs / count))
        else:
            idf.append(0)

    return idf

def print_feature_vectors_with_labels(emails, labels):
    all_ngrams, unique_tokens = build_vocab_and_ngrams(emails)
    tf_vectors = [compute_tf_vector(ngrams, unique_tokens) for ngrams in all_ngrams]
    idf = compute_idf(tf_vectors, unique_tokens)

    for i in range(len(emails)):
        tfidf_vector = [tf_vectors[i][j] * idf[j] for j in range(len(unique_tokens))]
        print(f"\nEmail {i+1} - Label: {labels[i]}")
        for j, value in enumerate(tfidf_vector):
            print(f"{unique_tokens[j]}: {value:.4f}")

def main():
    file_path = 'D:\\downloads\\spam.csv'
    emails, labels = load_emails_labels(file_path)
    print_feature_vectors_with_labels(emails, labels)

main()
