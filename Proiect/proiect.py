import os
import numpy as np
import re
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Setam directorul pentru cache-ul nltk
cache_dir = './data_cache'
nltk_data_path = os.path.join(cache_dir, 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)

# Initializam stemmer-ul si setul de stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Functie de preprocesare pentru stemming, stop words, caractere speciale
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # eliminam numerele
    text = re.sub(r'\W+', ' ', text)  # eliminam caractere speciale
    words = text.split()
    return [stemmer.stem(word) for word in words if word not in stop_words]

# Incarcam setul de date si il impartim in antrenare si testare
categories = ['alt.atheism', 'comp.graphics', 'sci.med']
newsgroups_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), data_home=cache_dir)
X_raw = newsgroups_data.data
y_raw = newsgroups_data.target
target_names = newsgroups_data.target_names
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Preprocesam textul pentru setul de antrenare si testare
X_train_processed = [" ".join(preprocess(doc)) for doc in X_train_raw]
X_test_processed = [" ".join(preprocess(doc)) for doc in X_test_raw]

# Vectorizare TF-IDF cu bigrame si limitare la 5000 de caracteristici
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X_train = vectorizer.fit_transform(X_train_processed).toarray()
X_test = vectorizer.transform(X_test_processed).toarray()

# Implementare manuala Naive Bayes
class_counts = defaultdict(int)
word_counts = defaultdict(lambda: np.zeros(X_train.shape[1]))
total_words_in_class = defaultdict(int)

# Calculam frecventele pe fiecare clasa si cuvant
for i in range(len(X_train)):
    label = y_train[i]
    class_counts[label] += 1
    word_counts[label] += X_train[i]
    total_words_in_class[label] += sum(X_train[i])

# Probabilitatile priore pentru clase
total_docs = len(X_train)
class_probs = {label: count / total_docs for label, count in class_counts.items()}

# Netezirea Laplace si probabilitatile conditionale
alpha = 0.1
vocab_size = X_train.shape[1]
word_probs = defaultdict(lambda: np.zeros(vocab_size))

for label in class_counts:
    word_probs[label] = (word_counts[label] + alpha) / (total_words_in_class[label] + alpha * vocab_size)

# Functia de clasificare
def classify_document(doc_vector, class_probs, word_probs):
    scores = {}
    for label in class_probs:
        score = np.log(class_probs[label])
        score += np.sum(doc_vector * np.log(word_probs[label] + 1e-10))
        scores[label] = score
    return max(scores, key=scores.get)

# Testam modelul si calculam acuratetea
correct_predictions = 0
for i in range(len(X_test)):
    predicted_label = classify_document(X_test[i], class_probs, word_probs)
    if predicted_label == y_test[i]:
        correct_predictions += 1

# Afisam acuratetea
accuracy = correct_predictions / len(y_test)
print(f"Acuratetea modelului: {accuracy:.2%}")
print(f"Predictii corecte: {correct_predictions}")
print(f"Predictii incorecte: {len(y_test) - correct_predictions}")
