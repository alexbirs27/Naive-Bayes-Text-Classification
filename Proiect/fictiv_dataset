import numpy as np
import re
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import nltk
import matplotlib.pyplot as plt

# Descarca resursele necesare de la NLTK
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Set de date fictiv extins
fictiv_dataset = [
    ("There is no evidence of any god", "alt.atheism"),
    ("Religious beliefs are subjective", "alt.atheism"),
    ("Believing in god is a personal choice", "alt.atheism"),
    ("Some people find meaning in religion", "alt.atheism"),
    ("Atheists often question religious practices", "alt.atheism"),
    ("The latest graphics card offers great performance", "comp.graphics"),
    ("Rendering in 3D is computationally expensive", "comp.graphics"),
    ("Vector graphics are resolution-independent", "comp.graphics"),
    ("Graphics software has evolved significantly", "comp.graphics"),
    ("Computer graphics improve gaming experience", "comp.graphics"),
    ("Exercise helps to improve mental health", "sci.med"),
    ("Vaccines have drastically reduced diseases", "sci.med"),
    ("Antibiotics fight bacterial infections", "sci.med"),
    ("Health is affected by diet and exercise", "sci.med"),
    ("Studies show that sleep improves memory", "sci.med"),
    ("OpenGL is a framework for rendering graphics", "comp.graphics"),
    ("Science provides explanations without invoking deities", "alt.atheism"),
    ("Medical advancements save millions of lives", "sci.med")
]

# Extragem textele si etichetele si transformam etichetele in format numeric
texts, labels = zip(*fictiv_dataset)
label_to_index = {label: idx for idx, label in enumerate(set(labels))}
index_to_label = {idx: label for label, idx in label_to_index.items()}
y_fictiv = np.array([label_to_index[label] for label in labels])

# Functia de preprocesare folosind lematizare
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # eliminam numerele
    text = re.sub(r'\W+', ' ', text)  # eliminam caracterele speciale
    words = text.split()
    return [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

# Preprocesam textele
texts_processed = [preprocess(doc) for doc in texts]

# Impartim setul de date fictiv in seturi de antrenament si testare
X_train_raw, X_test_raw, y_train, y_test = train_test_split(texts_processed, y_fictiv, test_size=0.3, random_state=42)

# Calculam frecventele cuvintelor pentru fiecare clasa
class_word_counts = defaultdict(lambda: defaultdict(int))
class_counts = defaultdict(int)

# Construim frecventele pentru setul de antrenament
for words, label in zip(X_train_raw, y_train):
    class_counts[label] += 1
    for word in words:
        class_word_counts[label][word] += 1

# Calculam probabilitatile a priori pentru clase
total_docs = len(X_train_raw)
class_probs = {label: count / total_docs for label, count in class_counts.items()}

# Functia de calcul al probabilitatilor pe baza frecventei
def calculate_class_probabilities(doc, class_probs, class_word_counts, class_counts):
    scores = {}
    for label in class_probs:
        score = class_probs[label]  # probabilitatea initiala a clasei
        total_words_in_class = sum(class_word_counts[label].values())
        for word in doc:
            word_freq = class_word_counts[label].get(word, 0)
            # Calculam probabilitatea fiecarui cuvant in document
            word_probability = (word_freq / total_words_in_class) if word_freq > 0 else 1e-10
            score *= word_probability
        scores[label] = score
    return scores

# Clasificam documentele de testare si calculam acuratetea
correct_predictions = 0
plt.figure(figsize=(14, 10))
for idx, (doc, actual_label) in enumerate(zip(X_test_raw, y_test)):
    scores = calculate_class_probabilities(doc, class_probs, class_word_counts, class_counts)
    predicted_label = max(scores, key=scores.get)  # selectam clasa cu scorul maxim
    if predicted_label == actual_label:
        correct_predictions += 1

    # Normalize scores and plot
    total_score = sum(scores.values())
    labels = [index_to_label[i] for i in sorted(scores)]
    probabilities = [scores[i] / total_score for i in sorted(scores)]
    plt.subplot(len(X_test_raw), 1, idx + 1)
    plt.bar(labels, probabilities, color='blue')
    plt.title(f"Document: {' '.join(doc)}")
    plt.ylabel('Probabilitate')

plt.tight_layout()
plt.show()

# Printeaza acuratetea modelului
accuracy = correct_predictions / len(y_test)
print("Acuratetea modelului este: {:.2%}".format(accuracy))
