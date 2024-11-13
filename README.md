Naive Bayes Text Classification Projects
These projects demonstrate the application of the Naive Bayes classifier to text classification tasks using different preprocessing techniques. The first project uses stemming to process text data from the 20 Newsgroups dataset to classify text into categories such as alt.atheism, comp.graphics, and sci.med. The second project uses lemmatization on a fictional dataset to classify text snippets by their simulated content type.

Contents
Project Setup
Mathematical Model
Implementation Details
Running the Projects
Evaluation and Results
References
Project Setup
Dependencies
Python 3.x
NumPy
Scikit-Learn
NLTK
Matplotlib
Installation
To install necessary libraries, run:

bash
Copy code
pip install numpy matplotlib scikit-learn nltk
Ensure NLTK datasets are downloaded using:

python
Copy code
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
Mathematical Model
Both projects employ a Multinomial Naive Bayes classifier which bases its calculations on Bayes' Theorem:

𝑃
(
𝐶
𝑘
∣
𝑥
)
=
𝑃
(
𝑥
∣
𝐶
𝑘
)
𝑃
(
𝐶
𝑘
)
𝑃
(
𝑥
)
P(C 
k
​
 ∣x)= 
P(x)
P(x∣C 
k
​
 )P(C 
k
​
 )
​
 

Where:

𝑃
(
𝐶
𝑘
∣
𝑥
)
P(C 
k
​
 ∣x) is the posterior probability of class 
𝐶
𝑘
C 
k
​
  given predictor(s) 
𝑥
x.
𝑃
(
𝑥
∣
𝐶
𝑘
)
P(x∣C 
k
​
 ) is the likelihood which is the probability of predictor given class 
𝐶
𝑘
C 
k
​
 .
𝑃
(
𝐶
𝑘
)
P(C 
k
​
 ) is the prior probability of class 
𝐶
𝑘
C 
k
​
 .
𝑃
(
𝑥
)
P(x) is the prior probability of predictor.
Preprocessing Techniques
Stemming Project: Utilizes PorterStemmer to reduce words to their word stems.
Lemmatization Project: Employs WordNetLemmatizer to reduce words to their lemmatized form based on their parts of speech.
Probability Calculations
Prior Probabilities are computed based on the frequency of each class in the training dataset.
Conditional Probabilities are adjusted using Laplace smoothing to avoid zero probability issues.
Implementation Details
Data Preparation
Both projects preprocess text data:

Stemming: Converts text to lowercase, removes numbers and special characters, and applies stemming.
Lemmatization: Converts text to lowercase, removes numbers and special characters, and applies lemmatization.
Vectorization
Text data is transformed into a vector space model using TF-IDF vectorization, considering the frequencies of terms adjusted by their inverse document frequency.

Classification
Using the precomputed probabilities, the classifier calculates the log probabilities for each class and predicts the class with the highest log probability to mitigate underflow issues.

Running the Projects
To run either project, navigate to the directory containing the script and run:

bash
Copy code
python naive_bayes_classifier.py
Evaluation and Results
The classifier's performance is evaluated based on its accuracy in predicting the correct class for new documents. Results are displayed in the console, including the accuracy and a breakdown of correct and incorrect predictions.

References
Naive Bayes - Scikit-Learn
Naive Bayes Classifiers - Towards Data Science
TF-IDF for Text Mining
Text Preprocessing Techniques
Evaluation Metrics for Classification Models
