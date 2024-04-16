import numpy as np
import pandas as pd
import re
import nltk
import chardet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

## 1 | Data Preprocessing ##
"""Prepare the dataset before training"""

# 1.1 Load dataset
dataset = pd.read_csv('Dataset/IMDB.csv', encoding_errors='replace')
print(f"Dataset shape : {dataset.shape}\n")
print(f"Dataset head : \n{dataset.head()}\n")

# 1.2 Output counts
print(f"Dataset output counts:\n{dataset.sentiment.value_counts()}\n")

# 1.3 Encode output column into binary
dataset.sentiment.replace('positive', 1, inplace=True)
dataset.sentiment.replace('negative', 0, inplace=True)
print(f"Dataset head after encoding :\n{dataset.head(10)}\n")

## 2 | Data cleaning ##
"""Clean dataset reviews by removing HTML tags"""

# 2.1 Remove HTML tags
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

dataset.review = dataset.review.apply(clean)
print(f"Review sample after removing HTML tags : \n{dataset.review[0]}\n")

# 2.2 Tokenize text
def tokenize_text(text):
    return word_tokenize(text)

dataset.review = dataset.review.apply(tokenize_text)
print(f"Review sample after tokenization : \n{dataset.review[0]}\n")

# 2.3 Remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

dataset.review = dataset.review.apply(remove_stopwords)
print(f"Review sample after removing stopwords : \n{dataset.review[0]}\n")

## 3 | Model Creation ##
"""Create model to fit it to the data"""

# 3.1 Creating Bag Of Words (BOW)
X = np.array(dataset.iloc[:,0].values)
y = np.array(dataset.sentiment.values)
X = cv.fit_transform(dataset.review).toarray()
print(f"=== Bag of words ===\n")
print(f"BOW X shape : {X.shape}")
print(f"BOW y shape : {y.shape}\n")
