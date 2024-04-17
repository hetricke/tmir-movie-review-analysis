import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nltk.download('stopwords')
nltk.download('punkt')

## 1 | Data Preprocessing ##
"""Prepare the dataset before training"""

# 1.1 Load dataset
dataset = pd.read_csv('Dataset/IMDB.csv', encoding_errors='replace', nrows=5)
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

#2.4 Convert review to a numpy array and pad
SET_SIZE = 200
def set_size(word_list):
    if len(word_list) >= SET_SIZE:
        return np.array(word_list[:SET_SIZE])
    
    else:
        word_list += [''] * (SET_SIZE - len(word_list))
        temp_array = np.array(word_list)
        return temp_array

dataset.review = dataset.review.apply(set_size)
print(f"Review sample after converting to numpy array : \n{dataset.review[0]}\n")

## 3 | Model Creation ##
"""Create model to fit it to the data"""

# 3.1 Creating Bag Of Words (BOW)
x_data = np.vstack(dataset.review)
y_data = dataset.sentiment.to_numpy()
#X = cv.fit_transform(dataset.review).toarray()
print(f"=== Bag of words ===\n")
print(f"BOW X shape : {x_data.shape}")
print(f"BOW y shape : {y_data.shape}\n")

#3.2 Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=0)

#3.3 Train Bayes Network
gauss_nb = GaussianNB()
trained_model = gauss_nb.fit(x_train, y_train)
y_pred = trained_model.predict(x_test)

print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))