#import libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# upload dataset

df = pd.read_csv("spam.csv", encoding='latin-1')
print(df.head())

# tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.text)

# word cluster analysis

# create dataframe including tf-idf scores