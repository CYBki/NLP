#import libraries

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# upload dataset

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]

# change the v2 column name to text
df = df.rename(columns={'v1': 'label', 'v2': 'text'})

# data cleaning  
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # stopwords çıkarma
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# apply cleaning function
df['text'] = df['text'].apply(clean_text)

# tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.text)

#print(df.head())

# word cluster analysis
feature_names = vectorizer.get_feature_names_out()
#print(feature_names)
tfidf_score = X.mean(axis = 0).A1
# print(tfidf_score)

# create dataframe including tf-idf scores
df_tfidf = pd.DataFrame({'word': feature_names, 'tfidf': tfidf_score})
df_tfidf_sorted = df_tfidf.sort_values(by='tfidf', ascending=False)
print(df_tfidf_sorted.head(10))
#print(df.columns)
#print(len(feature_names))