# import libraries

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import collections 
from collections import Counter

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')




# import dataset
df = pd.read_csv("IMDB Dataset.csv")

# get text data
documents = df["review"]
labels = df["sentiment"]   # positive or negative

# text cleaning
def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove digits
    text = re.sub(r"\d+", "", text)

    # remove special characters
    text = re.sub(r"[^A\w\s]", "", text)

    # remove short words
    text = " ".join([word for word in text.split() if len(word) > 2])

    return text  # return cleaned text

#print(clean_text("Sample Text! With digits 123 and short words a an the."))  # test the function

cleaned_doc = [clean_text(row) for row in documents]
#print(cleaned_doc[:5]) # print first 5 cleaned reviews


# define vectorizer
vectorizer = CountVectorizer()

# convert text into numerical vectors
X = vectorizer.fit_transform(cleaned_doc[:75])

#print(X)

# show vocabulary
feature_names = vectorizer.get_feature_names_out()
#print(f"Vocabulary: {feature_names}")

# vector representation
vector_representation2= X.toarray()
#print(f"Vector representation: {vector_representation2}")

df_bow = pd.DataFrame(vector_representation2, columns=feature_names)
#print(df_bow.head())

# word frequency analysis
word_counts = X.sum(axis=0).A1  # sum each column to get word frequencies
word_freq = dict(zip(feature_names, word_counts))  # convert to dictionary
# print(f"Word Frequencies: {word_freq}")

# most common words
most_common_words = Counter(word_freq).most_common(10)
# print(f"Most Common Words: {most_common_words}")


# remove stop words
stop_words = list(stopwords.words('english'))

# Which stopwords are in the word set?

present_stopwords = [w for w in feature_names if w in stop_words]
print(f"Present Stopwords: {present_stopwords}")

# redefine vectorizer with stop words removal
vectorizer_sw = CountVectorizer(stop_words=stop_words)
X_sw = vectorizer_sw.fit_transform(cleaned_doc[:75])

# new vocabulary and frequencies
feature_names_sw = vectorizer_sw.get_feature_names_out()
word_counts_sw = X_sw.sum(axis=0).A1
word_freq_sw = dict(zip(feature_names_sw, word_counts_sw))

print(f"Word Frequencies (without stop words): word_freq_sw")

most_common_words_sw = Counter(word_freq_sw).most_common(10)
print(f"Most Common Words (without stop words): {most_common_words_sw}")


# visualization 
import matplotlib.pyplot as plt

words, counts = zip(*most_common_words_sw)
plt.bar(words, counts)
plt.title("Most Common Words (without stopwords)")
plt.xticks(rotation=45)
plt.show()


# Create a word cloud to visualize word frequencies
from wordcloud import WordCloud

text = " ".join(cleaned_doc)
wc = WordCloud(width=800, height=400, background_color='white',
               stopwords=stop_words).generate(text)

plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

