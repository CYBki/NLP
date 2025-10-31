# import library
from sklearn.feature_extraction.text import CountVectorizer

# sample text

documents = [
    "I love natural language processing.",
    "N-grams are useful for text analysis.",
    "This is a sample document for n-gram modeling."
]
# n-gram model with 3 different N values: unigram, bigram, and trigram
vectorizer_unigram = CountVectorizer(ngram_range=(1, 1))
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2)) 
vectorizer_trigram =CountVectorizer(ngram_range=(3,3))

# unigram
X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()

# bigram
X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()
# print(bigram_features)

# trigram
X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()

# analysis of the results
print(f"Unigram Features: {unigram_features}")
print(f"Bigram Features: {bigram_features}")
print(f"Trigram Features: {trigram_features}")
