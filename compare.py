# import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# sample text
documents = [
    "I love natural language processing.",
    "N-grams are useful for text analysis.",
    "This is a sample document for n-gram modeling."
]

# 1️⃣ Unigram model
tfidf_uni = TfidfVectorizer(ngram_range=(1, 1))
X_uni = tfidf_uni.fit_transform(documents)
df_uni = pd.DataFrame(X_uni.toarray(), columns=tfidf_uni.get_feature_names_out())
print("\n--- Unigram TF-IDF ---")
print(df_uni.mean().sort_values(ascending=False).head(10))

# 2️⃣ Bigram model
tfidf_bi = TfidfVectorizer(ngram_range=(2, 2))
X_bi = tfidf_bi.fit_transform(documents)
df_bi = pd.DataFrame(X_bi.toarray(), columns=tfidf_bi.get_feature_names_out())
print("\n--- Bigram TF-IDF ---")
print(df_bi.mean().sort_values(ascending=False).head(10))

# 3️⃣ Trigram model
tfidf_tri = TfidfVectorizer(ngram_range=(3, 3))
X_tri = tfidf_tri.fit_transform(documents)
df_tri = pd.DataFrame(X_tri.toarray(), columns=tfidf_tri.get_feature_names_out())
print("\n--- Trigram TF-IDF ---")
print(df_tri.mean().sort_values(ascending=False).head(10))


"""
Output:
--- Unigram TF-IDF ---
for           0.206174
language      0.166667
natural       0.166667
love          0.166667
processing    0.166667
grams         0.141131
are           0.141131
analysis      0.141131
useful        0.141131
text          0.141131
dtype: float64

--- Bigram TF-IDF ---
love natural           0.192450
language processing    0.192450
natural language       0.192450
grams are              0.149071
are useful             0.149071
for text               0.149071
useful for             0.149071
text analysis          0.149071
document for           0.136083
for gram               0.136083
dtype: float64

--- Trigram TF-IDF ---
love natural language          0.235702
natural language processing    0.235702
are useful for                 0.166667
useful for text                0.166667
for text analysis              0.166667
grams are useful               0.166667
document for gram              0.149071
for gram modeling              0.149071
is sample document             0.149071
sample document for            0.149071
dtype: float64
"""