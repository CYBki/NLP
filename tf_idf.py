# import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Create a sample document
documents = ["Dog is a cute animal.",
             "Dogs and birds are adorable animals.",
             "Cows make milk."
            ]

# Define TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
# ransform texts into numerical representation
X = tfidf_vectorizer.fit_transform(documents)

# analyze the word cluster
feature_names = tfidf_vectorizer.get_feature_names_out()
#print(f"Vocabulary: {feature_names}")

# Vector representation
vector_representation = X.toarray()
#print(f"Vector representation: {vector_representation}")   
df_tfidf = pd.DataFrame(vector_representation, columns=feature_names)
#print(df_tfidf)

# show the mean TF-IDF score for each word
tf_idf = df_tfidf.mean(axis=0)
print(tf_idf)