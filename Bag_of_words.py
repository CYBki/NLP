# import count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# create dataset
documents = [
    "kedi bahçede",   # cat in the garden
    "kedi evde"       # cat at home
]

# define vectorizer
vectorizer = CountVectorizer()

# convert text into numerical vectors
X = vectorizer.fit_transform(documents)

# create vocabulary (bahçede, evde, kedi)
feature_names = vectorizer.get_feature_names_out()
print(f"Vocabulary: {feature_names}")

# vector representation
vector_representation = X.toarray()
print(f"Vector representation: {vector_representation}")

"""
Vocabulary: ['bahçede' 'evde' 'kedi']
Vector representation: [[1 0 1]
                        [0 1 1]]
"""
