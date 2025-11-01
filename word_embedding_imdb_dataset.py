# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# upload dataset
df = pd.read_csv('IMDB Dataset.csv')
documents = df["review"]

# text cleaning
def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # remove special characters
    text = text.replace("br", " ")  # remove HTML line breaks
    text = " ".join([word for word in text.split() if len(word) > 2 and word not in stop_words])  # remove short words
    return text
cleaned_documents = [clean_text(doc) for doc in documents]
# text tokenization
tokenized_documents = [simple_preprocess(doc) for doc in cleaned_documents]



# define the word2vec model
model = Word2Vec(sentences=tokenized_documents, vector_size=50, window=5, min_count=1, sg=0)
word_vectors = model.wv
words = list(word_vectors.index_to_key)
vectors = [word_vectors[word] for word in words]

# clustering KMeans K = 3
kmeans = KMeans(n_clusters=3)
kmeans.fit(vectors)
clusters = kmeans.labels_
#print(clusters)
# PCA 50 -> 2
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)


# 2d visualization
plt.figure(figsize=(12, 8))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=130, marker='X', label = "Center")  # cluster centers
plt.legend()

for i, word in enumerate(words):
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word, fontsize=7)
plt.title("Word Embeddings Clustering with KMeans")
#plt.show()



# show words grouped by cluster
clustered_words = pd.DataFrame({'word': words, 'cluster': clusters})
for i in range(3):  # since K = 3
    print(f"\nCluster {i}:")
    print(clustered_words[clustered_words['cluster'] == i]['word'].head(20).to_list())


from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

# calculate distances between each word vector and cluster centers
distances = cdist(vectors, kmeans.cluster_centers_, metric='euclidean')

# create DataFrame for readability
cluster_words = pd.DataFrame({'word': words, 'cluster': clusters})

# show top 10 representative words for each cluster
for i in range(3):  # K=3
    # find the indices of words closest to cluster center i
    idx = np.argsort(distances[:, i])[:10]
    representative_words = [words[j] for j in idx]
    print(f"\nCluster {i} representative words:")
    print(representative_words)


"""
 output:
Cluster 0:
['man', 'old', 'young', 'gets', 'role', 'guy', 'hes', 'family', 'performance', 'played', 'girl', 'woman', 'comes', 'goes', 'looks', 'job', 'plays', 'later', 'takes', 'american']

Cluster 1:
['pros', 'itd', 'dreadfully', 'badness', 'fuss', 'goofs', 'existent', 'goers', 'heartily', 'monstrosity', 'actorsactresses', 'duckling', 'listing', 'negatives', 'chronological', 'congratulations', 'rewatch', 'exceeded', 'compilation', 'dime']

Cluster 2:
['movie', 'film', 'one', 'like', 'good', 'even', 'would', 'time', 'really', 'see', 'story', 'well', 'much', 'get', 'bad', 'great', 'also', 'people', 'first', 'dont']

Cluster 0 representative words:
['dos', 'boozy', 'lulu', 'château', 'protégé', 'aur', 'ridge', 'yousef', 'isabella', 'tanaka']

Cluster 1 representative words:
['filmmakersfrom', 'gaybody', 'kleber', 'lîle', 'coeurs', 'tamera', 'okwwfst', 'isthomas', 'curatola', 'womenspicture']

Cluster 2 representative words:
['conclude', 'retrospect', 'trivial', 'forgivable', 'incomplete', 'unimportant', 'dumbed', 'sufficiently', 'overlook', 'compliment']  
   
    
 """