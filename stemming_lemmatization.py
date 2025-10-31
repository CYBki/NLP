import nltk

nltk.download('wordnet')  # the database required for lemmatization

from nltk.stem import PorterStemmer #function for stemming

# Create porter stemmer object
stemmer = PorterStemmer()
words = ["running", "runner", "better","ran", "runs", "went"]

# Apply stemming to each word in the list   
stems = [stemmer.stem(w) for w in words]

print(f"Stems : {stems}")   

from nltk.stem import WordNetLemmatizer #function for lemmatization

lemmatizer = WordNetLemmatizer()

words = ["running", "runner", "better","ran", "runs", "went"]

# Apply stemming to each word in the list   
lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words]

print(f"Lemmas : {lemmas}")   
