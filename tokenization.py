import nltk 
# nltk.download('punkt_tab')  required to split text into tokens at the word and sentence level

text = "Hello, world! This is a test ..."

# word tokenization: word_tokenize splits text into words,
# punctuation marks and spaces are treated as separate tokens.

word_tokens = nltk.word_tokenize(text)
print(word_tokens)

a = len(word_tokens)
print(a)


# sentence tokenization: sent_tokenize splits text into sentences.
# each sentence is considered a single token.
sentence_tokens = nltk.sent_tokenize(text)
print(sentence_tokens)