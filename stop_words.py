import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')  # the dataset with the most stopwords in different languages


# English stop words analysis (nltk)
stopwords_en = set(stopwords.words('english'))

print(stopwords_en)

text = "This is a sample sentence, showing off the stop words filtration."
text_list = text.split()
# if word not in stopwords_en, add to the filtered_words list
filtered_words = [word for word in text_list if word.lower() not in stopwords_en]
print(f"filtered_words : {filtered_words}")

# Turkish stop words analysis (nltk)
stop_words_tr = set(stopwords.words('turkish'))

metin = "Bu bir örnek cümledir, durdurma kelimeleri süzme işlemini göstermektedir."
metin_list = metin.split()

filtered_words_tr = [word for word in metin_list if word.lower() not in stop_words_tr]
print(f"filtered_words_tr : {filtered_words_tr}")
# stop words removal without using a library

tr_stop_words = ['acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu',
                 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hemen', 'hep',
                 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde']

metin2 = "Bu bir örnek cümledir, durdurma kelimeleri süzme işlemini göstermektedir mi acaba?"

filtered_words_tr2 = set([word for word in metin2.split() if word.lower().strip('.,!?') not in tr_stop_words])
print(f"filtered_words_tr2 : {filtered_words_tr2}") 