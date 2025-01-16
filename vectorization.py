
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    "This is the first document. jitesh ",
    "This document is the second document.  dewangan",
    "And this is the third one.",
    "Is this the first document?",
]

# Create an instance of the CountVectorizer
vectorizer = CountVectorizer()
# Fit and transform the corpus into a document-term matrix
X = vectorizer.fit_transform(corpus)

# Print the vocabulary and the document-term matrix
print("Vocabulary:", vectorizer.get_feature_names_out())
print("Document-Term Matrix:")
print(X.toarray())

## unigram and bigram 
## ngram_range=(min, max) -> use all n-gram in this range
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 3))
X2 = vectorizer2.fit_transform(corpus)
print("vocab: " , vectorizer2.get_feature_names_out())
print("Document-Term Matrix:")
print(X2.toarray())

#using TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    "This is the first document. jitesh ",
    "This document is the second document.  dewangan",
    "And this is the third one.",
    "Is this the first document?",
]
vectorizer3 = TfidfVectorizer()
X3 = vectorizer3.fit_transform(corpus)
print("vocab: " , vectorizer3.get_feature_names_out())
print("Document-Term Matrix:")
print(X3.toarray())

## using word2vec
from gensim.models import Word2Vec
def corpus_to_token(corpus):
    token = []
    for sent in corpus:
        #print(sent)
        token.append(sent.lower().split())
        #print(token)
    return token


tokenizedtext  = corpus_to_token(corpus)
word2vec_model = Word2Vec(sentences=tokenizedtext, vector_size=100, min_count=1, workers=4)

# STOP OWRLD removal from tkt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    token = []
    for sent in corpus:
        vocab = sent.split()
        token.append([word for word in vocab if word not in stop_words])
        print(token)
    return token
    #return [word for word in tokens if word not in stop_words]
print(remove_stopwords(corpus))






# In[ ]:





# In[ ]:




