import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec,KeyedVectors
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def preprocess_text(sentence):
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    tokens = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_sentence = ' '.join(tokens)
    return preprocessed_sentence
data=pd.read_csv("/content/dataset.csv")
df1=data['text1'].apply(preprocess_text).tolist()
df2=data['text2'].apply(preprocess_text).tolist()
sentences=df1+df2
individual_lists = [[word for word in sentence.split()] for sentence in sentences]
individual_lists
def train_glove_embeddings(sentences, embedding_dim, window_size, min_count):
    model = Word2Vec(sentences, vector_size=embedding_dim, window=window, min_count=min_count,epochs=epochs)
    word_vectors = model.wv
    return word_vectors
embedding_dim = 1024
window = 8
min_count = 2
epochs=5
glove_embeddings = train_glove_embeddings(individual_lists, embedding_dim, window, min_count)
print(glove_embeddings.similar_by_word("money"))
print(glove_embeddings.similar_by_word("football"))
print(glove_embeddings.similar_by_word("internet"))
glove_embeddings.save_word2vec_format('embedding')
def sentence_embedding(sentence, word_embeddings):
    words = sentence.split()
    embeddings = [word_embeddings[word] for word in words if word in word_embeddings]
    if len(embeddings) > 0:
        return(np.average(embeddings, axis=0))
    else:
        return None

sentence_1 = "The professional football player signed a multi-million dollar contract with the club."
sentence_2 = "The investment firm announced a new fund for emerging markets and global growth."
word_embeddings = glove_embeddings

embedding_1 = sentence_embedding(preprocess_text(sentence_1), word_embeddings)
embedding_2 = sentence_embedding(preprocess_text(sentence_2), word_embeddings)
if embedding_1 is not None and embedding_2 is not None:
    similarity_score = 1-cosine(embedding_1, embedding_2)
else:
    similarity_score = None
similarity_score