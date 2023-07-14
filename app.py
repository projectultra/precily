import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from gensim.models.keyedvectors import load_word2vec_format
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify
app = Flask(__name__)


def preprocess_text(sentence):
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    tokens = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_sentence = ' '.join(tokens)
    return preprocessed_sentence

def sentence_embedding(sentence, word_embeddings):
    words = sentence.split()
    embeddings = [word_embeddings[word] for word in words if word in word_embeddings]
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return None

def calculate_similarity(text1, text2):
    word_embeddings=load_word2vec_format(r'embedding')
    embedding_1 = sentence_embedding(preprocess_text(text1), word_embeddings)
    embedding_2 = sentence_embedding(preprocess_text(text2), word_embeddings)
    if embedding_1 is not None and embedding_2 is not None:
        similarity_score = 1 - cosine(embedding_1, embedding_2)
    else:
        similarity_score = None
    return similarity_score

@app.route('/api/endpoint', methods=['POST'])
def similarity_endpoint():
    # Retrieve the data from the request payload
    data = request.get_json()
    
    # Extract text1 and text2 from the payload
    text1 = data.get('text1')
    text2 = data.get('text2')
    
    # Perform similarity calculation or any other processing
    similarity_score = calculate_similarity(text1, text2)
    
    # Create the response payload
    response = {'similarity score': similarity_score}
    
    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run()