import numpy as np
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.keyedvectors import load_word2vec_format
from scipy.spatial.distance import cosine
import os
import logging
import json
import joblib

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "embedding"
    )
    model = joblib.load(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    sentence_1 = json.loads(raw_data)["text1"]
    sentence_2 = json.loads(raw_data)["text2"]
 
    def sentence_embedding(sentence, word_embeddings):
        words = sentence.split()
        embeddings = [word_embeddings[word] for word in words if word in word_embeddings]
        if len(embeddings) > 0:
            return np.mean(embeddings, axis=0)
        else:
            return None
           
    word_embeddings=load_word2vec_format('embedding')
    embedding_1 = sentence_embedding(sentence_1, word_embeddings)
    embedding_2 = sentence_embedding(sentence_2, word_embeddings)
    if embedding_1 is not None and embedding_2 is not None:
        similarity_score = 1 - cosine(embedding_1, embedding_2)
    else:
        similarity_score = None
    result = similarity_score
    logging.info("Request processed")
    return result