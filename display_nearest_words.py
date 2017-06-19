import numpy as np
import tensorflow as tf

def display_nearest_words(word, word_embedding_weights, vocab, k):
    word_id = list(vocab).index(word)
    distances = np.array([np.linalg.norm(word_embedding_weights[word_id] - word_embedding_weights[i]) for i in range(len(vocab))])
    return vocab[np.argsort(distances)[1:(k+1)]]