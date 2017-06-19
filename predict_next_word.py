import tensorflow as tf
import numpy as np 
from load_data import *


vocab = load_data()[-1]

vocab_size = vocab.shape[0]
batch_size = 256
input_size = 3
embed_size = 64
hidden_size = 256

with tf.device('/gpu:0'):
    word_embedding_weights = tf.Variable(tf.zeros([vocab_size, embed_size]))
    embed_to_hid_weights = tf.Variable(tf.zeros([embed_size*input_size, hidden_size]))
    embed_to_hid_bias = tf.Variable(tf.zeros([hidden_size]))
    hid_to_output_weights = tf.Variable(tf.zeros([hidden_size, vocab_size]))
    hid_to_output_bias = tf.Variable(tf.zeros([vocab_size]))

    def model(tf_X):
        word_embedding_layer = tf.nn.embedding_lookup(word_embedding_weights, tf_X)
        word_embedding_layer = tf.reshape(word_embedding_layer, (-1, embed_size*input_size))    
        hidden_layers_input = tf.matmul(word_embedding_layer, embed_to_hid_weights) + embed_to_hid_bias
        hidden_layers_output = tf.nn.sigmoid(hidden_layers_input)
        output = tf.matmul(hidden_layers_output, hid_to_output_weights) + hid_to_output_bias
        return output

    def predict_next_word(word1, word2, word3, model_file, k=5):
        word1_id = list(vocab).index(word1)
        word2_id = list(vocab).index(word2)
        word3_id = list(vocab).index(word3)   

        tf_X_test = tf.constant([word1_id, word2_id, word3_id])

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, model_file) # restore weights and bias from model file

        probs = tf.nn.softmax(model(tf_X_test))
        probs_array = session.run(probs).reshape((-1,))
        predicted_ids = np.argsort(-probs_array)[:k] # sorted by decrease order
        saver.save(session, model_file)
        session.close()
        for predicted_id in predicted_ids:
            print ("%s %s %s %s. Prob: %.5f\n" % (word1, word2, word3, vocab[predicted_id], probs_array[predicted_id]))
