import tensorflow as tf
import os
import numpy as np
from load_data import *

X_train, y_train, X_valid, y_valid, X_test, y_test, vocab = load_data()

vocab_size = vocab.shape[0]
batch_size = 32
input_size = 3
embed_size = 64
hidden_size = 256

reg_strength = 1e-7

graph = tf.Graph()
with graph.as_default():
	with tf.device('/gpu:0'):
	    tf_X_train = tf.placeholder(tf.int32, shape=(batch_size, input_size))
	    tf_y_train = tf.placeholder(tf.float32, shape=(batch_size, vocab_size))
	    tf_X_valid = tf.constant(X_valid)
	    tf_X_test = tf.constant(X_test)

	    # convert labels to one-hot vectors
	    train_labels = np.zeros((len(y_train), vocab_size))
	    train_labels[range(len(train_labels)), y_train.reshape((-1,))] = 1
	    valid_labels = np.zeros((len(y_valid), vocab_size))
	    valid_labels[range(len(valid_labels)), y_valid.reshape((-1,))] = 1
	    test_labels = np.zeros((len(y_test), vocab_size))
	    test_labels[range(len(test_labels)), y_test.reshape((-1,))] = 1

	    # model parameters
	    word_embedding_weights = tf.Variable(tf.truncated_normal([vocab_size, embed_size], stddev=0.01))
	    embed_to_hid_weights = tf.Variable(tf.truncated_normal([embed_size*input_size, hidden_size], stddev=0.01))
	    embed_to_hid_bias = tf.Variable(tf.zeros([hidden_size]))
	    hid_to_output_weights = tf.Variable(tf.truncated_normal([hidden_size, vocab_size], stddev=0.01))
	    hid_to_output_bias = tf.Variable(tf.zeros([vocab_size]))

	    def model(tf_X):
	        word_embedding_layer = tf.nn.embedding_lookup(word_embedding_weights, tf_X)
	        word_embedding_layer = tf.reshape(word_embedding_layer, (-1, embed_size*input_size))    
	        hidden_layers_input = tf.matmul(word_embedding_layer, embed_to_hid_weights) + embed_to_hid_bias
	        hidden_layers_output = tf.nn.sigmoid(hidden_layers_input)
	        output = tf.matmul(hidden_layers_output, hid_to_output_weights) + hid_to_output_bias
	        return output

	    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y_train, logits=model(tf_X_train))) + reg_strength*tf.square(tf.norm(hid_to_output_weights))
	    optimizer = tf.train.AdamOptimizer().minimize(loss)

	    train_predictions = tf.nn.softmax(model(tf_X_train))
	    valid_predictions = tf.nn.softmax(model(tf_X_valid))
	    test_predictions = tf.nn.softmax(model(tf_X_test))

	    valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels, logits=model(tf_X_valid)))
	    test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_labels, logits=model(tf_X_test)))

	    valid_accuracy = 100.0*tf.reduce_mean(tf.cast(tf.equal(tf.argmax(valid_predictions, axis=1), tf.argmax(valid_labels, axis=1)), tf.float32))

	    # train loss and validation loss are shown in tensorboard
	    loss_summary = tf.summary.scalar('train_loss', loss)
	    valid_loss_summary = tf.summary.scalar('valid_loss', valid_loss)
	    valid_acc_summary = tf.summary.scalar('valid_accuracy', valid_accuracy)

	    saver = tf.train.Saver(var_list=[word_embedding_weights, embed_to_hid_weights, embed_to_hid_bias, hid_to_output_weights, hid_to_output_bias])
    

def accuracy(predictions, labels):
    return 100.0*np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))


num_epochs = 30
num_batches = int(len(X_train) / batch_size) # num batches per epoch
valid_accuracy_history = [0]

show_training_CE_after = 100;
show_valid_CE_after = 1500;

stop_train_point = None
last_valid_eval_step = None
best_model_step = None
stop_train_patience = 2
stop_train = False


model_name = 'model-19-6-2017'
with tf.device('/gpu:0'), tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
    print('Initialized')

    train_writer = tf.summary.FileWriter('.\\%s\\tensorboard'%model_name, session.graph)
    tf.global_variables_initializer().run()

    p_word_embedding_weights = word_embedding_weights.eval()
    p_embed_to_hid_weights = embed_to_hid_weights.eval()
    p_embed_to_hid_bias = embed_to_hid_bias.eval()
    p_hid_to_output_weights = hid_to_output_weights.eval()
    p_hid_to_output_bias = hid_to_output_bias.eval()

    #saver.restore(session, '.\model.ckpt')
    count = 0
    for epoch in range(num_epochs):
        print ("-------------------------------")
        print ("EPOCH %d" % epoch)
        count = 0
        this_chunk_CE = 0
        trainset_CE = 0
        for step in range(num_batches):
            count += 1
            #mask = np.random.choice(range(X_train.shape[0]), batch_size)
            mask = np.arange(step*batch_size%X_train.shape[0], (step + 1)*batch_size%X_train.shape[0])
            batch_data = X_train[mask]
            batch_labels = train_labels[mask]
            _, l, predictions = session.run([optimizer, loss, train_predictions], feed_dict={tf_X_train: batch_data, tf_y_train:batch_labels})
            this_chunk_CE += (l - this_chunk_CE)/count
            trainset_CE += (l - trainset_CE)/(step+1)
            if step % show_training_CE_after == 0:
                print ("Average training loss at step %d: %f"%(step, this_chunk_CE))
                print ("Minibatch accuracy: %.3f%%" % accuracy(predictions, batch_labels))
                summary = session.run(loss_summary, feed_dict={tf_X_train: batch_data, tf_y_train:batch_labels})
                train_writer.add_summary(summary , epoch*num_batches+step)
                count = 0
                this_chunk_CE = 0
            if step % show_valid_CE_after == 0: # compute validation loss and validation accuracy
                valid_loss_value = valid_loss.eval()
                valid_accuracy_value = accuracy(valid_predictions.eval(), valid_labels)
                valid_accuracy_history.append(valid_accuracy_value)

                print ("Validation loss at step %d: %f"%(step, valid_loss_value))
                print ("Validation accuracy: %.3f%%" % valid_accuracy_value)
                summary = session.run(valid_loss_summary)
                train_writer.add_summary(summary, epoch*num_batches+step)
                summary = session.run(valid_acc_summary)
                train_writer.add_summary(summary, epoch*num_batches+step)

                # if validation accuracy in this time less than the last value, then we consider early stopping
                if valid_accuracy_value < valid_accuracy_history[-2] and stop_train_point == None: 
                    stop_train_point = len(valid_accuracy_history)-2
                    # copy the current values of the parameters to the temp variables
                    c_word_embedding_weights = word_embedding_weights.eval()
                    c_embed_to_hid_weights = embed_to_hid_weights.eval()
                    c_embed_to_hid_bias = embed_to_hid_bias.eval()
                    c_hid_to_output_weights = hid_to_output_weights.eval()
                    c_hid_to_output_bias = hid_to_output_bias.eval()
                    # rollback the parameters to their last values
                    session.run(word_embedding_weights.assign(p_word_embedding_weights))
                    session.run(embed_to_hid_weights.assign(p_embed_to_hid_weights))
                    session.run(embed_to_hid_bias.assign(p_embed_to_hid_bias))
                    session.run(hid_to_output_weights.assign(p_hid_to_output_weights))
                    session.run(hid_to_output_bias.assign(p_hid_to_output_bias))
                    # save the model
                    best_model_step = last_valid_eval_step
                    saver.save(session, '.\%s\model-step%d.ckpt'%(model_name, best_model_step))
                    # restore current values of parameters
                    session.run(word_embedding_weights.assign(c_word_embedding_weights))
                    session.run(embed_to_hid_weights.assign(c_embed_to_hid_weights))
                    session.run(embed_to_hid_bias.assign(c_embed_to_hid_bias))
                    session.run(hid_to_output_weights.assign(c_hid_to_output_weights))
                    session.run(hid_to_output_bias.assign(c_hid_to_output_bias))
                # if after less than or equal $stop_train_patience$ times, the validation accuracy is improved, not stopping
                if stop_train_point != None and len(valid_accuracy_history)-1 <= stop_train_point + stop_train_patience:
                    if valid_accuracy_value > valid_accuracy_history[stop_train_point]:
                        stop_train_point = None
                # if after more than $stop_train_patience$ times, the validation accuracy is not improved, we make early stoppping
                if stop_train_point != None and len(valid_accuracy_history)-1 > stop_train_point + stop_train_patience:
                    stop_train = True
                    print ("Early stopping at global step %d, best model is at global step %d"%(epoch*num_batches+step, best_model_step))
                    break

                last_valid_eval_step = epoch*num_batches+step
                p_word_embedding_weights = word_embedding_weights.eval()
                p_embed_to_hid_weights = embed_to_hid_weights.eval()
                p_embed_to_hid_bias = embed_to_hid_bias.eval()
                p_hid_to_output_weights = hid_to_output_weights.eval()
                p_hid_to_output_bias = hid_to_output_bias.eval()
        if stop_train == True:
            break
        print ("Training loss at epoch %d: %f"%(epoch, trainset_CE))
    
    if stop_train == False: # if not early stopping, save the last model        
        saver.save(session, '.\%s\model-final.ckpt'%model_name)
    else: # if early stopping, restore the best model
        saver.restore(session, '.\%s\model-step%d.ckpt'%(model_name, best_model_step))

    print ("Final validation loss: %f"%valid_loss.eval())
    print ("Final validation accuracy: %f%%"%accuracy(valid_predictions.eval(), valid_labels))
    print ("Test set loss: %f"%test_loss.eval())
    print ("Test set accuracy: %f%%"%accuracy(test_predictions.eval(), test_labels))
