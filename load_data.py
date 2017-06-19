import numpy as np
import scipy.io as sio

def load_data():    
    data = sio.loadmat("data.mat")['data'][0][0]
    test = data[0] # 4 x test_size
    train = data[1] # 4 x train_size
    valid = data[2] # 4 x valid_size
    vocab = data[3] # 1 x vocab_size

    test_size = test.shape[1]
    train_size = train.shape[1]
    valid_size = valid.shape[1]
    vocab_size = vocab.shape[1]

    # all the values are subtraced by 1 because the range is from 0
    X_test = test.T[:, :3] - 1 # test_size x 3
    X_train = train.T[:, :3] - 1 # train_size x 3
    X_valid = valid.T[:, :3] - 1 # valid_size x 3
    y_test = test.T[:, 3, None] - 1 # test_size x 1
    y_train = train.T[:, 3, None] - 1 # train_size x 1
    y_valid = valid.T[:, 3, None] - 1 # valid_size x 1
    vocab = vocab.reshape((-1,)) # vocab_size
    vocab = np.array([vocab[i][0] for i in range(vocab.shape[0])])

    return X_train, y_train, X_valid, y_valid, X_test, y_test, vocab