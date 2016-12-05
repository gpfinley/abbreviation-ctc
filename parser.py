__author__ = 'gpfinley'

"""

idea: train what is effectively a constituency parser using bi-directional LSTM that learns the boundaries of phrases
        (ends going forward, beginnings going backward)

still need: data source (and munging code; big), batching function (small), grad descent step (small), and decoder (big)

"""

import numpy as np
import tensorflow as tf
from read_word2vec import get_dictionary

LSTM_UNITS = 100
BATCH_SIZE = 100
NUM_ITERATIONS = 10000000
VECTORS_PATH = "wiki-vectors_lc.bin"

# maximum number of words per sentence
maxlen = 30

# EMBEDDINGS
embeddings_dic = get_dictionary(VECTORS_PATH)
words, embeddings = zip(*embeddings_dic.iteritems())
n_words = len(words)
embeddings_dimensionality = len(embeddings[0])

# "map" word indices to embeddings (perform mapping by multiplying a one-hot matrix by this matrix)
embeddings_matrix = np.array(embeddings)
embeddings = None

# map words to integers
vocab = {word:i for (i, word) in enumerate(words)}
vocab_size = len(words)


# todo: load treebank data and convert (inputs to list of word indices; outputs to n-hot phrase type vectors)
#           (or load embeddings directly rather than training on them!)



# number of types of phrases to detect
n_classes = 12

# num_sentences x max_seq_len; values in range(0, vocab_size)
this_batch_dense = tf.placeholder(dtype=tf.int32, shape=(None, maxlen))

# num_sentences x max_seq_len x vocab_size; one-hot
this_batch_onehot = tf.one_hot(this_batch_dense, depth=vocab_size)

# num_sentences x max_seq_len x embeddings_dimensionality
this_batch_embeddings = tf.einsum("abi,ic->abc", this_batch_onehot, embeddings_matrix)
# reverse this batch in time
this_batch_backwards_embeddings = tf.reverse(this_batch_embeddings, dims=(0,1,0))

this_batch_true_classes_f = tf.placeholder(dtype=tf.float32, shape=(None, n_classes))
this_batch_true_classes_b = tf.placeholder(dtype=tf.float32, shape=(None, n_classes))

lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_UNITS)
forward_rnn = tf.nn.rnn(cell=lstm_cell, inputs=this_batch_embeddings)
backward_rnn = tf.nn.rnn(cell=lstm_cell, inputs=this_batch_backwards_embeddings)

# project lstm outputs to classes of phrase types for foward pass
W_f = tf.Variable(initial_value=np.random.random(size=(LSTM_UNITS, n_classes)))
b_f = tf.Variable(initial_value=np.random.random(size=(1, n_classes)))

# ...for backward pass
W_b = tf.Variable(initial_value=np.random.random(size=(LSTM_UNITS, n_classes)))
b_b = tf.Variable(initial_value=np.random.random(size=(1, n_classes)))

class_layer_f = tf.matmul(forward_rnn, W_f) + b_f
class_layer_b = tf.matmul(backward_rnn, W_b) + b_b

# todo: confirm how this loss function works (logits on just input, n-hot on output??)
loss_f = tf.nn.sigmoid_cross_entropy_with_logits(class_layer_f, this_batch_true_classes_f)
loss_b = tf.nn.sigmoid_cross_entropy_with_logits(class_layer_b, this_batch_true_classes_b)

# todo: gradient descent step
update_step = ...


# todo: batch and feed data into graph

# denser representation: matrix: #_sentences x max_length (and values in domain of vocab_size)
#       (hopefully this can hold most or all of the data in memory?)
# use tf.one_hot to get: representation of input data: 3d tensor: #_sentences x max_length x vocab_size
# matmul step to get embeddings by matrix: vocab_size x embedding_length

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for _ in range(NUM_ITERATIONS):
        # todo: get next batch (dense matrix of words; forward classes; backward classes)
        feed_dict = {this_batch_dense : next_batch_dense,
                    this_batch_true_classes_f : next_batch_forward_classes,
                    this_batch_true_classes_b : next_batch_backward_classes}
        val_loss_f, val_loss_b, _ = session.run([loss_f, loss_b, update_step], feed_dict=feed_dict)

# todo: decode and score
#           how will decoding work? will be tricky with multiple (or zero) classes. todo: invent an algorithm for that?
