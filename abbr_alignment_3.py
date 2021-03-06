from __future__ import division
import tensorflow as tf
import numpy as np
import logging
import random
import editdistance

__author__ = 'gpfinley'


# TODO: show progress on held-out set during training


"""

Learn to map phrases to their abbreviations


Better code all around, and especially for GPU (load all values in dense format onto gpu)


some notes from implementation:
    - be sure there is no extra activation of rnn cell output and no activation on the layer immediately preceding ctc loss calculation
    - be sure the sequence_length is ALWAYS length of the long form, not abbr


"""

EMPTY = '_'
use_house_decoder = 1
scrub_abbr_periods = 1
case_sensitive = 1
acronyms_only = 1
batch_size = 100
lr =.00001
use_lstm = 1
bidirectional = 1
num_iter = 1000000
print globals()


logging.basicConfig(level=logging.DEBUG)

# read all lines from both databases and specify that held out starts right at beginning of adam

lrabr_lines = open('adam_lrabr_data/lrabr_only.txt').read().strip().split('\n')
adam_lines = open('adam_lrabr_data/adam_only.txt').read().strip().split('\n')[:10000]
begin_held_out = len(lrabr_lines) // 2

string_abbrs = lrabr_lines[::2] + adam_lines[::2]
string_longs = lrabr_lines[1::2] + adam_lines[1::2]

good_indices = [x for x in range(len(string_abbrs)) if len(string_abbrs[x]) < len(string_longs[x])]

string_abbrs = [string_abbrs[i] for i in good_indices]
string_longs = [string_longs[i] for i in good_indices]



print len(string_longs), 'string longs'
print len(string_abbrs), 'string abbrs (should be same)'



string_longs = [' '+x for x in string_longs]

maxlen = max([len(x) for x in string_longs])
# print len(long_abbr)

# Find all characters used in all long forms and abbreviations, including a null character
# all_chars = set.union(*[{letter for letter in x} for x in (long_abbr.keys() + long_abbr.values())])
all_chars = set.union(*[{letter for letter in x} for x in (string_longs + string_abbrs)])
all_chars = sorted(list(all_chars))
all_chars.append(EMPTY)
all_chars.insert(0, '@')
char2ind = {c:i for (i, c) in enumerate(all_chars)}
nchar = len(all_chars)

print 'character to integer dictionary:', char2ind
print all_chars

X = np.zeros((len(string_longs), maxlen)) + char2ind[EMPTY]
y = np.zeros((len(string_longs), maxlen))
# y = np.zeros((len(string_longs), maxlen, nchar))

for i in range(len(string_longs)):
    for t, char in enumerate(string_longs[i]):
        X[i, t] = char2ind[char]
    for t, char in enumerate(string_abbrs[i]):
        # y[i, t, char2ind[char]] = 1
        y[i, t] = char2ind[char]

X_lengths = np.array([len(x) for x in string_longs])




batch_counter = 0
# return the start and end indices for the next batch
def get_next_batch():
    global batch_counter
    batch_counter += 1
    if batch_counter * batch_size > len(string_longs):
        batch_counter = 0
        return 0
    return (batch_counter-1) * batch_size


# convert a matrix of integers (each row is a hypothesis for a given long form)
def hypothesis_to_readable(hypothesis):
    if use_house_decoder:
        return [''.join(all_chars[x] for x in example) for example in hypothesis]
    else:
        strings = []
        indices = hypothesis[0][0].indices
        values = hypothesis[0][0].values
        curexample = -1
        for index, value in zip(indices, values):
            if index[0] > curexample:
                curexample = index[0]
                strings.append('')
            strings[-1] += all_chars[value]
        return strings

        # print hypothesis
        # print hypothesis[0][0].values
        # if hypothesis is the output from ctc_greedy_decoder
        # return [''.join(all_chars[v] for v in example[0].values) for example in hypothesis]




# create held-out set

#begin_held_out = int(len(string_longs) * (1-held_out_frac))
held_out_X = X[begin_held_out:, :]
X = X[:begin_held_out, :]
held_out_y = y[begin_held_out:, :]
y = y[:begin_held_out, :]
held_out_X_lengths = X_lengths[begin_held_out:]
X_lengths = X_lengths[:begin_held_out]
held_string_longs = string_longs[begin_held_out:]
string_longs = string_longs[:begin_held_out]
held_string_abbrs = string_abbrs[begin_held_out:]
string_abbrs = string_abbrs[:begin_held_out]
logging.info('created held out set')


with tf.Graph().as_default():

    rnn_size = 2 * nchar

    all_input_lengths = tf.constant(X_lengths, dtype=tf.int32)
    all_dense_inputs = tf.constant(X, dtype=tf.int32)
    all_dense_outputs = tf.constant(y, dtype=tf.int32)

    batch_begin = tf.placeholder(tf.int32, shape=[1])

    batch_begin_tuple = tf.concat(0, [batch_begin, [0]])
    dense_inputs = tf.slice(all_dense_inputs, batch_begin_tuple, [batch_size, -1])
    input_lengths = tf.slice(all_input_lengths, batch_begin, [batch_size])
    dense_outputs = tf.slice(all_dense_outputs, batch_begin_tuple, [batch_size, -1])

    inputs = tf.one_hot(dense_inputs, nchar)

    def dense_to_sparse_args(dense):
        zero = tf.constant(0)
        nonzero_boolean = tf.not_equal(dense, zero)
        return tf.where(nonzero_boolean), tf.boolean_mask(dense, nonzero_boolean)

    output_indices, output_chars = dense_to_sparse_args(dense_outputs)

    labels = tf.SparseTensor(indices=output_indices, values=output_chars, shape=[batch_size, maxlen])
    labels = tf.to_int32(labels)

    if use_lstm:
        if bidirectional:
            lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(rnn_size, forget_bias=1., state_is_tuple=True)
            lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(rnn_size, forget_bias=1., state_is_tuple=True)
        else:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, forget_bias=1., state_is_tuple=True)
    else:
        if bidirectional:
            lstm_cell_1 = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
            lstm_cell_2 = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
        else:
            lstm_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)

    # initial_state = lstm_cell.zero_state(batch_size, tf.float32)

    # split up inputs to use for basic rnn function (batch represented as a list of vectors)
    inputs_list = tf.unstack(inputs, axis=1)

    with tf.variable_scope('Rnn'):
        # lstm_outputs_timefirst, state = tf.nn.rnn(lstm_cell, inputs_list, sequence_length=input_lengths, dtype=tf.float32) #initial_state=initial_state)
        if bidirectional:
            lstm_outputs_timefirst, state, statebackward = tf.nn.bidirectional_rnn(lstm_cell_1, lstm_cell_2, inputs_list, sequence_length=input_lengths, dtype=tf.float32)
        else:
            lstm_outputs_timefirst, state = tf.nn.rnn(lstm_cell, inputs_list, sequence_length=input_lengths, dtype=tf.float32)
    # reorder these (want batch_size * maxlen * len(all_chars))
    lstm_outputs = tf.transpose(lstm_outputs_timefirst, perm=[1,0,2])

    if bidirectional:
        W_size = rnn_size * 2
    else:
        W_size = rnn_size
    W = tf.Variable(np.random.random(size=(W_size, len(all_chars))), dtype=tf.float32)
    b_array = np.zeros(len(all_chars), dtype=np.float32)
    # build in a little initial bias towards null (hopefully speeds up training)
    b_array[-1] = .1
    b = tf.Variable(b_array, dtype=tf.float32)

    output_layer = tf.einsum('abi,ic->abc', lstm_outputs, W) + b
    cost = tf.nn.ctc_loss(output_layer, labels, input_lengths, ctc_merge_repeated=False, time_major=False)

    latest_state = state

    # optimizer = tf.train.GradientDescentOptimizer(lr)
    # optimizer = tf.train.AdagradOptimizer(lr)
    # optimizer = tf.train.AdadeltaOptimizer(lr)               # needs high learning rate (~100k X momentum rate)
    optimizer = tf.train.MomentumOptimizer(lr, .9)             # needs very low learning rate

    grads_and_vars = optimizer.compute_gradients(cost, tf.trainable_variables())
    train_op = optimizer.apply_gradients(grads_and_vars)

    if use_house_decoder:
        char_hypotheses = tf.arg_max(output_layer, 2)
    else:
        # can also decode with this, although it makes it harder to see the output at each time
        char_hypotheses = tf.nn.ctc_greedy_decoder(tf.transpose(output_layer, perm=[1,0,2]),
                                                        [maxlen] * batch_size,
                                                        merge_repeated=False)



    # FOR HELD OUT SET

    held_input_lengths = tf.constant(held_out_X_lengths, dtype=tf.int32)
    held_dense_inputs = tf.constant(held_out_X, dtype=tf.int32)
    held_dense_outputs = tf.constant(held_out_y, dtype=tf.int32)

    held_inputs = tf.one_hot(held_dense_inputs, nchar)

    held_output_indices, held_output_chars = dense_to_sparse_args(held_dense_outputs)

    held_labels = tf.SparseTensor(indices=held_output_indices, values=held_output_chars, shape=[len(held_out_X_lengths), maxlen])
    held_labels = tf.to_int32(held_labels)

    # split up inputs to use for basic rnn function (batch represented as a list of vectors)
    held_inputs_list = tf.unstack(held_inputs, axis=1)
    with tf.variable_scope('Rnn', reuse=True):
        if bidirectional:
            held_lstm_outputs_timefirst, _, _ = tf.nn.bidirectional_rnn(lstm_cell_1, lstm_cell_2, held_inputs_list, sequence_length=held_input_lengths, dtype=tf.float32)
        else:
            held_lstm_outputs_timefirst, _ = tf.nn.rnn(lstm_cell, held_inputs_list, sequence_length=held_input_lengths, dtype=tf.float32)

    # reorder these (want batch_size * maxlen * len(all_chars))
    held_lstm_outputs = tf.transpose(held_lstm_outputs_timefirst, perm=[1,0,2])

    held_output_layer = tf.einsum('abi,ic->abc', held_lstm_outputs, W) + b
    held_cost = tf.nn.ctc_loss(held_output_layer, held_labels, held_input_lengths, ctc_merge_repeated=False, time_major=False)

    if use_house_decoder:
        held_hypotheses = tf.arg_max(held_output_layer, 2)
    else:
        # can also decode with this, although it makes it harder to see the output at each time
        held_hypotheses = tf.nn.ctc_greedy_decoder(tf.transpose(held_output_layer, perm=[1,0,2]),
                                                        [maxlen] * len(held_out_X_lengths),
                                                        merge_repeated=False)







    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        costs_file = open('costs.txt', 'w')
        dev_file = open('dev_log.txt', 'w')

        for iternum in range(num_iter):

            print iternum

            next_batch_begin = get_next_batch()

            feed_dict={batch_begin : [next_batch_begin]}

            thiscost, state, _, train_hypothesis = session.run([cost,
                                                                 latest_state,
                                                                 train_op,
                                                                 char_hypotheses],
                                                                 feed_dict=feed_dict)

            # print 'costs', thiscost
            mean_cost = sum(thiscost) / batch_size
            print 'mean training cost', mean_cost
            costs_file.write(str(mean_cost) + '\n')
            if iternum % 10 == 0:
                costs_file.flush()

            # edit_dist_hypotheses = [x.replace('_','') for x in readable_hypotheses]
            # edits = [editdistance.eval(hyp, gold) for (hyp, gold) in zip(edit_dist_hypotheses, stringabbrs)]
            # print 'mean edit dist', sum(edits) / batch_size


            if iternum % 100 == 0:
                readable_hypotheses = hypothesis_to_readable(train_hypothesis)
                print zip(string_longs[next_batch_begin:next_batch_begin+batch_size],
                          string_abbrs[next_batch_begin:next_batch_begin+batch_size],
                          readable_hypotheses)


                held_cost_run, held_hypothesis_run = session.run([held_cost, held_hypotheses])
                held_readable_hypotheses = hypothesis_to_readable(held_hypothesis_run)

                mean_dev_cost = sum(held_cost_run) / len(held_cost_run)
                print 'mean dev cost', mean_dev_cost
                dev_file.write(str(mean_dev_cost) + '\n')
                dev_file.write(str(zip(held_string_longs,
                                   held_string_abbrs,
                                   held_readable_hypotheses)))
                dev_file.write('\n')
                dev_file.flush()



                print_first_x = 100
                print zip(held_string_longs[:print_first_x],
                          held_string_abbrs[:print_first_x],
                          held_readable_hypotheses[:print_first_x])

    # todo: save model parameters somehow and enable decoding
    costs_file.close()
    dev_file.close()
