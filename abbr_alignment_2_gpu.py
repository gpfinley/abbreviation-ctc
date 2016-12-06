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

use_lrabr = True
EMPTY = '_'
use_house_decoder = True
scrub_abbr_periods = True
case_sensitive = True
acronyms_only = True
batch_size = 100
lr =.00001
held_out_frac = .1
use_lstm = False
print globals()


logging.basicConfig(level=logging.DEBUG)

if use_lrabr:

    skip_between_every = 0
    total_to_use = 100000
    # todo: free up enough memory to do all of them? right now it doesn't do longform duplicates, and quits after a few
    used_longs = set()
    string_longs = []
    string_abbrs = []
    # with open('/Volumes/gregdata/metathesaurus/2015AA/LEX/LRABR') as f:
    with open('/Users/gpfinley/LRABR') as f:
        used = 0
        skipcounter = 0
        for line in f:
            _, abbr, type, _, long, _ = line.split('|')
            if acronyms_only and type != 'acronym':
                continue
            if skipcounter:
                skipcounter -= 1
                continue
            if scrub_abbr_periods:
                abbr = abbr.replace('.', '')
            if not case_sensitive:
                abbr = abbr.lower()
                long = long.lower()
            if long not in used_longs and len(long) > len(abbr):
                used_longs.add(long)
                string_longs.append(long)
                string_abbrs.append(abbr)
                used += 1
                if used >= total_to_use and total_to_use > 0: break
                skipcounter = skip_between_every

    logging.info('read from LRABR')
    random.seed(10)
    random.shuffle(string_longs)
    logging.info('shuffled longforms')
    random.seed(10)
    random.shuffle(string_abbrs)
    logging.info('shuffled abbrs')



else:
    # for sanity checks
    string_longs = ['word one',
                    'wrd two',
                    'one wrd',
                    'two times',
                    'word word',
                    'tee wee',
                    'we we we',
                    'two one three',
                    'why one two',
                    'oh oh why']
    string_abbrs = ['WO',
                    'WT',
                    'OW',
                    'TT',
                    'WW',
                    'TW',
                    'WWW',
                    'TOT',
                    'WOT',
                    'OOW']


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


# create held-out set

begin_held_out = int(len(X) * (1-held_out_frac))
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




num_iter = 800000

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
        lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, forget_bias=1., state_is_tuple=True)
    else:
        lstm_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)

    # initial_state = lstm_cell.zero_state(batch_size, tf.float32)

    # split up inputs to use for basic rnn function (batch represented as a list of vectors)
    inputs_list = tf.unstack(inputs, axis=1)
    with tf.variable_scope('Rnn'):
        lstm_outputs_timefirst, state = tf.nn.rnn(lstm_cell, inputs_list, sequence_length=input_lengths, dtype=tf.float32) #initial_state=initial_state)

    # reorder these (want batch_size * maxlen * len(all_chars))
    lstm_outputs = tf.transpose(lstm_outputs_timefirst, perm=[1,0,2])

    W = tf.Variable(np.random.random(size=(rnn_size, len(all_chars))), dtype=tf.float32)
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

    # TODO: verify that this uses the same lstm cell as the training data
    # split up inputs to use for basic rnn function (batch represented as a list of vectors)
    held_inputs_list = tf.unstack(held_inputs, axis=1)
    with tf.variable_scope('Rnn', reuse=True):
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

            readable_hypotheses = hypothesis_to_readable(train_hypothesis)

            # edit_dist_hypotheses = [x.replace('_','') for x in readable_hypotheses]
            # edits = [editdistance.eval(hyp, gold) for (hyp, gold) in zip(edit_dist_hypotheses, stringabbrs)]
            # print 'mean edit dist', sum(edits) / batch_size

            print zip(string_longs[next_batch_begin:next_batch_begin+batch_size],
                      string_abbrs[next_batch_begin:next_batch_begin+batch_size],
                      readable_hypotheses)

            if iternum % 100 == 0:
                held_cost_run, held_hypothesis_run = session.run([held_cost, held_hypotheses])
                mean_dev_cost = sum(held_cost_run) / len(held_cost_run)
                print 'mean dev cost', mean_dev_cost
                dev_file.write(str(mean_dev_cost) + '\n')
                dev_file.write(str(zip(held_string_longs,
                                   held_string_abbrs,
                                   readable_hypotheses)))
                dev_file.write('\n')
                dev_file.flush()


                readable_hypotheses = hypothesis_to_readable(held_hypothesis_run)

                print_first_x = 100
                print zip(held_string_longs[:print_first_x],
                          held_string_abbrs[:print_first_x],
                          readable_hypotheses[:print_first_x])

    # todo: save model parameters somehow and enable decoding
    costs_file.close()
    dev_file.close()
