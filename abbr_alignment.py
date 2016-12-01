"""

Learn to map phrases to their abbreviations


some notes from implementation:
    - be sure there is no extra activation of rnn cell output and no activation on the layer immediately preceding ctc loss calculation
    - be sure the sequence_length is ALWAYS length of the long form, not abbr


"""

from __future__ import division
import tensorflow as tf
import numpy as np
import logging
import random
import editdistance

__author__ = 'gpfinley'

use_lrabr = True
EMPTY = '_'
use_house_decoder = True
scrub_abbr_periods = True
case_sensitive = True

logging.basicConfig(level=logging.DEBUG)

if use_lrabr:

    skip_between_every = 0
    total_to_use = 1000
    # todo: free up enough memory to do all of them? right now it doesn't do longform duplicates, and quits after a few
    used_longs = set()
    string_longs = []
    string_abbrs = []
    # with open('/Volumes/gregdata/metathesaurus/2015AA/LEX/LRABR') as f:
    with open('/Users/gpfinley/LRABR') as f:
        used = 0
        skipcounter = 0
        for line in f:
            if skipcounter:
                skipcounter -= 1
                continue
            _, abbr, _, _, long, _ = line.split('|')
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

    # string_longs = ['word one',
    #                 'wrd oh',
    #                 'one wrd',
    #                 'wuh oh',
    #                 'word word',
    #                 'oy oy',
    #                 'we we',
    #                 'why oy',
    #                 'ohhh way',
    #                 'oh why']
    # string_abbrs = ['WO',
    #                 'WO',
    #                 'OW',
    #                 'WO',
    #                 'WW',
    #                 'OO',
    #                 'WW',
    #                 'WO',
    #                 'OW',
    #                 'OW']


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

print 'character to integer dictionary:', char2ind
print all_chars

# one-hot vectorization of long and short forms
# use the last character index, not zero, for padding the inputs (outputs need zero because we will use np.nonzero)
# X = np.zeros((len(long_abbr), maxlen, len(all_chars)), dtype=np.bool) + char2ind[EMPTY]
X = np.zeros((len(string_longs), maxlen, len(all_chars)), dtype=np.bool) + char2ind[EMPTY]
y = np.zeros((len(string_longs), maxlen, len(all_chars)), dtype=np.bool)

logging.info('building one-hot feature vectors...')
for i in range(len(string_longs)):
    for t, char in enumerate(string_longs[i]):
        X[i, t, char2ind[char]] = 1
    for t, char in enumerate(string_abbrs[i]):
        y[i, t, char2ind[char]] = 1
    np.set_printoptions(threshold=np.nan)
logging.info('built one-hot feature vectors.')

batch_counter = 0
batch_size = 100

class Align:

    def __init__(self):

        self.lr = .1

        rnn_size = 2 * len(all_chars)

        self.input_lengths = tf.placeholder(tf.int32, shape=[batch_size], name='input_lengths')
        # self.output_lengths = tf.placeholder(tf.int32, shape=[batch_size], name='output_lengths')

        self.inputs = tf.placeholder(tf.float32, shape=[batch_size, maxlen, len(all_chars)], name='inputs')

        # first dimension will be the number of nonzeros
        self.nonzero_label_indices = tf.placeholder(tf.int64, shape=[None, 3], name='labels')
        # get just the first two dimensions of each label (the batch number and time step)
        nonzeros = tf.squeeze(tf.slice(self.nonzero_label_indices, [0,0], [-1,2]))
        # get the third element of each label (the character values)
        chars = tf.squeeze(tf.slice(self.nonzero_label_indices, [0,2], [-1,1]))

        self.labels = tf.SparseTensor(indices=nonzeros, values=chars, shape=[batch_size, maxlen])
        self.labels = tf.to_int32(self.labels)

        # lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, forget_bias=1., state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)

        initial_state = lstm_cell.zero_state(batch_size, tf.float32)

        # split up inputs to use for basic rnn function (batch represented as a list of vectors)
        inputs_list = tf.unstack(self.inputs, axis=1)
        # todo: put lengths in
        lstm_outputs_timefirst, state = tf.nn.rnn(lstm_cell, inputs_list, sequence_length=self.input_lengths, initial_state=initial_state)

        # reorder these (want batch_size * maxlen * len(all_chars))
        lstm_outputs = tf.transpose(lstm_outputs_timefirst, perm=[1,0,2])

        # I think the lstm/rnn automatically applies an activation function...
        # lstm_outputs = tf.nn.tanh(lstm_outputs)

        self.W = tf.Variable(np.random.random(size=(rnn_size, len(all_chars))), dtype=tf.float32)
        b_array = np.zeros(len(all_chars), dtype=np.float32)
        # build in a little initial bias towards null
        b_array[-1] = .1
        b = tf.Variable(b_array, dtype=tf.float32)

        output_layer = tf.einsum('abi,ic->abc', lstm_outputs, self.W) + b
        # if no projection:
        # output_layer = lstm_outputs

        # todo: don't use sigmoid? not sure if ctc_loss's internal softmax wants sigmoided outputs or not
        # output_layer_logits = tf.sigmoid(output_layer)

        # # KLUDGE: pass gold in as a self var (preprocessed using numpy, not tf)
        # self.gold_output_layer = tf.placeholder(dtype=tf.float32, shape=(batch_size, maxlen, len(all_chars)))

        # create a variable of the highest index. subtract that index for all entries in the sparse matrix, then add back in the values at the same points
        # mask = tf.Variable(np.ones((batch_size, maxlen, len(all_chars))) + len(all_chars), dtype=tf.int32, trainable=False)
        # charspresent = tf.sparse_to_dense(nonzeros, (batch_size, maxlen, len(all_chars)), tf.ones_like(chars, dtype=tf.int32))
        # mask = tf.subtract(mask, charspresent)
        # gold_output_layer = tf.add(mask, tf.sparse_tensor_to_dense(labels))

        # gold_output_layer = tf.sparse_tensor_to_dense(labels)

        # NOTE: I don't think that abbr lengths are right to use here! Should use input lengths of original sequences, I guess?
        # self.gold_cost = tf.nn.ctc_loss(self.gold_output_layer, self.labels, self.output_lengths, ctc_merge_repeated=False, time_major=False)
        # self.cost = tf.nn.ctc_loss(output_layer_logits, self.labels, self.output_lengths, ctc_merge_repeated=False, time_major=False)
        # self.gold_cost = tf.nn.ctc_loss(self.gold_output_layer, self.labels, self.input_lengths, ctc_merge_repeated=False, time_major=False)
        self.cost = tf.nn.ctc_loss(output_layer, self.labels, self.input_lengths, ctc_merge_repeated=False, time_major=False)


        self.latest_state = state

        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # optimizer = tf.train.AdagradOptimizer(self.lr)
        optimizer = tf.train.AdadeltaOptimizer(self.lr)               # needs high learning rate (~100k X momentum rate)
        # optimizer = tf.train.MomentumOptimizer(self.lr, .9)             # needs very low learning rate

        grads_and_vars = optimizer.compute_gradients(self.cost, tf.trainable_variables())
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        # todo: how is this actually working?
        # grads_and_vars = zip(tf.gradients(self.cost, tf.trainable_variables()), tf.trainable_variables())
        # self.train_op = optimizer.apply_gradients(grads_and_vars)

        if use_house_decoder:
            # # for decoding only: softmax activation of output layer, then argmax to find character index
            # #       (softmax is done automatically when tensorflow calculates the ctc loss)
            # softmax = tf.nn.softmax(output_layer_logits)
            # self.char_hypotheses = tf.arg_max(softmax, 2)
            self.char_hypotheses = tf.arg_max(output_layer, 2)
        else:
            # can also decode with this, although it makes it harder to see the output at each time
            self.char_hypotheses = tf.nn.ctc_greedy_decoder(tf.transpose(output_layer, perm=[1,0,2]),
                                                            [maxlen] * batch_size,
                                                            merge_repeated=False)


        if 1: return
        # debug:
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print lstm_outputs.get_shape()
            print self.W.get_shape()
            print output_layer.get_shape()
            # session.run(tf.Print(lstm_outputs.get_shape()))
            # session.run(tf.Print(output_layer.get_shape()))
            print 'init state:', session.run(initial_state)
            print 'trainable var names:', [v.name for v in tf.trainable_variables()]
            # print 'trainable vars:', session.run(tf.trainable_variables())
            # print 'trainable var sizes:', session.run([v.get_shape() for v in tf.trainable_variables()])
            # session.run(tf.global_variables_initializer())
            # nextlong, nextabbr = get_next_batch()
            # print session.run(self.cost, feed_dict={self.nonzero_label_indices: nextabbr, self.inputs: nextlong})
            # print session.run(tf.shape(lstm_outputs), feed_dict={self.nonzero_label_indices: nextabbr, self.inputs: nextlong})
            # print session.run(self.cost, feed_dict={self.sparse_labels: nextabbr, self.inputs: nextlong})



# return a tuple with (longform_labels, abbr_labels, longforms, abbrs)
def get_next_batch():
    global batch_counter
    # print 'on batch', batch_counter
    start = batch_counter * batch_size
    end = (batch_counter+1) * batch_size
    if end > len(X):
        batch_counter = 0
        start = 0
        end = batch_size
    nextlongs = X[start:end]
    # get abbrs in the form of all nonzero elements (was all one-hot vectors before)
    nextabbrs = np.transpose(np.nonzero(y[start:end]))
    # todo: temp: don't update the batch counter; simulate multiple epochs per iteration (see if it can overfit a small set)
    batch_counter += 1
    return nextlongs, nextabbrs, string_longs[start:end], string_abbrs[start:end]


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



def main(_):
    # rand_init_scale = .1

    num_iter = 800000

    costs_file = open('costs.txt', 'w')

    with tf.Graph().as_default():
            # initializer = tf.random_uniform_initializer(-rand_init_scale, rand_init_scale)
            # with tf.variable_scope('Train', reuse=None, initializer=initializer):
            logging.info('building tensorflow graph...')
            m = Align()
            logging.info('built tensorflow graph.')

        # with tf.variable_scope('Model', reuse=None, initializer=initializer):
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                for iternum in range(num_iter):

                    nextlong, nextabbr, stringlongs, stringabbrs = get_next_batch()
                    if nextlong is None:
                        print 'no more batches available'
                        break

                    input_lengths = [len(x) for x in stringlongs]

                    # this is probably not necessary (delete!):
                    # gold_output_layer = np.zeros((batch_size, maxlen, len(all_chars)), dtype=np.float32)
                    # gold_output_layer[:,:,len(all_chars)-1] = 1
                    # for (indx, indy, val) in nextabbr:
                    #     gold_output_layer[indx, indy, val] = 1
                    #     gold_output_layer[indx, indy, len(all_chars)-1] = 0

                    feed_dict={m.inputs: nextlong,
                               m.input_lengths : input_lengths,
                               m.nonzero_label_indices: nextabbr,
                               }

                    cost, state, _, hypothesis = session.run([m.cost, m.latest_state, m.train_op, m.char_hypotheses],
                                                             feed_dict=feed_dict)

                    print 'costs', cost
                    mean_cost = sum(cost) / batch_size
                    print 'mean cost', mean_cost
                    costs_file.write(str(mean_cost) + '\n')
                    if iternum % 10 == 0:
                        costs_file.flush()

                    readable_hypotheses = hypothesis_to_readable(hypothesis)
                    edit_dist_hypotheses = [x.replace('_','') for x in readable_hypotheses]
                    edits = [editdistance.eval(hyp, gold) for (hyp, gold) in zip(edit_dist_hypotheses, stringabbrs)]
                    print 'mean edit dist', sum(edits) / batch_size

                    print zip(stringlongs, stringabbrs, readable_hypotheses)



if __name__ == "__main__":
    tf.app.run()

