import tensorflow as tf
import numpy as np

# st = tf.SparseTensor([[0,0,2],[1,1,1],[2,2,0]], [1,1,1], shape=[3,3,3])
# # st = tf.SparseTensor(indices=[[0],[3],[5]], values=[1,1,1], shape=[6])
#
# with tf.Session() as session:
#     print session.run(st)
#     print session.run(
#         tf.sparse_tensor_to_dense(st)
#     )
#     dense = tf.sparse_tensor_to_dense(st)
#     nonzeros = tf.count_nonzero(dense)
#
#     print session.run([
#         dense,
#         nonzeros
#         ]
#     )
#
#     stuff = tf.constant([[0,0,2],[1,1,1],[2,2,0],[0,1,6]])
#     print session.run([
#         tf.squeeze(tf.slice(stuff, [0,0], [-1,2])),
#         tf.squeeze(tf.slice(stuff, [0,2], [-1,1]))
#     ])

#
# tens = tf.Variable(np.array([[[1, 0, 0],[0, 1, 0]],[[0,0,1],[1,0,.5]],
#                              [[0, 1, .5], [0, 0, 1]], [[0, 0, 1], [0, 1, 0]]]), tf.float32)
#
# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     print session.run(tens)
#     print session.run(
#         tf.argmax(tens, dimension=2)
#     )
#     lets = session.run(tf.argmax(tens, dimension=2))
#     for x in lets:
#         phrase = ''.join(['a'*l for l in x])
#         print phrase


inputs = np.array([[
  [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
  [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,]]], dtype=np.float32)

inputs *= 10

labels_indices = np.array([[0,0],[0,1]], dtype=np.int32)
labels_values = np.array([4, 2], dtype=np.int32)
labels = tf.SparseTensor(labels_indices, labels_values, inputs.shape[:2])



# labels = tf.SparseTensor([[0,0], [0,2], [0,3], [1,0], [1,1]], [0, 1, 3, 2, 2], [2,5])
# inputs = np.array([[[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,0,0,1]],
#                    [[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,1],[0,0,0,0,1]]], dtype=np.float32)


lengths = [10]

# lengths = [3, 2]

loss = tf.nn.ctc_loss(inputs, labels, lengths, time_major=False, ctc_merge_repeated=False)

decoder_inputs = inputs.transpose(1,0,2)
decoder = tf.nn.ctc_greedy_decoder(decoder_inputs, lengths)

with tf.Session() as session:
    print session.run(tf.sparse_tensor_to_dense(labels))
    print session.run(tf.arg_max(tf.nn.softmax(inputs), dimension=2))
    print session.run(decoder)
    print session.run(loss)
