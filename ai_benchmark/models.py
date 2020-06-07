# -*- coding: utf-8 -*-
# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

from ai_benchmark.utils import tf
from ai_benchmark.model_utils import *


def LSTM_Sentiment(input_tensor):

    #  Reference Paper: https://www.bioinf.jku.at/publications/older/2604.pdf

    lstmCell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(1024)
    output_rnn, _ = tf.compat.v1.nn.dynamic_rnn(lstmCell, input_tensor, dtype=tf.float32)

    W_fc = tf.Variable(tf.random.truncated_normal([1024, 2]))
    b_fc = tf.Variable(tf.constant(0.1, shape=[2]))

    output_transposed = tf.transpose(output_rnn, perm=[1, 0, 2])
    output = tf.gather(output_transposed, int(output_transposed.get_shape()[0]) - 1)

    return tf.identity(tf.matmul(output, W_fc) + b_fc, name="output")


def PixelRNN(inputs):

    #  Reference Paper: https://arxiv.org/abs/1601.06759
    #  Reference Code: https://github.com/carpedm20/pixel-rnn-tensorflow

    normalized_inputs = inputs / 255.0
    output = conv2d(normalized_inputs, 16, [7, 7], mask_type="a", scope="conv_inputs")

    for idx in range(7):
        output = diagonal_bilstm(output, scope='LSTM%d' % idx)

    for idx in range(2):
        output = tf.nn.relu(conv2d(output, 32, [1, 1], mask_type="b", scope='CONV_OUT%d' % idx))

    conv2d_out_logits = conv2d(output, 3, [1, 1], mask_type="b", scope='conv2d_out_logits')
    return tf.nn.sigmoid(conv2d_out_logits) * 255
