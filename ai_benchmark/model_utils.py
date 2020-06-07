# -*- coding: utf-8 -*-
# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

from tensorflow.python.ops import rnn_cell
from ai_benchmark.utils import tf, np


class DiagonalLSTMCell(rnn_cell.RNNCell):

    def __init__(self, hidden_dims, height, channel):
        self._num_unit_shards = 1
        self._forget_bias = 1.0

        self._height = height
        self._channel = channel

        self._hidden_dims = hidden_dims
        self._num_units = self._hidden_dims * self._height
        self._state_size = self._num_units * 2
        self._output_size = self._num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, i_to_s, state, scope="DiagonalBiLSTMCell"):

        c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
        h_prev = tf.slice(state, [0, self._num_units], [-1, self._num_units])

        with tf.compat.v1.variable_scope(scope):

            conv1d_inputs = tf.reshape(h_prev, [-1, self._height, 1, self._hidden_dims], name='conv1d_inputs')

            conv_s_to_s = conv1d(conv1d_inputs, 4 * self._hidden_dims, 2, scope='s_to_s')
            s_to_s = tf.reshape(conv_s_to_s, [-1, self._height * self._hidden_dims * 4])

            lstm_matrix = tf.sigmoid(s_to_s + i_to_s)

            i, g, f, o = tf.split(lstm_matrix, 4, 1)

            c = f * c_prev + i * g
            h = tf.multiply(o, tf.tanh(c), name='hid')

        new_state = tf.concat([c, h], 1)
        return h, new_state


def conv2d(inputs, num_outputs, kernel_shape, mask_type, scope="conv2d"):

    with tf.compat.v1.variable_scope(scope):

        WEIGHT_INITIALIZER = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        batch_size, height, width, channel = inputs.get_shape().as_list()

        kernel_h, kernel_w = kernel_shape

        center_h = kernel_h // 2
        center_w = kernel_w // 2

        weights_shape = [kernel_h, kernel_w, channel, num_outputs]
        weights = tf.compat.v1.get_variable("weights", weights_shape,
                                            tf.float32, WEIGHT_INITIALIZER, None)

        mask = np.ones((kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)

        mask[center_h, center_w + 1:, :, :] = 0.0
        mask[center_h + 1:, :, :, :] = 0.0

        if mask_type == 'a':
            mask[center_h, center_w, :, :] = 0.0

        weights = weights * tf.constant(mask, dtype=tf.float32)
        outputs = tf.nn.conv2d(input=inputs, filters=weights, strides=[1, 1, 1, 1], padding="SAME", name='outputs')

        biases = tf.compat.v1.get_variable("biases", [num_outputs, ], tf.float32, tf.compat.v1.zeros_initializer(), None)
        outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

        return outputs


def conv1d(inputs, num_outputs, kernel_size, scope="conv1d"):

    with tf.compat.v1.variable_scope(scope):

        WEIGHT_INITIALIZER = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        batch_size, height, _, channel = inputs.get_shape().as_list()

        kernel_h, kernel_w = kernel_size, 1

        weights_shape = [kernel_h, kernel_w, channel, num_outputs]
        weights = tf.compat.v1.get_variable("weights", weights_shape, tf.float32, WEIGHT_INITIALIZER, None)

        outputs = tf.nn.conv2d(input=inputs, filters=weights, strides=[1, 1, 1, 1], padding="SAME", name='outputs')

        biases = tf.compat.v1.get_variable("biases", [num_outputs,], tf.float32, tf.compat.v1.zeros_initializer(), None)
        outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

        return outputs


def skew(inputs, scope="skew"):

    with tf.compat.v1.name_scope(scope):

        batch, height, width, channel = inputs.get_shape().as_list()
        rows = tf.split(inputs, height, 1)

        new_width = width + height - 1
        new_rows = []

        for idx, row in enumerate(rows):
            transposed_row = tf.transpose(tf.squeeze(row, [1]), perm=[0, 2, 1])
            squeezed_row = tf.reshape(transposed_row, [-1, width])
            padded_row = tf.pad(squeezed_row, paddings=((0, 0), (idx, height - 1 - idx)))

            unsqueezed_row = tf.reshape(padded_row, [-1, channel, new_width])
            untransposed_row = tf.transpose(unsqueezed_row, perm=[0, 2, 1])
            new_rows.append(untransposed_row)

        outputs = tf.stack(new_rows, axis=1, name="output")

    return outputs


def unskew(inputs, width=None, scope="unskew"):

    with tf.compat.v1.name_scope(scope):

        batch, height, skewed_width, channel = inputs.get_shape().as_list()
        width = width if width else height

        new_rows = []
        rows = tf.split(inputs, height, 1)

        for idx, row in enumerate(rows):
            new_rows.append(tf.slice(row, [0, 0, idx, 0], [-1, -1, width, -1]))

        outputs = tf.concat(new_rows, 1, name="output")

    return outputs


def diagonal_lstm(inputs, scope='diagonal_lstm'):
    with tf.compat.v1.variable_scope(scope):

        skewed_inputs = skew(inputs, scope="skewed_i")

        input_to_state = conv2d(skewed_inputs, 64, [1, 1], mask_type="b", scope="i_to_s")
        column_wise_inputs = tf.transpose(input_to_state, perm=[0, 2, 1, 3])

        batch, width, height, channel = column_wise_inputs.get_shape().as_list()
        rnn_inputs = tf.reshape(column_wise_inputs, [-1, width, height * channel])

        cell = DiagonalLSTMCell(16, height, channel)

        outputs, states = tf.compat.v1.nn.dynamic_rnn(cell, inputs=rnn_inputs, dtype=tf.float32)
        width_first_outputs = tf.reshape(outputs, [-1, width, height, 16])

        skewed_outputs = tf.transpose(width_first_outputs, perm=[0, 2, 1, 3])
        outputs = unskew(skewed_outputs)

        return outputs


def diagonal_bilstm(inputs, scope='diagonal_bilstm'):
    with tf.compat.v1.variable_scope(scope):

        def reverse(inputs):
            return tf.reverse(inputs, [2])

        output_state_fw = diagonal_lstm(inputs, scope='output_state_fw')
        output_state_bw = reverse(diagonal_lstm(reverse(inputs), scope='output_state_bw'))

        batch, height, width, channel = output_state_bw.get_shape().as_list()

        output_state_bw_except_last = tf.slice(output_state_bw, [0, 0, 0, 0], [-1, height-1, -1, -1])
        output_state_bw_only_last = tf.slice(output_state_bw, [0, height-1, 0, 0], [-1, 1, -1, -1])
        dummy_zeros = tf.zeros_like(output_state_bw_only_last)

        output_state_bw_with_last_zeros = tf.concat([output_state_bw_except_last, dummy_zeros], 1)

        return output_state_fw + output_state_bw_with_last_zeros

