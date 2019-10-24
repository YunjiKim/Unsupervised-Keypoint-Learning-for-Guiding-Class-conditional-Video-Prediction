import tensorflow as tf


def conv(x, channels, kernel=4, stride=2, pad=0, use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        x = tf.layers.conv2d(inputs=x, filters=channels, padding='same', kernel_size=kernel, \
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), strides=stride,
                             use_bias=use_bias)
        return x


def batch_norm(x, train_mode, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x, epsilon=1e-05, center=True, scale=True, scope=scope, is_training=train_mode)


def lstm_model(layers):
    lstm_cells = [tf.nn.rnn_cell.LSTMCell(units, name='basic_lstm_cell', state_is_tuple=True) for units in layers]
    lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0) for cell in lstm_cells]
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
    return stacked_lstm


def to_coord(input_, input_size, output_size, stddev=0.02, bias_start=0.0):
    with tf.variable_scope("fully_connected", reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [input_size, output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(bias_start))
        return tf.tanh(tf.matmul(input_, W) + b)
