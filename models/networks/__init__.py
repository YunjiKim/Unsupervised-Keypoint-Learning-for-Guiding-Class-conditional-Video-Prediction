import tensorflow as tf

from . import layers
from .vgg import Vgg19
import utils.model as model_utils

def encoder(x, train_mode, filters=32):
    with tf.variable_scope('encoder'):
        block_features = []
        x = layers.conv(x, filters, kernel=7, stride=1, scope='conv_1')
        x = layers.batch_norm(x, train_mode, scope='b_norm_1')
        x = tf.nn.relu(x)
        x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_2')
        x = layers.batch_norm(x, train_mode, scope='b_norm_2')
        x = tf.nn.relu(x)
        block_features.append(x)
        for i in range(3):
            filters *= 2
            x = layers.conv(x, filters, kernel=3, stride=2, scope='conv_%d' % (i * 2 + 3))
            x = layers.batch_norm(x, train_mode, scope='b_norm_%d' % (i * 2 + 3))
            x = tf.nn.relu(x)
            x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d' % (i * 2 + 4))
            x = layers.batch_norm(x, train_mode, scope='b_norm_%d' % (i * 2 + 4))
            x = tf.nn.relu(x)
            block_features.append(x)
        return block_features


def image_encoder(x, train_mode):
    with tf.variable_scope('image_encoder'):
        block_features = encoder(x, train_mode)
        block_features = [x] + block_features
        return block_features


def pose_encoder(x, n_pts, train_mode, final_res=128, filters=128):
    with tf.variable_scope('pose_encoder', reuse=tf.AUTO_REUSE):
        block_features = encoder(x, train_mode)
        x = block_features[-1]
        size = x.shape.as_list()[1:3]
        conv_id = 1
        for i in range(4):
            if i > 0:
                x = layers.conv(tf.concat([x, block_features[-1 * (i + 1)]], axis=-1), filters, kernel=3, stride=1,
                         scope='conv_%d_0' % conv_id)
            else:
                x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d_0' % conv_id)
            x = layers.batch_norm(x, train_mode, scope='b_norm_%d_0' % conv_id)
            x = tf.nn.relu(x)
            x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d_1' % conv_id)
            x = layers.batch_norm(x, train_mode, scope='b_norm_%d_1' % conv_id)
            x = tf.nn.relu(x)
            if size[0] == final_res:
                x = layers.conv(x, n_pts, kernel=1, stride=1)
                break
            else:
                x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d_0' % (conv_id + 1))
                x = layers.batch_norm(x, train_mode, scope='b_norm_%d_0' % (conv_id + 1))
                x = tf.nn.relu(x)
                x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d_1' % (conv_id + 1))
                x = layers.batch_norm(x, train_mode, scope='b_norm_%d_1' % (conv_id + 1))
                x = tf.nn.relu(x)
                x = tf.image.resize_images(x, [2 * s for s in size])
            size = x.shape.as_list()[1:3]
            conv_id += 2
            if filters >= 8: filters /= 2

        xshape = x.shape.as_list()
        gauss_y, gauss_y_prob = model_utils.get_coord(x, 2, xshape[1])  # B,NMAP
        gauss_x, gauss_x_prob = model_utils.get_coord(x, 1, xshape[2])  # B,NMAP
        gauss_mu = tf.stack([gauss_x, gauss_y], axis=2)
        return gauss_mu


def translator(x, train_mode, final_res=128, filters=256):
    with tf.variable_scope('translator'):
        size = x.shape.as_list()[1:3]
        conv_id = 1
        while size[0] <= final_res:
            x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d_0' % conv_id)
            x = layers.batch_norm(x, train_mode, scope='b_norm_%d_0' % conv_id)
            x = tf.nn.relu(x)
            x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d_1' % conv_id)
            x = layers.batch_norm(x, train_mode, scope='b_norm_%d_1' % conv_id)
            x = tf.nn.relu(x)
            if size[0] == final_res:
                crude_output = layers.conv(x, 3, kernel=3, stride=1, scope='conv_%d_0' % (conv_id + 1))
                mask = layers.conv(x, 1, kernel=3, stride=1, scope='conv_%d_1' % (conv_id + 1))
                mask = tf.nn.sigmoid(mask)
                break
            else:
                x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d_0' % (conv_id + 1))
                x = layers.batch_norm(x, train_mode, scope='b_norm_%d_0' % (conv_id + 1))
                x = tf.nn.relu(x)
                x = layers.conv(x, filters, kernel=3, stride=1, scope='conv_%d_1' % (conv_id + 1))
                x = layers.batch_norm(x, train_mode, scope='b_norm_%d_1' % (conv_id + 1))
                x = tf.nn.relu(x)
                x = tf.image.resize_images(x, [2 * s for s in size])
            size = x.shape.as_list()[1:3]
            conv_id += 2
            if filters >= 8: filters /= 2
    return crude_output, mask


def vae_encoder(x, f_pt, act_code, cell_info, vae_dim):
    with tf.variable_scope('vae_encoder'):
        cell = layers.lstm_model(cell_info)
        state = cell.zero_state(tf.shape(x)[0], tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)
        logit = tf.contrib.layers.fully_connected(tf.concat([outputs[:, -1, :], f_pt, act_code], axis=-1), vae_dim * 2)
        mu = logit[:, :vae_dim]
        stddev = logit[:, vae_dim:]
        return mu, stddev


def vae_decoder(x, f_pt, act_code, cell_info, vae_dim, n_pts):
    with tf.variable_scope('vae_decoder', reuse=tf.AUTO_REUSE):
        cell = layers.lstm_model(cell_info)
        state = cell.zero_state(tf.shape(x)[0], tf.float32)
        input_ = tf.contrib.layers.fully_connected(tf.concat([x, f_pt, act_code], axis=-1), 32)
        empty_input = tf.zeros_like(input_)
        outputs = []
        output, state = cell(input_, state)
        outputs.append(tf.expand_dims(layers.to_coord(output, cell_info[-1], n_pts * 2), axis=1))
        for i in range(31):
            output, state = cell(empty_input, state)
            outputs.append(tf.expand_dims(layers.to_coord(output, cell_info[-1], n_pts * 2), axis=1))
        outputs = tf.concat(outputs, axis=1)
        return outputs


def seq_discr(x):
    with tf.variable_scope('seq_discr', reuse=tf.AUTO_REUSE):
        cell = layers.lstm_model([1024, 1024])
        state = cell.zero_state(tf.shape(x)[0], tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)
        logit = tf.contrib.layers.fully_connected(outputs, 1)
        return logit[:, -1, :]


def img_discr(x):
    with tf.variable_scope('img_discr', reuse=tf.AUTO_REUSE):
        channel = 64
        x = layers.conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_0')
        x = tf.nn.leaky_relu(x, 0.01)
        for i in range(1, 6):
            x = layers.conv(x, channel * 2, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_' + str(i))
            x = tf.nn.leaky_relu(x, 0.01)
            channel = channel * 2
        logit = layers.conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, scope='D_logit')
        return logit
