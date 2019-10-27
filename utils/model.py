import random

import tensorflow as tf
import numpy as np

"""
Utility module for model.
Codes are heavily brought from `https://github.com/tomasjakab/imm`.
Many thanks for the great code.
"""


def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def get_n_colors(n, pastel_factor=0.9):
    colors = []
    for i in range(n):
        colors.append(generate_new_color(colors, pastel_factor=0.9))
    return colors


def colorize_point_maps(maps, colors):
    n_maps = maps.shape.as_list()[-1]
    hmaps = [tf.expand_dims(maps[..., i], axis=3) * np.reshape(colors[i], [1, 1, 1, 3])
             for i in range(n_maps)]
    return tf.reduce_max(hmaps, axis=0)


def get_gaussian_maps(mu, shape_hw, inv_std=14.3):
    mu_x, mu_y = mu[:, :, 0:1], mu[:, :, 1:2]
    y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))
    x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))
    mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)
    y = tf.reshape(y, [1, 1, shape_hw[0], 1])
    x = tf.reshape(x, [1, 1, 1, shape_hw[1]])
    g_y = tf.square(y - mu_y)
    g_x = tf.square(x - mu_x)
    dist = (g_y + g_x) * inv_std ** 2
    g_yx = tf.transpose(tf.exp(-dist), perm=[0, 2, 3, 1])
    return g_yx


def get_coord(x, other_axis, axis_size):
    # get "x-y" coordinates:
    g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
    g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
    coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size))  # W
    coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
    g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
    return g_c, g_c_prob
