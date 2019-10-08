from __future__ import print_function
from __future__ import absolute_import

import tensorflow.contrib.slim as slim
import tensorflow as tf
import os.path as osp

# network definition:
from unsup_combine.models.imm_model import IMMModel
from unsup_combine.utils.box import Box
from unsup_combine.utils.colorize import colorize
from unsup_combine.datasets import penn_dataset_center

import argparse
import metayaml

import subprocess as sp


config = Box(metayaml.read(['configs/experiments/penn-40pts.yaml']))
train_config = config.training
# data_dir = train_config.datadir
train_config.datadir = '/dataset/imm/penn'
NUM_STEPS = 30000000

session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True
    
# open session
with tf.Session(config=session_config) as sess:
    
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # dynamic import of a dataset class
    dset_class = penn_dataset_center.PENNDataset

    train_dset = dset_class(train_config.datadir, subset='train',\
                            n_action=opts['n_action'], n_landmark=opts['n_landmark'], order_stream= True)
    train_dset = train_dset.get_dataset(opts['batch_size'], repeat=False, shuffle=False, num_preprocess_threads=12)

    test_dset = dset_class(train_config.datadir, subset='test',\
                           n_action=opts['n_action'], n_landmark=opts['n_landmark'], order_stream= True, max_samples=1000)
    test_dset = test_dset.get_dataset(opts['batch_size'], repeat=False, shuffle=False, num_preprocess_threads=12)
    
    # set up inputs
    handle_pl = tf.placeholder(tf.string, shape=[])
    base_iterator = tf.data.Iterator.from_string_handle(handle_pl, train_dset.output_types, train_dset.output_shapes)
    inputs = base_iterator.get_next()
    
    model = IMMModel(sess, config, opts['cell_info'], opts['vae_dim'], global_step, training=False)
    model.build(inputs)

    # show network architecture
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    output_ = model.evaluate(opts, test_dset, test_dset, handle_pl, checkpoint_fname)

    
    
