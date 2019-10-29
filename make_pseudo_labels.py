from os import path as osp

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

import utils
from models import KeypointModel
from data import KeypointDataLoader


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path of the configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path of the pretrained keypoints detector')
    args = parser.parse_args()

    config = utils.load_config(args.config)
    paths_config = config['paths']
    data_dir = paths_config['data_dir']
    batch_size = 1
    keypoints_root_dir = osp.join(data_dir, 'pseudo_labels')

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    # directory setup
    utils.touch_dir(keypoints_root_dir)

    if not tf.gfile.Exists(args.checkpoint) and not tf.gfile.Exists(args.checkpoint + '.index'):
        raise Exception('checkpoint not found at %s' % args.checkpoint)

    # start session
    with tf.Session(config=session_config) as sess:
        # import dataset
        train_loader = KeypointDataLoader(data_dir, 'train')
        test_loader = KeypointDataLoader(data_dir, 'test')
        train_dataset = train_loader.get_dataset(batch_size,
                                                 repeat=False,
                                                 shuffle=False,
                                                 num_preprocess_threads=12)
        test_dataset = test_loader.get_dataset(batch_size,
                                               repeat=False,
                                               shuffle=False,
                                               num_preprocess_threads=12)

        # setup inputs
        training_pl = tf.placeholder(tf.bool)
        handle_pl = tf.placeholder(tf.string, shape=[])
        base_iterator = tf.data.Iterator.from_string_handle(handle_pl, train_dataset.output_types,
                                                            train_dataset.output_shapes)
        inputs = base_iterator.get_next()

        # initializing models
        model = KeypointModel(config)
        print('model initialized')
        model.build(inputs)

        # variables initialization
        tf.logging.set_verbosity(tf.logging.INFO)
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        sess.run([global_init, local_init])

        # data iterator initialization
        train_iterator = train_dataset.make_initializable_iterator()
        test_iterator = test_dataset.make_initializable_iterator()
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        # checkpoint restoration
        model.restore(sess, args.checkpoint)

        # iterator initialization
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)

        # running on train dataset
        n_iters = utils.training.get_n_iterations(train_loader.length(), batch_size)
        feed_dict = {handle_pl: train_handle, training_pl: False}
        for _ in range(n_iters):
            outputs = model.run(sess, feed_dict)
            _save_output(keypoints_root_dir, outputs)
            pass
        print('iteration through train set finished')

        # running on test dataset
        n_iters = utils.training.get_n_iterations(test_loader.length(), batch_size)
        feed_dict = {handle_pl: test_handle, training_pl: False}
        for _ in range(n_iters):
            outputs = model.run(sess, feed_dict)
            _save_output(keypoints_root_dir, outputs)
            pass
        print('iteration through test set finished')
    pass


def _save_output(keypoints_root_dir, outputs):
    np.save(osp.join(keypoints_root_dir, '{:04d}.npy'.format(outputs['idx'][0])),
            outputs['pts'][0, :outputs['len'][0], ::])
    pass


if __name__ == '__main__':
    main()
