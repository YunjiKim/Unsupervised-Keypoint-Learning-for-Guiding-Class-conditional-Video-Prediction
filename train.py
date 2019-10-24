from os import path as osp

import tensorflow as tf
from argparse import ArgumentParser

from models import DetectorTranslatorModel, MotionGeneratorModel
from data import ImagePairDataLoader, SequenceDataLoader
import utils
from utils import training as training_utils


def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str,
                        choices=['detector_translator', 'motion_generator'],
                        help='which mode to train')
    parser.add_argument('--config', type=str, help='path of the configuration file')
    args = parser.parse_args()

    config = utils.load_config(args.config)

    paths_config = config['paths']
    train_config = config['training']

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    # open tf session
    with tf.Session(config=session_config) as sess:
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # initializing datasets
        batch_size = train_config['batch_size']
        train_loader = _get_dataloader_by_mode(args.mode, 'train', config)
        test_loader = _get_dataloader_by_mode(args.mode, 'test', config)
        train_dataset = train_loader.get_dataset(batch_size,
                                                 repeat=True,
                                                 shuffle=True,
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
        model = _get_model_by_mode(args.mode, config, global_step)
        print('model initialized')
        model.build(inputs)

        # training config variables
        n_epochs = train_config['n_steps']
        summary_interval = train_config['summary_interval']
        test_interval = train_config['test_interval']
        checkpoint_interval = train_config['checkpoint_interval']
        log_interval = train_config['log_interval']

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

        # loggers initialization
        model.initialize_loggers(paths_config['log_dir'], sess)

        # main training loop start
        print('training start')
        start_step = sess.run(global_step)
        sess.run(train_iterator.initializer)

        for step in range(int(start_step), n_epochs):
            should_write_log = step % log_interval == 0
            should_write_summary = step % summary_interval == 0
            should_run_test = step % test_interval == 0
            should_save_checkpoint = step % checkpoint_interval == 0

            feed_dict = {handle_pl: train_handle, training_pl: True}
            model.train_step(sess, feed_dict, step, batch_size,
                             should_write_log=should_write_log,
                             should_write_summary=should_write_summary)

            if should_save_checkpoint:
                model.save_checkpoint(sess, step)

            if should_run_test:
                # running test

                n_test_iters = training_utils.get_n_iterations(test_loader.length(), batch_size)
                feed_dict = {handle_pl: test_handle, training_pl: False}
                sess.run(test_iterator.initializer)
                test_results = []

                for test_idx in range(n_test_iters):
                    result = model.test_step(sess, feed_dict, step, test_idx, batch_size)
                    test_results.append(result)
                    pass

                model.collect_test_results(test_results, step)

            pass
    pass


def _get_model_by_mode(mode, config, global_step):
    if mode == 'detector_translator':
        return DetectorTranslatorModel(config, global_step, is_training=True)
    if mode == 'motion_generator':
        return MotionGeneratorModel(config, global_step, is_training=True)
    else:
        raise Exception('unknown model %s' % mode)


def _get_dataloader_by_mode(mode, subset, config):
    is_train = subset == 'train'
    data_dir = config['paths']['data_dir']

    max_samples = None
    if is_train:
        max_samples = 1000

    if mode == 'detector_translator':
        return ImagePairDataLoader(data_dir, subset,
                                   random_order=is_train,
                                   randomness=is_train,
                                   max_samples=max_samples)
    elif mode == 'motion_generator':
        model_config = config['model']
        n_points = model_config['n_pts']
        n_action = model_config['n_action']

        return SequenceDataLoader(data_dir, subset,
                                  n_points=n_points, n_action=n_action,
                                  random_order=is_train,
                                  randomness=is_train,
                                  max_samples=max_samples)
    else:
        raise Exception('unknown dataloader %s' % mode)


if __name__ == '__main__':
    main()
