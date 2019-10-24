from abc import ABC, abstractmethod
from os import path as osp

import tensorflow as tf


class BaseModel(ABC):
    """
    Base training model. All models which will inherit this class.
    Abstract base class.
    """

    # this variable should be set in child class
    name = 'base_model'
    trainable = True

    def __init__(self, is_training=True):
        super(BaseModel, self).__init__()
        self.is_training = is_training
        self.log_dir = None
        self.train_writer = None
        self.test_writer = None
        self.saver = None
        pass

    @abstractmethod
    def build(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def train_step(self,
                   sess,
                   feed_dict,
                   step,
                   batch_size,
                   should_write_log=False, should_write_summary=False):
        """
        Train step function. It defines how one train step will proceed.
        It should includes forward, optimization, and any logging if needed.
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, sess, feed_dict, step, test_idx, batch_size):
        """
        Test step function. It defines how one test step will proceed.
        Any returned values will be collected as a list,
        and will be passed to `collect_test_results` method.
        :return: any
        """
        raise NotImplementedError

    @abstractmethod
    def collect_test_results(self, results, step):
        """
        Further processing of test results if needed.
        :param results: list of return values of test_step function.
        :param step: global step
        """
        raise NotImplementedError

    def initialize_loggers(self, log_dir, sess):
        """
        Initializes summary_writer, checkpoint saver
        :param log_dir:
        :param sess:
        """
        name = self.__class__.name
        self.log_dir = log_dir

        if self.is_training:
            self.train_writer = tf.summary.FileWriter(osp.join(log_dir, name, 'train'), sess.graph)
            self.test_writer = tf.summary.FileWriter(osp.join(log_dir, name, 'test'))
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        pass

    def save_checkpoint(self, sess, step):
        name = self.__class__.name
        checkpoint_path = osp.join(self.log_dir, name, 'model.ckpt')
        self.saver.save(sess, checkpoint_path, global_step=step)
        pass

    def restore(self, sess, checkpoint_path):
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        vars_to_restore = tf.global_variables()
        checkpoint_vars = reader.get_variable_to_shape_map().keys()
        vars_to_restore = [v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
        print('vars-to-RESTORE:')
        print('\n'.join([v.name for v in vars_to_restore]))
        restorer = tf.train.Saver(var_list=vars_to_restore)
        restorer.restore(sess, checkpoint_path)
        pass
