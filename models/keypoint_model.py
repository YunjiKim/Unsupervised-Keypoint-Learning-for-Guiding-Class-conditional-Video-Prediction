import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from . import networks
from utils import model as model_utils


class KeypointModel(BaseModel):
    """
    Models for extracting keypoints.
    This model is not trainable.
    """

    name = 'stage1'

    def __init__(self, config):
        super(KeypointModel, self).__init__(False)

        # configuration variables
        model_config = config['model']
        paths_config = config['paths']

        self.n_points = model_config['n_pts']
        self.log_dir = paths_config['log_dir']

        self.colors = model_utils.get_n_colors(model_config['n_pts'], pastel_factor=0.0)

        # inputs
        self.input_im = None
        self.input_idx = None
        self.input_len = None

        # outputs
        self.output_ops = None
        self.points = None
        pass

    def build(self, inputs):
        # input setup
        self.input_im = inputs['image']
        self.input_idx = inputs['idx']
        self.input_len = inputs['len']

        self.points = networks.pose_encoder(tf.reshape(self.input_im, [-1, 128, 128, 3]),
                                            self.n_points,
                                            False)
        self.output_ops = {
            'pts': tf.reshape(self.points, [-1, 663, 40, 2]),
            'idx': self.input_idx,
            'len': self.input_len,
            'im': self.input_im
        }
        pass

    def run(self, sess, feed_dict):
        return sess.run(self.output_ops, feed_dict=feed_dict)

    def train_step(self,
                   sess,
                   feed_dict,
                   step,
                   batch_size,
                   should_write_log=False, should_write_summary=False):
        """
        This model is not trainable
        """
        raise NotImplementedError

    def test_step(self, sess, feed_dict, step, test_idx, batch_size):
        """
        This model has no test step
        """
        raise NotImplementedError

    def collect_test_results(self, results, step):
        """
        This model has no test step
        """
        raise NotImplementedError
