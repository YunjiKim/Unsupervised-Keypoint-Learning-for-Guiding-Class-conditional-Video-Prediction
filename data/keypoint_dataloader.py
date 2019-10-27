from os import path as osp
import os

import tensorflow as tf
import numpy as np
from PIL import Image

from .base_dataloader import BaseDataLoader
from utils import data as data_utils

# TODO: make image size as configurable parameter
IMAGE_SIZE = 128
MIN_IMAGE_SEQ_LEN = 663


class KeypointDataLoader(BaseDataLoader):

    def __init__(self, data_dir, subset, max_samples=None):
        super(KeypointDataLoader, self).__init__()

        self._data_dir = data_dir
        self._max_samples = max_samples

        with open(osp.join(data_dir, subset + '_set.txt'), 'r') as f:
            self._images = f.read().splitlines()

        self._total = len(self._images)
        if max_samples is not None:
            self._total = min(max_samples, len(self._images))
        print(subset + 'set : ', self._total)
        pass

    def length(self):
        return self._total

    def get_sample_shape(self):
        return {
            'image': [MIN_IMAGE_SEQ_LEN, 128, 128, 3],
            'len': None,
            'idx': None
        }

    def get_sample_dtype(self):
        return {
            'image': tf.float32,
            'len': tf.int16,
            'idx': tf.int16
        }

    def sample_generator(self):
        for idx in range(self._total):
            yield self._get_image_at(idx)

    def map_fn(self, inputs):
        with tf.name_scope('proc_im_pair'):
            return {
                'image': inputs['image'] * 2.0 - 1.0,
                'len': inputs['len'],
                'idx': inputs['idx']
            }

    def _get_image_at(self, idx):
        img_path, n_act = self._images[idx].split()
        file_len = len(os.listdir(osp.join(self._data_dir, img_path)))

        # load images
        image = Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(1) + '.jpg'))
        crop_size, ratio = data_utils.center_crop(image, IMAGE_SIZE)

        image_seq = []
        for i in range(file_len):
            im = Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(i + 1) + '.jpg'))
            im = im.resize([int(w / ratio), int(h / ratio)]).crop(crop_size)
            image_seq.append(np.expand_dims(im, 0))
            pass

        image_seq = np.concatenate(image_seq, axis=0)

        if file_len < MIN_IMAGE_SEQ_LEN:
            zero_im = np.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])
            im_pads = np.tile(zero_im, [663 - file_len, 1, 1, 1])
            image_seq = np.concatenate([image_seq, im_pads], axis=0)

        return {
            'image': image_seq / 255.0,
            'idx': int(img_path.split('/')[-1]),
            'len': file_len
        }
