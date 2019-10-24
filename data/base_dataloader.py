from abc import ABC, abstractmethod

import tensorflow as tf


class BaseDataLoader(ABC):

    def __init__(self):
        super(BaseDataLoader, self).__init__()
        pass

    @abstractmethod
    def length(self):
        raise NotImplementedError

    @abstractmethod
    def get_sample_dtype(self):
        raise NotImplementedError

    @abstractmethod
    def get_sample_shape(self):
        raise NotImplementedError

    @abstractmethod
    def sample_generator(self):
        raise NotImplementedError

    @abstractmethod
    def map_fn(self, inputs):
        raise NotImplementedError

    def get_dataset(self,
                    batch_size,
                    repeat=False, shuffle=False,
                    num_preprocess_threads=12, prefetch=True):
        def sample_generator_fn():
            return self.sample_generator()

        sample_dtype = self.get_sample_dtype()
        sample_shape = self.get_sample_shape()
        dataset = tf.data.Dataset.from_generator(sample_generator_fn, sample_dtype, sample_shape)

        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(2000)

        dataset = dataset.map(self.map_fn, num_parallel_calls=num_preprocess_threads)
        dataset = dataset.batch(batch_size)

        if prefetch:
            dataset = dataset.prefetch(1)

        return dataset
