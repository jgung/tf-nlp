from typing import List, Iterable, Union

import tensorflow as tf
from tensorflow.python.data.experimental import shuffle_and_repeat, bucket_by_sequence_length, AUTOTUNE
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.data.experimental import sample_from_datasets, choose_from_datasets

from tfnlp.common.constants import LENGTH_KEY


def _add_uniform_noise(value, amount):
    value = tf.cast(value, dtype=tf.float32)
    noise_value = value * tf.constant(amount, dtype=tf.float32)
    noise = tf.random.uniform(shape=[], minval=-noise_value, maxval=noise_value)
    return tf.cast(value + noise, dtype=tf.int32)


def _compute_dataset_weights(datasets):
    total = 0
    weights = []
    for path in datasets:
        count = float(sum(1 for _ in tf.python_io.tf_record_iterator(path)))
        total += count
        weights.append(count)
    return [weight / total for weight in weights]


def make_dataset(extractor,
                 paths: Union[str, Iterable],
                 batch_size: int = 16,
                 bucket_sizes: List[int] = None,
                 evaluate: bool = False,
                 num_parallel_calls: int = 8,
                 num_parallel_reads: int = 1,
                 max_epochs: int = -1,
                 length_noise_stdev: int = 0.1,
                 buffer_size: int = 100000,
                 batch_buffer_size: int = 512,
                 caching=True,
                 random_seed=None):
    if bucket_sizes is None:
        bucket_sizes = [5, 10, 25, 50, 100]
    if not isinstance(paths, Iterable) or isinstance(paths, str):
        paths = [paths]

    datasets = []
    for path in paths:
        dataset = tf.data.TFRecordDataset([path], num_parallel_reads=num_parallel_reads)

        if caching:
            dataset = dataset.cache()

        if not evaluate:
            # shuffle TF records
            dataset = shuffle_and_repeat(buffer_size=buffer_size, count=max_epochs, seed=random_seed)(dataset)

        # parse serialized TF records into dictionaries of Tensors for each feature
        dataset = dataset.map(extractor.parse, num_parallel_calls=num_parallel_calls)

        # bucket dataset by sequence length, applying random noise to sequences so we don't repeat the same buckets across epochs
        dataset = dataset.apply(bucket_by_sequence_length(element_length_func=lambda elem: _add_uniform_noise(elem[LENGTH_KEY],
                                                                                                              length_noise_stdev),
                                                          bucket_boundaries=bucket_sizes,
                                                          bucket_batch_sizes=(len(bucket_sizes) + 1) * [batch_size],
                                                          padded_shapes=extractor.get_shapes(),
                                                          padding_values=extractor.get_padding()))
        if not evaluate:
            # now sort bucketed batches -- maybe not efficient, but let's ensure our training set order is really random
            dataset = dataset.shuffle(batch_buffer_size, seed=random_seed)

        # seems to be generally set to 1 or 2
        dataset = dataset.prefetch(AUTOTUNE)
        datasets.append(dataset)

    if len(datasets) == 1:
        return DatasetV1Adapter(datasets[0])

    choice_dataset = tf.data.Dataset.range(len(datasets)).repeat()
    if evaluate:
        result = choose_from_datasets(datasets, choice_dataset)
    else:
        result = sample_from_datasets(datasets, weights=_compute_dataset_weights(paths), seed=random_seed)

    return DatasetV1Adapter(result)


def padded_batch(extractor, placeholder, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(placeholder)
    dataset = dataset.map(lambda x: extractor.parse(x, train=False))
    dataset = dataset.padded_batch(batch_size, extractor.get_shapes(train=False), extractor.get_padding(train=False))
    iterator = dataset.make_initializable_iterator()
    with tf.control_dependencies([iterator.initializer]):
        return iterator.get_next()
