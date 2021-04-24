import os
import traceback
import numpy as np
import tensorflow as tf

from Data.preprocessing import default_uint8_image_preprocessing

def _parse_func(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    img = tf.io.decode_raw(features['image_raw'], tf.float64)
    img = tf.reshape(img, shape=(32, 32, 3))
    label = tf.cast(features['label'], tf.int64)
    return img, label

def _load_tfrecord(record_path: str, batch_size: int, parallel_batches: int):
    assert os.path.exists(record_path)
    inputs = tf.data.TFRecordDataset(record_path)
    inputs = inputs.apply(tf.data.experimental.map_and_batch(_parse_func, num_parallel_calls=parallel_batches, batch_size=batch_size))

    return inputs


if __name__ == '__main__':
    iter = _load_tfrecord(r'D:\yuyan-workspace\tmp\cifar_100_train_records', batch_size=32, parallel_batches=16)
    numpy_iter = iter.as_numpy_iterator()

    ans = 0
    for item in numpy_iter:
        # print(len(item), item[0].shape)
        ans += 1
    print(ans)