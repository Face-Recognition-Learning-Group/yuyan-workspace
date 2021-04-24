import os
import traceback
import numpy as np
import tensorflow as tf

from Data.preprocessing import default_uint8_image_preprocessing

def _parse_func(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    img = tf.io.decode_jpeg(features['image_raw'])
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, shape=(112, 112, 3))
    label = tf.cast(features['label'], tf.int64)
    return img, label

def _load_tfrecord(record_path: str, batch_size: int, parallel_batches: int, shuffle_buffer_size=100000):
    assert os.path.exists(record_path)
    inputs = tf.data.TFRecordDataset(record_path)
    inputs = inputs.shuffle(shuffle_buffer_size)
    inputs = inputs.apply(tf.data.experimental.map_and_batch(_parse_func, num_parallel_calls=parallel_batches, batch_size=batch_size))
    return inputs


if __name__ == '__main__':
    iter = _load_tfrecord(r'D:\yuyan-workspace\tmp\face_train.records', batch_size=16, parallel_batches=16)
    s = 0
    for each_item in iter.as_numpy_iterator():
        if s < max(each_item[1]):
            s = max(each_item[1])
    print(s)
