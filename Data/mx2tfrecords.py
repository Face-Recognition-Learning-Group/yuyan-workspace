import io
import os
import cv2

import argparse
import PIL.Image
import numpy as np
import mxnet as mx
import tensorflow as tf

def save_tf_records(imgidx, imgrec, save_filename):
    writer = tf.io.TFRecordWriter(save_filename)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


if __name__ == '__main__':
    idx_path = r'D:\yuyan-workspace\tmp\faces_vgg_112x112\train.idx'
    train_path = r'D:\yuyan-workspace\tmp\faces_vgg_112x112\train.rec'
    id2range = {}
    data_shape = (3, 112, 112)

    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, train_path, 'r')

    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    print(header.label)
    imgidx = list(range(1, int(header.label[0])))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        id2range[identity] = (a, b)

    save_tf_records(imgidx, imgrec, r'D:\yuyan-workspace\tmp\face_train.records')
