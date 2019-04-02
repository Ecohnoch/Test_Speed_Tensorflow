import tensorflow as tf 
import numpy as np

import os
import time
import threading

from utils.my_utils import load_wave_list
from thin_resnet import resnet34

BATCH_SIZE = 128
WAV_DIR    = '/data/ChuyuanXiong/up/wav'
SPLIT_FILE = 'utils/vox1_split_backup.txt'

PLACE_HOLDER = {
	'thin_resnet': [tf.placeholder(tf.float32, [None, 257, 250, 1], name='audio_input'), tf.placeholder(tf.int64, [None], name='audio_label')]
}


def audio_data_extracted(fileName, rootdir):
    train_list = np.loadtxt(fileName, str)
    train_file_list = np.array([os.path.join(rootdir, i[1]) for i in train_list])
    train_label_list= np.array([int(i[0]) for i in train_list])
    return train_file_list, train_label_list


def feed_dict_test():
    train_file_list, train_label_list = audio_data_extracted(SPLIT_FILE, WAV_DIR)

    def get_batch(dataset, start, batch_size=128):
        end = (start + batch_size) % dataset.shape[0]
        if end > start:
            return load_wave_list(dataset[start:end]), end, False
        
        sub_dataset = np.hstack((dataset[start:], dataset[:end]))
        return load_wave_list(sub_dataset), end, True

    def get_label_batch(dataset, start, batch_size=128):
        end = (start + batch_size) % dataset.shape[0]
        if end > start:
            return dataset[start:end], end, False

        return np.hstack((dataset[start:], dataset[:end])), end, True

    x = PLACE_HOLDER['thin_resnet'][0]
    y = PLACE_HOLDER['thin_resnet'][1]

    emb = resnet34(x)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        idx_train = 0
        idx_train_label = 0



        batch_10_start = time.time()
        for i in range(10):
            start_time = time.time()
            batch_train, idx_train, end_epoch = get_batch(train_file_list, idx_train, batch_size=BATCH_SIZE)
            batch_train_label, idx_train_label, end_epoch = get_label_batch(train_label_list, idx_train_label, batch_size=BATCH_SIZE)
            batch_train = np.array(batch_train)

            batch_start_time = time.time()
            emb_val = sess.run(emb, feed_dict={x: batch_train})
            batch_end_time   = time.time()
            print('Counter:', i, 'Step time:', batch_end_time - start_time, 'Batch time:', batch_end_time - batch_start_time)
        batch_10_end   = time.time()

        print('10 Batch time consumed: ', batch_10_end - batch_10_start)




idx_thread_train = 0
idx_thread_label = 0

def multi_thread_test():
    train_file_list, train_label_list = audio_data_extracted(SPLIT_FILE, WAV_DIR)

    def get_batch(dataset, start, batch_size=128):
        end = (start + batch_size) % dataset.shape[0]
        if end > start:
            return load_wave_list(dataset[start:end]), end, False
        
        sub_dataset = np.hstack((dataset[start:], dataset[:end]))
        return load_wave_list(sub_dataset), end, True

    def get_label_batch(dataset, start, batch_size=128):
        end = (start + batch_size) % dataset.shape[0]
        if end > start:
            return dataset[start:end], end, False

        return np.hstack((dataset[start:], dataset[:end])), end, True

    x = PLACE_HOLDER['thin_resnet'][0]
    y = PLACE_HOLDER['thin_resnet'][1]

    with tf.device('/cpu:0'):
        q = tf.FIFOQueue(BATCH_SIZE*3, [tf.float32, tf.int64], shapes=[[257, 250, 1], []])
        enqueue_op = q.enqueue_many([x, y])
        x_b, y_b = q.dequeue_many(BATCH_SIZE)

    emb = resnet34(x_b)

    coord = tf.train.Coordinator()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        def enqueue_batches():
            while not coord.should_stop():
                global idx_thread_train
                global idx_thread_label
                batch_train, idx_train, end_epoch = get_batch(train_file_list, idx_thread_train, batch_size=BATCH_SIZE)
                batch_train_label, idx_train_label, end_epoch = get_label_batch(train_label_list, idx_thread_train, batch_size=BATCH_SIZE)
                batch_train = np.array(batch_train)
                sess.run(enqueue_op, feed_dict={x: batch_train, y: batch_train_label})
                if end_epoch:
                    idx_thread_train = 0
                    idx_thread_label = 0

        num_threads = 3
        for j in range(num_threads):
            t = threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()

        batch_10_start = time.time()
        for i in range(10):
            start_time = time.time()
            batch_start_time = time.time()
            emb_val = sess.run(emb)
            batch_end_time   = time.time()
            print('Counter:', i, 'Step time:', batch_end_time - start_time, 'Batch time:', batch_end_time - batch_start_time)
        batch_10_end   = time.time()

        print('10 Batch time consumed: ', batch_10_end - batch_10_start)
        coord.request_stop()
        coord.join()


def generate_tfrecord():
	pass

def tfrecord_test():
	pass






if __name__ == '__main__':
	multi_thread_test()


