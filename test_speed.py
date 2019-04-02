import tensorflow as tf 
import numpy as np

import os
import time
import threading

from utils.my_utils import load_wave_list
from utils.my_utils import load_data
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


def generate_tfrecord(output_path):
    train_file_list, train_label_list = audio_data_extracted(SPLIT_FILE, WAV_DIR)
    writer = tf.python_io.TFRecordWriter(os.path.join(output_path, 'tran.tfrecords'))
    for ind, (file, label) in enumerate(zip(train_file_list, train_label_list)):
        audio = load_data(file)
        audio_raw = audio.tobytes()
        label = int(label)
        example = tf.train.Example(features=tf.train.Features(feature={
                'audio_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_raw])),
                'label':     tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
        writer.write(example.SerializeToString())
        if i != 0 and ind % 1000 == 0:
            print("%d num audios processed" % ind)
        if i != 0 and ind % 10000== 0:
            print("10000 num processed")
            break
    writer.close()

def parse_function(example_proto):
    features = {
        'audio_raw': tf.FixedLenFeature([], tf.string),
        'label':     tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(example_proto, features)
    audio_file = tf.decode_raw(features['audio_raw'], tf.float32)
    audio_file = tf.reshape(audio_file, [257, 250, 1])
    label      = tf.cast(features['label'], tf.int64)

    return audio_file, label

def tfrecord_test(trans_file):
    dataset = tf.data.TFRecordDataset(trans_file)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    iterator= dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    x = PLACE_HOLDER['thin_resnet'][0]
    y = PLACE_HOLDER['thin_resnet'][1]

    emb = resnet34(x)


    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sess.run(iterator.initializer)

        batch_10_start = time.time()
        for i in range(10):
            start_time = time.time()
            batch_train, batch_train_label = sess.run(next_element)

            batch_start_time = time.time()
            emb_val = sess.run(emb, feed_dict={x: batch_train})
            batch_end_time   = time.time()
            print('Counter:', i, 'Step time:', batch_end_time - start_time, 'Batch time:', batch_end_time - batch_start_time)
        batch_10_end   = time.time()
        print('10 Batch time consumed: ', batch_10_end - batch_10_start)







if __name__ == '__main__':
	# multi_thread_test()
    # generate_tfrecord('/data/ChuyuanXiong/up/audio_trans')
    tfrecord_test('/data/ChuyuanXiong/up/audio_trans/tran.tfrecords')


