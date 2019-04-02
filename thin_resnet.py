import tensorflow as tf 
import numpy as np 

def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse):
	filters1, filters2, filters3 = filters

	conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
	bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'

	x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, kernel_initializer='orthogonal', name=conv_name_1, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = tf.layers.conv2d(x, filters2, kernel_size, use_bias=False, padding='SAME', kernel_initializer='orthogonal', name=conv_name_2, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = tf.layers.conv2d(x, filters3, (1,1),  use_bias=False, kernel_initializer='orthogonal', name=conv_name_3, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)
	
	x = tf.add(input_tensor, x)
	x = tf.nn.relu(x)
	return x


def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2)):
	filters1, filters2, filters3 = filters

	conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
	bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'
	x = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides=strides, use_bias=False, kernel_initializer='orthogonal', name=conv_name_1, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = tf.layers.conv2d(x, filters2, kernel_size, padding='SAME', use_bias=False, kernel_initializer='orthogonal', name=conv_name_2, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = tf.layers.conv2d(x, filters3, (1,1), use_bias=False, kernel_initializer='orthogonal', name=conv_name_3, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

	conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
	bn_name_4   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_shortcut'
	shortcut = tf.layers.conv2d(input_tensor, filters3, (1,1), use_bias=False, strides=strides, kernel_initializer='orthogonal', name=conv_name_4, reuse=reuse)
	shortcut = tf.layers.batch_normalization(x, training=is_training, name=bn_name_4, reuse=reuse)

	x = tf.add(shortcut, x)
	x = tf.nn.relu(x)
	return x


def resnet34(input_tensor, is_training=True, pooling_and_fc=True, reuse=False):
	x = tf.layers.conv2d(input_tensor, 64, (7,7), strides=(1,1), kernel_initializer='orthogonal', use_bias=False, padding='SAME', name='conv1_1/3x3_s1', reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name='bn1_1/3x3_s1', reuse=reuse)
	x = tf.nn.relu(x)
	x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), name='mpool1')

	x1 = conv_block_2d(x, 3, [48, 48, 96], stage=2, block='1a', strides=(1,1), is_training=True, reuse=reuse)
	x1 = identity_block2d(x1, 3, [48, 48, 96], stage=2, block='1b', is_training=True, reuse=reuse)

	x2 = conv_block_2d(x1, 3, [96, 96, 128], stage=3, block='2a', strides=(2,2), is_training=True, reuse=reuse)
	x2 = identity_block2d(x2, 3, [96, 96, 128], stage=3, block='2b', is_training=True, reuse=reuse)
	x2 = identity_block2d(x2, 3, [96, 96, 128], stage=3, block='2c', is_training=True, reuse=reuse)

	x3 = conv_block_2d(x2, 3, [128, 128, 256], stage=4, block='3a', strides=(2,2), is_training=True, reuse=reuse)
	x3 = identity_block2d(x3, 3, [128, 128, 256], stage=4, block='3b', is_training=True, reuse=reuse)
	x3 = identity_block2d(x3, 3, [128, 128, 256], stage=4, block='3c', is_training=True, reuse=reuse)
	x3 = identity_block2d(x3, 3, [128, 128, 256], stage=4, block='3d', is_training=True, reuse=reuse)


	x4 = conv_block_2d(x3, 3, [256, 256, 512], stage=5, block='4a', is_training=True, reuse=reuse)
	x4 = identity_block2d(x4, 3, [256, 256, 512], stage=5, block='4b', is_training=True, reuse=reuse)
	x4 = identity_block2d(x4, 3, [256, 256, 512], stage=5, block='4c', is_training=True, reuse=reuse)

	if pooling_and_fc:
		pooling_output = tf.layers.max_pooling2d(x4, (3,1), strides=(2,2), name='mpool2')
		fc_output      = tf.layers.conv2d(pooling_output, 512, (7, 1), name='fc1', reuse=reuse)

	return fc_output

if __name__ == '__main__':
	example_data = [np.random.rand(257, 250, 1)]
	x = tf.placeholder(tf.float32, [None, 257, 250, 1])
	y = resnet50(x, is_training=True, reuse=False)
	print(y)

	with tf.Session() as sess:
		writer = tf.summary.FileWriter("logs/", sess.graph)
		init = tf.global_variables_initializer()
		sess.run(init)
