# resnet.py
'''
实现 34-layer 残差网络
'''

import tensorflow as tf

def conv(x, filter_height, filter_width, out_channels, stride_y, stride_x, name, is_training=True):
	'''
	定义一个卷积层 
	Args:
		x:输入图片集合 维度为[batch, in_height, in_width, in_channels] 
		weights:共享权重[filter_height, filter_width, in_channels, out_channels]
			filter_height:  过滤器高度
			filter_width:   过滤器宽度
			in_channels:    输入通道数 
			out_channels:   过滤器个数
		biases:共享偏置 [out_channels]
		stride_x:      x轴步长
		stride_y:      y轴步长
		name：         卷积层名字
		padding:       卷基层填充方式 'VALID'或者'SAME'
	'''
	# 获取输入通道数
	in_channels = int(x.get_shape()[-1])
	padding = 'SAME'

	with tf.name_scope(name) as scope:
		shape = [filter_height, filter_width, in_channels, out_channels]
		filter = tf.get_variable(scope+'f', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv = tf.nn.conv2d(x, filter, strides=[1, stride_y, stride_x, 1], padding=padding)
		# Batch Normalization
		bn = tf.layers.batch_normalization(conv, training=is_training)
		# ReLu
		relu = tf.nn.relu(bn, name=scope)

		return relu

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name):
	'''
	定义一个池化层 最大下采样操作
	Args:
		x:输入图片集合 维度为[batch, in_height, in_width, in_channels] 
		filter_height: 滑动窗口高度
		filter_width:  滑动窗口宽度
		stride_x:      x轴步长
		stride_y:      y轴步长
		name:          池化层名字
		padding:       填充方式 'SAME'或者'VALID'
	'''
	padding = 'SAME'
	return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

def block(x, filter_height, filter_width, out_channels, stride_y, stride_x, name, is_training=True):
	'''
	定义一个block层，包含两个conv操作和一个shoct cut操作，即 Conv -> BN -> ReLu
	Args:
		x: 输入的图片像素, 维度 [batch_size, in_height, in_width, in_channels]
		filter: 卷积核, 维度 [filter_height, filter_width, in_channels, out_channels]
		name: block名称
		padding: 卷积层填充方式
	'''
	conv_1 = conv(x, filter_height, filter_width, out_channels, stride_y, stride_x, name+'_1', is_training)
	conv_2 = conv(conv_1, filter_height, filter_width, out_channels, 1, 1, name+'_2', is_training)

	same = True
	for i in range(len(x.get_shape())):
		if (x.get_shape()[i] == conv_2.get_shape()[i]) != None:
			if x.get_shape()[i] != conv_2.get_shape()[i]:
				same = False
		
	if same:
		return tf.add(x, conv_2, name)
	else:
		x_height, x_width = int(x.get_shape()[1]), int(x.get_shape()[2])
		conv_2_height, conv_2_width = int(conv_2.get_shape()[1]), int(conv_2.get_shape()[2])
		stride_height = int(round(x_height / conv_2_height))
		stride_width = int(round(x_width / conv_2_width))
		short_cut = conv(x, 1, 1, out_channels, stride_height, stride_width, name+'sc', is_training)
		return tf.add(short_cut, conv_2, name)

def fc(x, num_out, name):
	'''
	全连接网络
	'''
	num_in = int(x.get_shape()[-1])

	with tf.name_scope(name) as scope:
		shape = [num_in, num_out]
		weights = tf.get_variable(scope+'w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
		biases = tf.Variable(tf.constant(0.0, shape=[num_out]), dtype=tf.float32, trainable=True, name=scope+'b')
		return tf.nn.xw_plus_b(x, weights, biases, name=scope)

def print_dimension(t):
	'''
	输出指定张量的维度
	'''
	print (t.op.name, '\t', t.get_shape().as_list())


class resnet(object):
	'''
	ResNet网络结构类
	'''
	def __init__(self, n_classes):
		'''
		创建残差神经网络的图结构
		Args:
			input_x: 输入的图片像素数据
			input_y: 输入的图片标签数据
			n_classes: 分类总数
			out: 最终分类结果
		'''
		self.input_x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
		self.input_y = tf.placeholder(tf.float32, shape=[None, n_classes], name='input_y')
		self.is_training = tf.placeholder(tf.bool)
		self.n_classes = n_classes
		self.out = self.create()

	def create(self):
		'''
		创建ResNet网络，返回网络输出
		网络结构中的卷积层: 7*7 64(*1), 3*3 64(*6), 3*3 128(*8), 3*3 256(*12), 3*3 512(*6)
		'''
		print_dimension(self.input_x)

		# 1st Part: Conv (BN, ReLu) -> MaxPool
		self.conv1 = conv(self.input_x, filter_height=7, filter_width=7, out_channels=64, stride_y=2, stride_x=2, name='conv1', is_training=self.is_training)
		print_dimension(self.conv1)
		self.pool1 = max_pool(self.conv1, filter_height=3, filter_width=3, stride_y=2, stride_x=2, name='pool1')
		print_dimension(self.pool1)

		# 2st Part: Conv (BN, ReLu) * 6
		self.block2_1 = block(self.pool1, filter_height=3, filter_width=3, out_channels=64, stride_y=1, stride_x=1, name='blk2_1', is_training=self.is_training)
		print_dimension(self.block2_1)
		self.block2_2 = block(self.block2_1, filter_height=3, filter_width=3, out_channels=64, stride_y=1, stride_x=1, name='blk2_2', is_training=self.is_training)
		print_dimension(self.block2_2)
		self.block2_3 = block(self.block2_2, filter_height=3, filter_width=3, out_channels=64, stride_y=1, stride_x=1, name='blk2_3', is_training=self.is_training)
		print_dimension(self.block2_3)

		# 3rd Part: Conv (BN, ReLu) * 8
		self.block3_1 = block(self.block2_3, filter_height=3, filter_width=3, out_channels=128, stride_y=2, stride_x=2, name='blk3_1', is_training=self.is_training)
		print_dimension(self.block3_1)
		self.block3_2 = block(self.block3_1, filter_height=3, filter_width=3, out_channels=128, stride_y=1, stride_x=1, name='blk3_2', is_training=self.is_training)
		print_dimension(self.block3_2)
		self.block3_3 = block(self.block3_2, filter_height=3, filter_width=3, out_channels=128, stride_y=1, stride_x=1, name='blk3_3', is_training=self.is_training)
		print_dimension(self.block3_3)
		self.block3_4 = block(self.block3_3, filter_height=3, filter_width=3, out_channels=128, stride_y=1, stride_x=1, name='blk3_4', is_training=self.is_training)
		print_dimension(self.block3_4)

		# 4th Part: Conv (BN, ReLu) * 12
		self.block4_1 = block(self.block3_4, filter_height=3, filter_width=3, out_channels=256, stride_y=2, stride_x=2, name='blk4_1', is_training=self.is_training)
		print_dimension(self.block4_1)
		self.block4_2 = block(self.block4_1, filter_height=3, filter_width=3, out_channels=256, stride_y=1, stride_x=1, name='blk4_2', is_training=self.is_training)
		print_dimension(self.block4_2)
		self.block4_3 = block(self.block4_2, filter_height=3, filter_width=3, out_channels=256, stride_y=1, stride_x=1, name='blk4_3', is_training=self.is_training)
		print_dimension(self.block4_3)
		self.block4_4 = block(self.block4_3, filter_height=3, filter_width=3, out_channels=256, stride_y=1, stride_x=1, name='blk4_4', is_training=self.is_training)
		print_dimension(self.block4_4)
		self.block4_5 = block(self.block4_4, filter_height=3, filter_width=3, out_channels=256, stride_y=1, stride_x=1, name='blk4_5', is_training=self.is_training)
		print_dimension(self.block4_5)
		self.block4_6 = block(self.block4_5, filter_height=3, filter_width=3, out_channels=256, stride_y=1, stride_x=1, name='blk4_6', is_training=self.is_training)
		print_dimension(self.block4_6)

		# 5th Part: Conv(BN, ReLu) * 6
		self.block5_1 = block(self.block4_6, filter_height=3, filter_width=3, out_channels=512, stride_y=2, stride_x=2, name='blk5_1', is_training=self.is_training)
		print_dimension(self.block5_1)
		self.block5_2 = block(self.block5_1, filter_height=3, filter_width=3, out_channels=512, stride_y=1, stride_x=1, name='blk5_2', is_training=self.is_training)
		print_dimension(self.block5_2)
		self.block5_3 = block(self.block5_2, filter_height=3, filter_width=3, out_channels=512, stride_y=1, stride_x=1, name='blk5_3', is_training=self.is_training)
		print_dimension(self.block5_3)

		# 6th Part: global AvgPool -> FC
		self.pool2 = tf.reduce_mean(self.block5_3, reduction_indices=[1, 2], name='pool2')
		print_dimension(self.pool2)
		self.out = fc(self.pool2, self.n_classes, name='output')
		print_dimension(self.out)

		# 计算交叉熵损失函数
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.input_y))

		# 计算准确率
		with tf.name_scope('accuracy'):
			self.correct = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

		return self.out
