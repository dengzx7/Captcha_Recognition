# alexnet.py
'''
AlexNet为8层深度网络，其中5层卷积层和3层全连接层，不计LRN层和池化层
'''

import tensorflow as tf

def conv(x, filter_height, filter_width, out_channels, stride_y, stride_x, name, padding='SAME'):
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

	with tf.name_scope(name) as scope:
		shape = [filter_height, filter_width, in_channels, out_channels]
		weights = tf.get_variable(scope+'w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
		# Variable创建变量，自动检测有没有重命名，有就自行处理
		biases = tf.Variable(tf.constant(0.0, shape=[out_channels]), dtype=tf.float32, trainable=True, name=scope+'b')
		# 这里strides的每个元素分别与[batch, in_height, in_width, in_channels]对应
		conv = tf.nn.conv2d(x, weights, strides=[1, stride_y, stride_x, 1], padding=padding)
		# 添加共享偏置
		bias = tf.nn.bias_add(conv, biases)
		# 添加relu激活函数
		relu = tf.nn.relu(bias, name=scope)

		return relu

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
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
	return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

def fc(x, num_out, name, relu=True):
	'''
	全连接网络
	'''
	num_in = int(x.get_shape()[-1])
	with tf.name_scope(name) as scope:
		# Variables创建变量，weights and biases
		shape = [num_in, num_out]
		weights = tf.get_variable(scope+'w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
		biases = tf.Variable(tf.constant(0.0, shape=[num_out]), dtype=tf.float32, trainable=True, name=scope+'b')

		if relu:
			act = tf.nn.xw_plus_b(x, weights, biases)
			# Apply ReLu non linearity
			relu = tf.nn.relu(act, name=scope)
			return relu
		else:
			act = tf.nn.xw_plus_b(x, weights, biases, name=scope)
			return act

def lrn(x, radius, alpha, beta, name, bias=1.0):
	'''
	局部相应归一化操作
	Args:
		depth_radius: 归一化窗口的半径长度 一般设置为5 也是 论文中的n/2的值
		alpha:        超参数 一般设置为1e-4
		beta:         指数 一般设置为0.5
		bias:         偏置 一般设置为1.0
		name:         lrn层名字
	'''
	return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
	'''
	弃权操作
	Args:
		x:         激活函数之后的输出
		keep_prob: 弃权概率 即每个神经元被保留的概率
	'''
	return tf.nn.dropout(x, keep_prob)

def print_dimension(t):
	'''
	输出指定张量的维度
	'''
	print (t.op.name, '\t', t.get_shape().as_list())


class alexnet(object):
	'''
	AlexNet网络结构类
	'''
	def __init__(self, n_classes):
		'''
		Create the graph of the AlexNet model
		Args:
			x: 输入张量的占位符
			keep_prob: 每个神经元保留的概率
		'''
		self.input_x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='input')
		self.input_y = tf.placeholder(tf.float32, shape=[None, n_classes], name='input_y')
		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		self.n_classes = n_classes
		self.out = self.create()
		self.print_ = None

	def create(self):
		'''
		创建AlexNet网络 返回网络输出的张量（未经过激活函数）
		共有8层网络，5层卷积层，3个全连接网络
		'''
		print_dimension(self.input_x)

		# 1st Layer: Conv (w ReLu) -> Lrn -> Pool
		self.conv1 = conv(self.input_x, filter_height=11, filter_width=11, out_channels=96, stride_y=4, stride_x=4, padding='VALID', name='conv1')
		print_dimension(self.conv1)
		self.norm1 = lrn(self.conv1, radius=4, alpha=1e-4, beta=0.75, name='norm1')
		self.pool1 = max_pool(self.norm1, filter_height=3, filter_width=3, stride_y=2, stride_x=2, padding='VALID', name='pool1')
		print_dimension(self.pool1)

		# 2nd Layer: Conv (w ReLu) -> Lrn -> Pool with 2 groups
		self.conv2 = conv(self.pool1, filter_height=5, filter_width=5, out_channels=256, stride_y=1, stride_x=1, name='conv2')
		print_dimension(self.conv2)
		self.norm2 = lrn(self.conv2, radius=4, alpha=1e-4, beta=0.75, name='norm2')
		self.pool2 = max_pool(self.norm2, filter_height=3, filter_width=3, stride_y=2, stride_x=2, padding='VALID', name='pool2')
		print_dimension(self.pool2)

		# 3rd Layer: Conv (w ReLu)
		self.conv3 = conv(self.pool2, filter_height=3, filter_width=3, out_channels=384, stride_y=1, stride_x=1, name='conv3')
		print_dimension(self.conv3)

		# 4th Layer: Conv (w ReLu) splitted into two groups
		self.conv4 = conv(self.conv3, filter_height=3, filter_width=3, out_channels=384, stride_y=1, stride_x=1, name='conv4')
		print_dimension(self.conv4)

		# 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
		self.conv5 = conv(self.conv4, filter_height=3, filter_width=3, out_channels=256, stride_y=1, stride_x=1, name='conv5')
		print_dimension(self.conv5)
		self.pool5 = max_pool(self.conv5, filter_height=3, filter_width=3, stride_y=2, stride_x=2, padding='VALID', name='pool5')
		print_dimension(self.pool5)

		# 6th Layer: Flatten -> FC (w ReLu) -> Dropout
		flattened = tf.reshape(self.pool5, [-1, 6*6*256])
		self.fc6 = fc(flattened, 4096, name='fc6')
		print_dimension(self.fc6)
		self.dropout6 = dropout(self.fc6, self.keep_prob)

		# 7th Layer: FC (w ReLu) -> Dropout
		self.fc7 = fc(self.dropout6, 4096, name='fc7')
		print_dimension(self.fc7)
		self.dropout7 = dropout(self.fc7, self.keep_prob)

		# 8th Layer: FC and return unscaled activations
		self.out = fc(self.dropout7, self.n_classes, relu=False, name='output')
		print_dimension(self.out)

		# 计算交叉熵损失函数
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.input_y))

		# 计算准确率
		with tf.name_scope('accuracy'):
			self.correct = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

		return self.out
