# train.py
'''
用于训练AlexNet网络
由于内存有限，不能一次读取所有的数据，因此采用每次读取小批量的图片
1. 在datagenerator.py文件中 把所有图片保存成bmp格式文件
2. 存储一个dict字典 每个元素为 (图片的相对路径,标签) 将这个字典序列化保存到文件
3. 读取该文件，把文件名顺序打乱，分成多组，每次加载mini_batch个图片进行训练
4. 在加载图片时，需要把图片放大为227*227的 这是由于AlexNet网络输入的维度
'''

import tensorflow as tf
from alexnet import alexnet
from resnet import resnet
import preprocess_captcha
import numpy as np
import random
import os
import json
import time
import matplotlib.pyplot as plt
from utils import *

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'alexnet', 'model names including alexnet and resnet.')
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_integer('batch_size', 64, 'minibatch size.')
flags.DEFINE_float('dropout', 0.5, 'dropout rate (1 - keep probability).')

def model_run():
	'''
	由于数据量比较大，图片不能一次全部加载进来，因此采用分批读取图片数据
	由于这里每次从硬盘读取batch_size大小的图片，速度较慢
	'''
	'''
	**********************************************
	**                加载数据                   **
	**********************************************
	'''
	# [(文件路径,标签),....]
	training_data, test_data = load_data()

	'''
	**********************************************
	**              定义网络参数                 **
	**********************************************
	'''
	# 模型保存所在文件目录
	model_str = FLAGS.model
	model_name = 'model_param/{0}.ckpt'.format(model_str)
	record_path = './record'

	# 如果目录不存在则创建目录
	if not os.path.isdir(os.path.dirname(model_name)):
		os.mkdir(os.path.dirname(model_name))
	if not os.path.isdir(record_path):
		os.mkdir(record_path)

	# 定义网络超参数
	learning_rate = FLAGS.learning_rate
	training_epoches = FLAGS.epochs
	batch_size = FLAGS.batch_size

	# 输出保存所在文件目录
	record = open('record/record.txt', 'a')
	print ('-----------------------------------', file=record)
	print ('{0} for classifying captcha'.format(model_str.title()), file=record)
	print ('learning_rate', learning_rate, file=record)
	print ('training_epoches', training_epoches, file=record)
	print ('batch_size', batch_size, file=record)
	print (time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), file=record)
	print ('-----------------------------------', file=record)
	record.close()

	# 定义网络参数
	n_classes = 10 + 26 + 26
	pdropout = FLAGS.dropout
	n_train = len(training_data)
	n_test = len(test_data)

	'''
	**********************************************
	**            构建模型 开始测试              **
	**********************************************
	'''
	if model_str == 'alexnet':
		model = alexnet(n_classes)
		image_size = (227, 227)
	elif model_str == 'resnet':
		model = resnet(n_classes)
		image_size = (224, 224)
	else:
		raise RuntimeError('No such model, please choose "alexnet" or "resnet"')

	# 创建Saver op用于保存和恢复所有变量
	saver = tf.train.Saver()

	# 定义学习步骤
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train = optimizer.minimize(model.loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# 如果该文件存在，恢复模型数据
		if os.path.isfile(model_name + '.meta'):
			saver.restore(sess, model_name)

		print ('\nTraining...')
		print ('The result is writing to file "./record/"')

		# 用于可视化结果
		global_train_accuracy = []
		global_test_accuracy = []

		# 开始迭代
		for epoch in range(training_epoches):
			# 打乱训练集
			random.shuffle(training_data)
			# 分组
			mini_batchs = [training_data[k : k + batch_size] for k in range(0, n_train, batch_size)]
			total_iteration_num = n_train // batch_size

			training_accuracy_sum = []
			count = 0
			# 遍历每一个mini_batch
			for mini_batch in mini_batchs:
				# 获取图片和标签
				x_batch, y_batch = preprocess_captcha.get_image_data_and_label(mini_batch, image_size=image_size, one_hot=True)
				
				# 开始训练，train是由optimizer.minimize(model.loss)定义的优化器
				if model_str == 'alexnet':
					feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.keep_prob: pdropout}
				elif model_str == 'resnet':
					feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.is_training: True}

				t = time.time()
				train.run(feed_dict=feed_dict)

				# 每一次训练一批后，输出训练集准确率
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				if model_str == 'alexnet':
					feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.keep_prob: pdropout}
				elif model_str == 'resnet':
					feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.is_training: True}
				training_accuracy, training_cost = sess.run([model.accuracy, model.loss], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
				if count == 0:
					writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
					writer.add_run_metadata(run_metadata, "epoch%03d" % epoch)
					writer.close()

				record = open('record/record.txt', 'a')
				print('Epoch {0} Iteration {1}/{2}: Training set accuracy {3:.3}, loss {4:.3}, time {5:.3}'.format(epoch, count, total_iteration_num, np.mean(training_accuracy), np.mean(training_cost), time.time()-t), file=record)
				record.close()
				count += 1
				training_accuracy_sum.append(np.mean(training_accuracy))

			if epoch % 5 == 0:
				global_train_accuracy.append(np.mean(training_accuracy_sum))

			'''
			每一轮训练完成做测试
			输出测试集准确率，测试时使用较小的mini_batch来测试
			''' 
			# 输出保存所在文件目录
			predictf = open('record/predict.txt', 'a')
			print ('-----------------------------------', file=predictf)
			print ('Epoch {0}'.format(epoch), file=predictf)
			print (time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), file=predictf)
			print ('-----------------------------------', file=predictf)
			predictf.close()
			# 分组，4张连续图片组成一张验证码
			test_batch_size = 1
			mini_batchs = [test_data[k: k+test_batch_size] for k in range(0, n_test, test_batch_size)]
			group_size = 4

			test_accuracy_sum = []
			test_cost_sum = []
			predict = ''
			real = ''
			printNumRemain = 10
			# 遍历每一个mini_batch，当buffer长度为4时，计算精度和损失值加入sum
			for mini_batch in mini_batchs:
				x_batch, y_batch = preprocess_captcha.get_image_data_and_label(mini_batch, image_size=image_size, one_hot=True)

				# 校验
				if model_str == 'alexnet':
					feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.keep_prob: pdropout}
				elif model_str == 'resnet':
					feed_dict = {model.input_x: x_batch, model.input_y: y_batch, model.is_training: False}
				test_predict, test_cost = sess.run([model.out, model.loss], feed_dict=feed_dict)
				predict += convertToStr(np.argmax(test_predict, 1)[0])
				real += mini_batch[0][1]

				# 比较一张验证码图片的预测结果是否正确，记录前10次预测结果
				if len(predict) == group_size:
					predictf = open('record/predict.txt', 'a')
					if printNumRemain > 0:
						print ('Predict:', predict, '\t', 'Real:', real, file=predictf)
					correctCount = compare(predict, real)
					if predict == real:
						test_accuracy_sum.append(1)
						if printNumRemain > 0:
							print ('True\t', 'Accuracy: {0}/{1}'.format(correctCount, group_size), file=predictf)
					else:
						test_accuracy_sum.append(0)
						if printNumRemain > 0:
							print ('False\t', 'Accuracy: {0}/{1}'.format(correctCount, group_size), file=predictf)
					predictf.close()
					predict = ''
					real = ''
					printNumRemain -= 1

				test_cost_sum.append(test_cost)

			# 迭代完一轮测试集，求平均
			record = open('record/record.txt', 'a')
			print ('Epoch {0} Test set accuracy {1:.3}, loss {2:.3}\n'.format(epoch, np.mean(test_accuracy_sum), np.mean(test_cost_sum)), file=record)
			record.close()

			if epoch % 5 == 0:
				global_test_accuracy.append(np.mean(test_accuracy_sum))

			# 保存模型
			if epoch == training_epoches - 1:
				# tensorflow的saver，用于保存模型，先删除原先模型文件
				if remove_model_file(model_name) == True:
					saver.save(sess, model_name)
					record = open('record/record.txt', 'a')
					print ('The model is saved successfully!', file=record)
					print (time.strftime("%a %b %d %H:%M:%S %Y\n", time.localtime()), file=record)
					record.close()

	'''
	**********************************************
	**                 展示结果                  **
	**********************************************
	'''
	# 画出结果图，training_accuracy和test_accuracy的折线统计图
	x = [i for i in range(0, training_epoches, 5)]
	plt.plot(x, global_train_accuracy, marker='o', mec='r', mfc='w', label='train accuary')
	plt.plot(x, global_test_accuracy, marker='^', ms=10, label='test accuary')
	plt.legend()
	plt.margins(0)
	plt.subplots_adjust(bottom=0.10)
	plt.xlabel('Epoch')
	plt.ylabel('Accuary')
	plt.title('{0} for Captcha Recognition'.format(model_str.title()))
	names = [i for i in range(0, training_epoches, 5)]
	plt.xticks(x, names, rotation=1)
	plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
	plt.savefig('record/result.png')
	plt.show()

if __name__ == '__main__':
	model_run()
