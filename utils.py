# utils.py
'''
一些辅助函数
'''

import json
import os

def remove_model_file(file):
	'''
	删除保存的模型文件
	删除成功返回True 否则返回 False
	Args:
		file: 模型文件名
	'''
	path = [file + item for item in ('.index', '.meta')]
	path.append(os.path.join(os.path.dirname(path[0]), 'checkpoint'))
	for f in path:
		if os.path.isfile(f):
			os.remove(f)
	ret = True
	for f in path:
		if os.path.isfile(f):
			ret = False
	return ret

def load_data():
	'''
	读取数据 (image_path, label)
	'''
	with open('dataset/train_file_labels.json') as f:
		training_data = json.load(f)
	with open('dataset/test_file_labels.json') as f:
		test_data = json.load(f)
	return training_data, test_data

def convertToStr(index):
	'''
	将整型标签转换为字符型标签
	0-9   -> 0-9
	10-35 -> a-z
	36-61 -> A-Z
	'''
	if 0 <= index <= 9:
		ret = chr(index + ord('0'))
	elif 10 <= index <= 35:
		ret = chr(index - 10 + ord('a'))
	elif 36 <= index <= 61:
		ret = chr(index - 36 + ord('A'))
	return ret

def compare(predict, real):
	'''
	比对预测结果和真实值，返回预测正确的字符个数
	'''
	correctCount = 0
	for i in range(len(predict)):
		if predict[i] == real[i]:
			correctCount += 1
	return correctCount
