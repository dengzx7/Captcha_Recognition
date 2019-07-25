# preprocess_captcha.py
'''
预处理验证码图片
Preprocess the captcha images
1. Eliminating interferential curve
2. Graying and binarizing
3. Denoising
4. Image segmentation
'''

import os
from PIL import Image
from PIL import ImageDraw
import json
import numpy as np
import cv2

row, col = 32, 90
target_resize = (92, 32)
preprocess_path = 'dataset/preprocess/'
segmentation_path = 'dataset/segmentation/'

def makeDir():
	'''
	创建目录
	Make directory 'dataset/preprocess' and 'dataset/segmentation/'
	'''
	# If directory already exists
	if os.path.isdir(preprocess_path):
		print (preprocess_path, 'directory exists')
		return
	if os.path.isdir(segmentation_path):
		print (segmentation_path, 'directory exists')
		return

	os.mkdir(preprocess_path)
	os.mkdir(segmentation_path)

	for i in range(10 + 26 + 26):
		os.mkdir(segmentation_path + str(i))

def makePath(index, dir):
	'''
	建立图片路径
	Make path string of images in 'dataset/dir/xxxx.png' format
	Args:
		index: integer type in 'xxxx' format, standing for image's id
		dir: two options for dir, 'images' or 'preprocess'
	'''
	path = 'dataset/' + dir + '/'
	if index < 10:
		image_index = '000'
	elif index < 100:
		image_index = '00'
	elif index < 1000:
		image_index = '0'
	image_index += str(index)
	path += image_index + '.png'
	return image_index, path

def readImage(path):
	'''
	使用Image读取图片
	Read image, return Iamge type
	Args:
		path: the path of the image
	'''
	image = Image.open(path)
	return image

def eliminateInterferentialCurve(image):
	'''
	消除验证码图片中的干扰线
	Replace pixel to (255, 255, 255) which is in paticular range (0-15, 0-15, 0-15)
	Args:
		image: image obj from PIL.Image
	'''
	table_RGB = [[0 for j in range(col)] for i in range(row)]
	for i in range(row):
		for j in range(col):
			pixel = image.getpixel((j, i))
			if 0 <= pixel[0] <= 15 and 0 <= pixel[1] <= 15 and 0 <= pixel[2] <= 15:
				table_RGB[i][j] = (255, 255, 255)
			else:
				table_RGB[i][j] = pixel

	draw = ImageDraw.Draw(image)
	for i in range(row):
		for j in range(col):
			draw.point((j, i), table_RGB[i][j])

def grayingAndBinarying(image):
	'''
	灰度化和二值化图片
	Change pixel to {0, 1}, with thresold = 128
	Args:
		image: image obj from PIL.Image agter eiliminating interferential curve
	'''
	thresold = 170
	image = image.convert("L")

	table_01 = []
	for i in range(256):
		if i < thresold:
			table_01.append(0)
		else:
			table_01.append(1)
	image = image.point(table_01, '1')
	return image

def countOnes(table_Bin, x, y):
	'''
	滤波处理，计算像素点(x, y)周围像素值为1的像素点的个数
	Count the number of '1' in binary image around pixel (x, y)
	Args:
		table_Bin: the 2-D table with binary values of a binary image
		x: table coordinate x
		y: table coordinate y
	'''
	if table_Bin[x][y] == 1:
		return 8
	direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
	count = 0
	for dir in direction:
		if table_Bin[x + dir[0]][y + dir[1]] == 1:
			count += 1
	return count

def denoising(image, image_index):
	'''
	去噪
	Remove pepper noise from the image
	Args:
		image: image obj from PIL.Image after graying and binarizing
		index_index: string type in 'xxxx' format, standing for image's id
	'''
	table_Bin = [[0 for j in range(col)] for i in range(row)]
	for i in range(row):
		for j in range(col):
			table_Bin[i][j] = image.getpixel((j, i))

	# Count for the number of pixel '1' around a paticular pixel
	# If the count >= thresold, change the pixel to '1'
	thresold = 7
	for i in range(1, row - 1):
		for j in range(1, col - 1):
			if countOnes(table_Bin, i, j) >= thresold:
				table_Bin[i][j] = 1
			else:
				table_Bin[i][j] = 0

	draw = ImageDraw.Draw(image)
	for i in range(row):
		for j in range(col):
			draw.point((j, i), table_Bin[i][j])
			
	# resize image to target_resize
	image.resize(target_resize, Image.ANTIALIAS).save(preprocess_path + image_index + '.png')

def segmentation(iter, image, image_index, train_file_labels):
	'''
	平均分割验证码图片
	Average split image
	Args:
		iter: the process of iter_th image
		image: image obj from PIL.Image after denoising
		index_index: string type in 'xxxx' format, standing for image's id
		train_file_labels: list that store (image_path, label)
	'''
	f = open('dataset/labels/labels2.txt')
	labels = []
	for line in f.readlines():
		labels.append(line.strip())
	f.close()

	image_index, path = makePath(iter, 'preprocess')
	image = readImage(path)
	# Every image has four characters
	for i in range(4):
		child = image.crop((i * 23, 0, (i + 1) * 23, 32))
		# Get label
		if 'a' <= labels[iter - 1][i] <= 'z':
			label = str(ord(labels[iter - 1][i]) - 97 + 10)
		elif 'A' <= labels[iter - 1][i] <= 'Z':
			label = str(ord(labels[iter - 1][i]) - 65 + 10 + 26)
		else:
			label = labels[iter - 1][i]
		child_path = segmentation_path + label + '/' + image_index + '-' + str(i) + '.png'
		child.save(child_path)
		train_file_labels.append((child_path, labels[iter - 1][i]))

def store(train_file_labels, test_file_labels):
	'''
	存储(文件路径, 标签)列表到json
	Store (image_path, label) list to json
	Args:
		train_file_labels: list that store (image_path, label)
	'''
	with open('dataset/train_file_labels.json', 'w') as f:
		json.dump(train_file_labels, f)
	with open('dataset/test_file_labels.json', 'w') as f:
		json.dump(test_file_labels, f)
	print ('\n(image, label) paths have been written successully!')
	print ('preprocessing for dataset has completed!')

def get_one_hot_label(labels, depth):
    '''
    把标签二值化  返回numpy.array类型
    Binarize labels, return numpy.array type
    Args:
        labels: the set of labels
        depth: the class number of labels
    '''   
    m = np.zeros([len(labels), depth])
    for i in range(len(labels)):
    	if '0' <= labels[i] <= '9':
    		m[i][ord(labels[i]) - 48] = 1
    	elif 'a' <= labels[i] <= 'z':
    		m[i][10 + ord(labels[i]) - 97] = 1
    	elif 'A' <= labels[i] <= 'Z':
    		m[i][10 + 26 + ord(labels[i]) - 65] = 1
    return m

def get_image_data_and_label(value, image_size='NONE', depth=10+26+26, one_hot=False):
    '''
    获取图片数据，以及标签数据 注意每张图片维度为 n_w x n_h x n_c
    Get images and labels
    Args:
        value: list containing tuples like (x, y)
            x: image path
            y: label
        image_size: image size
        one_hot: binarize labels
        depth: the class number of labels
    '''
    # images
    x_batch = []
    # labels
    y_batch = []

    for image in value:
    	if image_size == 'NONE':
    		x_batch.append(cv2.imread(image[0]) / 255)
    	else:
    		# resize image to image_size
    		x_batch.append(cv2.resize(cv2.imread(image[0]), image_size) / 255)
    	y_batch.append(image[1])

    if one_hot == True:
    	# Binarize labels
    	y_batch = get_one_hot_label(y_batch, depth)

    return np.asarray(x_batch, dtype=np.float32), np.asarray(y_batch, dtype=np.float32)

def preprocess():
	'''
	预处理
	Preprocession of raw data
	1. Eliminating interferential curve
	2. Graying and binarizing
	3. Denoising
	4. Image segmentation
	'''
	makeDir()
	# Begin processing, image 1-800 for training, image 801-920 for testing
	train_total = 800
	test_total = 120
	total = 920
	train_file_labels = []
	test_file_labels = []
	for i in range(1, total + 1):
		print ('processing image', i)
		image_index, path = makePath(i, 'images')
		image = readImage(path)
		eliminateInterferentialCurve(image)
		image = grayingAndBinarying(image)
		denoising(image, image_index)
		if i <= train_total:
			segmentation(i, image, image_index, train_file_labels)
		else:
			segmentation(i, image, image_index, test_file_labels)
	store(train_file_labels, test_file_labels)

if __name__ == '__main__':
	preprocess()
