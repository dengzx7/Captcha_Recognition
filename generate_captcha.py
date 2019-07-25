# generate_captcha.py
'''
从中山大学本科教务系统爬取验证码 'https://cas.sysu.edu.cn/cas/captcha.jsp'
Download captcha images from SYSU's CAS by using python requests
'''

import requests
import urllib
import os

def getImage():
	'''
	从中山大学本科教务系统爬取验证码
	Download captcha images from SYSU's CAS
	'''
	'''
	**********************************************
	**             Set up directory             **
	**********************************************
	'''
	root = 'dataset'

	# If root already exists
	if os.path.isdir(root):
		print (root, 'directory exists')
		return

	os.mkdir(root)

	# If root failed to be built
	if not os.path.isdir(root):
		print (root, 'directory failed to be built')
		return

	# Set up directory for 'images' and 'labels'
	images = os.path.join(root, 'images')
	os.mkdir(images)
	labels = os.path.join(root, 'labels')
	os.mkdir(labels)

	'''
	**********************************************
	**             Download images              **
	**********************************************
	'''
	url = 'https://cas.sysu.edu.cn/cas/captcha.jsp'
	total = 920
	print ('images download begin...')
	for i in range(1, total + 1):
		if i < 10:
			target = '000'
		elif i < 100:
			target = '00'
		elif i < 1000:
			target = '0'
		print ('download images', i)
		urllib.request.urlretrieve(url, images + '\\' + target + str(i) + '.png')
	print ('images download complete!')

if __name__ == '__main__':
	getImage()
