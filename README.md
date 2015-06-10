# FlowerRecognization
PR/IP Project

#RENAME THE DATASET

import os

fp = open("label.txt")

image_index = []

image_index = fp.readline().split(';')

image_class = fp.readline().split(';')

pwd = os.getcwd()

for i in range(1,len(image_index) + 1):
	if i < 10:
		num_0 = '0000'
	elif i < 100:
		num_0 = '000'
	elif i < 1000:
		num_0 = '00'
	else:
		num_0 = '0'  


	old = pwd + '/image_' + num_0 + str(i) + '.jpg'
	new = pwd + '/image_' + num_0 + str(i) + '_class_' + str(int(image_class[i - 1])) + '.jpg'
	os.rename(old, new)