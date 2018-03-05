# Program to get some training or test data for image clasification

# Example to excute:
# python captureData.py --split train --classification dog -- fps 10

import numpy as np
import cv2
import time
import os
import argparse

# Get the command line args
parser = argparse.ArgumentParser()
parser.add_argument("--split", default='train', help="Split type(ie. train,test)")
parser.add_argument("--classification", default='class1', help="Class name of the object")
parser.add_argument("--fps", type=int, default=3, help="Frame per sec. Default: 3")
parser.add_argument("--one_img", default='false', help="Capture one image")

args = parser.parse_args()

if args.one_img == 'true':
	dir = 'data/single_imgs'
else:
	dir = 'data/'+args.split+'/'+args.classification

# Make dir if necessary
if not os.path.isdir(dir):
	os.makedirs(dir)

# Set Capture Device
cap = cv2.VideoCapture(0) 

#Set Width and Height (320,240)
cap.set(3,320)
cap.set(4,240)

# Warmup...
time.sleep(2)

while(True):
	# Capture frame-by-frame
	ret, img = cap.read()
	
	if args.one_img == 'true':
		cv2.imwrite(os.path.join(dir , 'img.jpg'),img)
		cv2.imshow('frame',cv2.flip( img, 1 ))
		print('Single image taken!')
		print('Saved to: '+dir + '/img.jpg')
		break
	
	stamp = str(time.time())

	cv2.imwrite(os.path.join(dir , args.classification+stamp+'.jpg'),img)
	
	cv2.imshow('frame',cv2.flip( img, 1 ))
	
	time.sleep(1/args.fps)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()