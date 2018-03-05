# This program uses a trainind graph to classify video in real time.

# Example execution:
# python habitBreaker.py --fps -1 --notify dog

import numpy as np
import tensorflow as tf
import argparse
import cv2
import time
import os

# For notification
import requests
from win10toast import ToastNotifier

# Set to True if you want to receive notification via windows notification center
windows_notify = False

# For use with Particle Cloud API
particle_notify = False
device_id = 'DEVICE ID GOES HERE'
access_token = 'ACCESS TOKEN GOES HERE'
func_name = 'FUNCTION NAME GOES HERE'
particalArgs = 'FUNCTION ARGUMENTS GOES HERE'

toaster = ToastNotifier()

# Set Capture Device
cap = cv2.VideoCapture(0) 

#Set Width and Height 
cap.set(3,320)
cap.set(4,240)

# Get the command line args
parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=int, default=10, help="Frame per sec. Default: 10")
parser.add_argument("--notify",type=str, default='none', help="The class you want to be notified about.")
args = parser.parse_args()

# The lable of the class you would like to notify you about.
bad_habit_label = args.notify
	
def run_classification():
	
	
	with open('data/retrained_labels.txt', 'r') as fin:
		labels = [line.rstrip('\n') for line in fin]

    # Unpersists graph from file
	with tf.gfile.FastGFile("data/retrained_graph.pb", 'rb') as fin:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(fin.read())
		_ = tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
	
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

		while(True):
			# Read in the image_data
			_, image = cap.read()
			
			#Convert to jpeg format b/c InceptionV3 graph only supports JPEG images
			image = cv2.imencode('.jpg', image)[1].tostring()

			try:
				predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
				prediction = predictions[0]
			except:
				print("Error making prediction.")
				sys.exit()

			# List of predictions
			preds = list(zip(labels,prediction))
			pred = preds.sort(key=lambda x: x[1], reverse=True)
			pred = preds[0][0]

			if pred == bad_habit_label:
				
				prob = np.round(preds[0][1],4)*100
				print('STOP ' + pred.upper() + '!!!     Prob:',str(prob)+'%')
				
				# Send a notification to my lamp to flash red
				if particle_notify == True:
					r = requests.post('https://api.particle.io/v1/devices/'+str(device_id)+'/'+func_name, data={'args': particalArgs, 'access_token':access_token})
				
				# Windos Notification
				if windows_notify == True:
					toaster.show_toast('STOP ' + pred.upper() + '!!!', 'Prob: '+str(prob)+'%')
			
			if args.fps >= 0:
				time.sleep(1/args.fps)

			

if __name__ == '__main__':

	run_classification()
	
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()