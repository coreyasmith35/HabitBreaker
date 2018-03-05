# This program prints a single prediction of an image
# To Run:
# python singlePred.py --img data/singlePred/img.jpg

import sys
import tensorflow as tf
import argparse

# Get the command line args
parser = argparse.ArgumentParser()
parser.add_argument("--img", help="location of image")
args = parser.parse_args()
	
# Get the list of lables	
with open('data/retrained_labels.txt', 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
		
# Unpersists graph from file
with tf.gfile.FastGFile("data/retrained_graph.pb", 'rb') as fin:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(fin.read())
	_ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	# Read in the image_data
	image_data = tf.gfile.FastGFile(args.img, 'rb').read()

	# Make the prediction
	try:
		predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
		prediction = predictions[0]
	except:
		print("Error could not make prediction. Something went wrong.")
		sys.exit()

	# List of predictions
	preds = list(zip(labels,prediction))
	preds.sort(key=lambda x: x[1], reverse=True)
	print(preds)
	print('Prediction:',preds[0][0])

