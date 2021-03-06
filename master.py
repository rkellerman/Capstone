# Arduino import statements
from nanpy import (ArduinoApi, SerialManager)
from time import sleep

# Deep Learning import statements
import os
from PIL import Image, ImageFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model,load_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import scipy.misc
from skimage import transform
import warnings
import cv2
from picamera import PiCamera

###  BEGIN GLOBAL ARDUINO DEFINITIONS  ###

# define connections and arduino object variables
print "Attempting connection to arduino"
try:
	connection = SerialManager()
	a = ArduinoApi(connection = connection)
	print "Connection to arduino successful"
except:
	print "Failed to connect to Arduino"
	exit()

# define logic control output pins
in1 = 9
in2 = 8
in3 = 7
in4 = 6

# define channel enable output pins
ENA = 10
ENB = 5

### END GLOBAL ARDUINO PIN DEFINITIONS  ###

### BEGIN GLOBAL DEEP LEARNING DEFINITIONS  ###

camera = PiCamera()
dir_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(dir_path, "input/runtime/image.jpg")
cnn = load_model('cnn.h5')

### END GLOBAL DEEP LEARNING DEFINITIONS  ##

### BEGIN ARDUINO HELPER FUNCTIONS

def mForward():
	
	a.digitalWrite(ENA, a.HIGH)
	a.digitalWrite(ENB, a.HIGH)
	
	a.digitalWrite(in1, a.LOW)
	a.digitalWrite(in2, a.LOW)
	a.digitalWrite(in3, a.HIGH)
	a.digitalWrite(in4, a.HIGH)
	
	print "Car moving forward\n"
	
	return

def mBack():

	a.digitalWrite(ENA, a.HIGH)
	a.digitalWrite(ENB, a.HIGH)
	
	a.digitalWrite(in1, a.HIGH)
	a.digitalWrite(in2, a.LOW)
	a.digitalWrite(in3, a.HIGH)
	a.digitalWrite(in4, a.LOW)

	print "Car moving backward\n"
	
	return

def cycle():
	a.digitalWrite(ENA, a.HIGH)
        a.digitalWrite(ENB, a.HIGH)	

	for i in range(2):
		if i == 0:
			a.digitalWrite(in1, a.LOW)
		else:
			a.digitalWrite(in1, a.HIGH)
	
		for j in range(2):
			if j == 0:
				a.digitalWrite(in2, a.LOW)
			else:
				a.digitalWrite(in2, a.HIGH)

			for k in range(2):
				if k == 0:
					a.digitalWrite(in3, a.LOW)
				else:
					a.digitalWrite(in3, a.HIGH)

				for l in range(2):
					if l == 0:
						a.digitalWrite(in4, a.LOW)
					else:
						a.digitalWrite(in4, a.HIGH)

					print i
					print j
					print k
					print l
					print '\n'	
					sleep(3)


def stopCar():
	
	a.digitalWrite(ENA, a.LOW)
	a.digitalWrite(ENB, a.LOW)
	
	return;

def ArduinoSetup():

	a.pinMode(in1, a.OUTPUT)
	a.pinMode(in2, a.OUTPUT)
	a.pinMode(in3, a.OUTPUT)
	a.pinMode(in4, a.OUTPUT)
	a.pinMode(ENA, a.OUTPUT)
	a.pinMode(ENB, a.OUTPUT)
	
	print "Arduino pin setup complete\n"

	return


###  END ARDUINO HELPER FUNCTIONS  ###

def moveCarForward(amount):
	
	mForward()
	sleep(amount)
	stopCar()
		
	return;

def takePicture():
	
	stopCar()
	
	# take picture
	camera.start_preview()
	sleep(1)
	camera.capture(img_path)
	camera.stop_preview()
	
	# load picture
	im = Image.open(img_path).convert("RGB")
	im.load();
	im = np.asarray(im, dtype = "float32")
	im = im/255
	im = transform.resize(im, (49, 49));
	check = np.array([im])
		
	return check;


def classifyWeedOrCrop(pic):
	return "crop"

def classifyCrop(pic):
	
	prediction = cnn.predict(pic)

	return prediction[0]

def process():
	pic = takePicture();
	result = classifyWeedOrCrop(pic);
	if (result == "weed"):
		print "weed";
	else:
		print "crop"
		cropType = classifyCrop(pic);
		print cropType;


def main():
	
	ArduinoSetup()
	cycle()



	amount = 3; # seconds
	while True:
		moveCarForward(amount);
		process();

	print("we are done")

main();


