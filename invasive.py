import os, csv
from PIL import Image, ImageFile
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
# import keras
# from keras.models import Sequential,Input,Model,load_model
# from keras.layers import Dense,Dropout,Flatten
# from keras.layers import Conv2D,MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
# import scipy.misc
from skimage import transform
import warnings
import cv2

warnings.filterwarnings("ignore")

dir = "./input/invasive"
train_dir = dir + "/train"

records = []

with open(dir + "/train_labels.csv", 'rb') as f:
	reader = csv.reader(f);
	lines = [row for row in reader][1:];
	records = [[str(x) + ".jpg",y] for x,y in lines]

df_train = pd.DataFrame.from_records(records, columns = ['image', 'category'])

X = []
count = 0
bad_images = []
for i in (train_dir + "/" + df_train["image"]):
	img = Image.open(i)
	img.load()
	img = np.asarray(img, dtype = "float32")
	img = img/255
	data = transform.resize(img, (49, 49))
	if data.size != 7203:
		bad_images.append(count)
	count = count + 1

df_train = df_train.drop(df_train.index[bad_images]);

for i in (train_dir + "/" + df_train['image']):
	img = Image.open(i).convert('RGB')
	img = np.asarray(img, dtype='float32')
	img = img/255
	data = transform.resize(img,(49,49))
	X.append(data)

X = np.array(X)

y = np.array(df_train['category'].astype('category').cat.codes)


