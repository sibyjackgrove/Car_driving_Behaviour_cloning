import tensorflow as tf
import csv as csv
import cv2
import numpy as np
import time as time
import os.path
print(tf.__version__)

#Get path names for training images from csv file
lines =[]
csv_file = "driving_log.csv"
with open(csv_file) as csvfile:     #Extract csv file
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print("Sample data from file:\n",lines[0])

#Separate images and measurements
images=[]
measurements =[]

for line in lines:
    source_path=line[0]
    #print(source_path)
               
    filename = source_path.split('center')[-1] #for use in AWS instance
    #print(filename)
    current_path = 'IMG/' + 'center' + filename       #for use in AWS instance
    #current_path =  source_path #+  folder        #for use in Local machine
    if os.path.isfile(current_path) == True:
        #print(current_path)
        image_BGR = cv2.imread(current_path)            #Import image using cv2 library
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB) #Convert BGR to RGB
        images.append(image)                 #Append image to list of images
        measurement = float(line[3])               #Import steering angle 
        measurements.append(measurement)         #Append to list of steering angle

#Convert to numpy array
X_train = np.array(images)       
images = None
y_train = np.array(measurements)
measurement = None

# TODO: Number of training examples
assert(len(X_train) == len(y_train))  #Check whether number of samples of input is equal to number of samples of outputs
n_train =len(X_train)
print("Image data set =",X_train.shape)
print("Number of training examples =", n_train)

### Data exploration visualization code goes here.
import random
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
index = random.randint(0, len(X_train))  #Find random image from dataset
image = X_train[index].squeeze()   


plt.figure(figsize=(1,1))
plt.imshow(image)
print("Corresponding Steering angle =",y_train[index])   #Corresponding Steering angle

def resize_img(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 66),method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

use_existing = False
if os.path.isfile('model.h5') == True and use_existing == True:           #Check if previous model exists
    print("Model Exists...loading")
    existing_model = tf.contrib.keras.models.load_model('model.h5')
    print("Model Loaded")
    existing_model.summary()
    #print(load_model.get_config())
else:
    print("Using New Model")
    #Define model using Keras
    drop_rate = 0.5
    #bias0 = tf.contrib.keras.initializers.glorot_uniform()   #Select different initializer
    #bias0 = tf.contrib.keras.initializers.RandomUniform() 
    bias0 =  'zeros'
    #Model definition starts
    model = tf.contrib.keras.models.Sequential() 
    model.add(tf.contrib.keras.layers.InputLayer(input_shape=(160,320,3),name='InputLayer'))  #Input
    model.add(tf.contrib.keras.layers.Cropping2D(cropping=((50,20),(0,0)),name='CropImage'))  #Cropping image
    model.add(tf.contrib.keras.layers.Lambda(resize_img,name='ResizeImage'))          #Resizing image
    model.add(tf.contrib.keras.layers.Lambda(lambda x:x/255.0 - 0.5,name='Normalizing'))  #Normalizing values
    
	#Convolutional neural network starts
    model.add(tf.contrib.keras.layers.Conv2D(filters=4,kernel_size=(5,5),strides=(2,2),activation='relu',bias_initializer=bias0,
                                             name='ConvLayer1'))

    model.add(tf.contrib.keras.layers.Conv2D(filters=6,kernel_size=(5,5),strides=(2,2),activation='relu',bias_initializer=bias0,
                                             name='ConvLayer2'))

    model.add(tf.contrib.keras.layers.Conv2D(filters=8,kernel_size=(5,5),strides=(2,2),activation='relu',bias_initializer=bias0,
                                             name='ConvLayer3'))
    model.add(tf.contrib.keras.layers.Dropout(drop_rate,name='DropOut1'))
    model.add(tf.contrib.keras.layers.Conv2D(filters=10,kernel_size=(5,5),strides=(2,2),activation='relu',bias_initializer=bias0,
                                             name='ConvLayer4'))
    #model.add(tf.contrib.keras.layers.Dropout(drop_rate,name='DropOut1'))
    model.add(tf.contrib.keras.layers.Flatten(name='Flatten'))
    #model.add(tf.contrib.keras.layers.Dropout(drop_rate))
    model.add(tf.contrib.keras.layers.Dense(units=10,bias_initializer=bias0,name='FeedForward1'))
    model.add(tf.contrib.keras.layers.Activation('relu',name='ReLU1'))
    #model.add(tf.contrib.keras.layers.Dropout(drop_rate,name='DropOut1'))
    model.add(tf.contrib.keras.layers.Dense(units=5,bias_initializer=bias0,name='FeedForward2'))
    model.add(tf.contrib.keras.layers.Activation('relu',name='ReLU2'))
    #model.add(tf.contrib.keras.layers.Dropout(drop_rate,name='DropOut3'))
    model.add(tf.contrib.keras.layers.Dense(units=1,bias_initializer=bias0,name='OutputLayer'))
	
	#Compile model to use mse loss function and Adam optimizer
    model.compile(loss='mse',optimizer='adam')
    model.summary()

#Training starts
t = time.time()
model.fit(X_train,y_train,batch_size=32,epochs=3,validation_split = 0.2,shuffle =True)
print("Time: %.3f minutes" % ((time.time() - t)/60))
model.save('model.h5')
print("Model Saved")

#Check predicted steering angle
index = random.randint(0, len(X_train))  #Find random image from dataset
samples=1
test_image = X_train[index:index+samples]
image = X_train[index].squeeze()   
plt.figure(figsize=(1,1))
plt.imshow(image)

print("Corresponding Steering angle =",y_train[index:index+samples])   #Corresponding Steering angle
print("Predicted Steering angle:",model.predict(test_image,batch_size=1))


