# To Capture Frame
import cv2
from cv2 import normalize

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf

model = tf.keras.models.load_module("keras_model.h5")

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		img = cv2.resize(frame,(224,244))

		
		# expand the dimensions
		test_imgage= np.array(img,dtype=np.float32)
		test_imgage= np.expand_dims(test_imgage,axis=0)
		
		# normalize it before feeding to the model
		normalize_image = test_imgage/255.0
		
		# get predictions from the model
		pridection = model.predict(normalize_image)
		print("prediction",pridection)
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
