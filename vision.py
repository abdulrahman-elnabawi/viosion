import cv2
import numpy as np
from scipy.ndimage import maximum_filter , minimum_filter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models


img = cv2.imread(r"C:\Users\interface\Desktop\viosion\img5.jpg")

#===2D Convolution( Image Filtering )

# kernal = np.ones((7, 7), np.float32)/49
# dst = cv2.filter2D(img, -1, kernal)
# cv2.imshow('Or', img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#====Image Blurring(Image Smoothing)

# blur =cv2.blur(img, (7,7))
# cv2.imshow('Or', img)
# cv2.imshow('blur', blur)
# cv2.waitKey()
# cv2.destroyAllWindows()


#=====Gaussian Filtering
# Gaussian = cv2.GaussianBlur(img, (5,5), 0)
# cv2.imshow('Or', img)
# cv2.imshow('Gaussian', Gaussian)
# cv2.waitKey()
# cv2.destroyAllWindows()


#====Median Filtering
# median = cv2.medianBlur(img, 5)
# cv2.imshow('Or', img)
# cv2.imshow('median', median)
# cv2.waitKey()
# cv2.destroyAllWindows()


#====negative Filtering
# negative = 255-img
# cv2.imshow('Or', img)
# cv2.imshow('negative', negative)
# cv2.waitKey()
# cv2.destroyAllWindows()


#====Thresholding or gray scale
# t=127
# ret, thresh1 = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
# cv2.imshow('Or', img)
# cv2.imshow('thresh1', thresh1)
# print(ret)
# cv2.waitKey()
# cv2.destroyAllWindows()


#====adaptive thresholding
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imshow('Or', img)
# cv2.imshow('thresh1', th2)
# cv2.waitKey()
# cv2.destroyAllWindows()

#=================slide 2====================

###laplacian filter

# filterdimg=cv2.Laplacian(img, cv2.CV_64F, ksize=3)
# abs_filterdimg = cv2.convertScaleAbs(filterdimg)
# cv2.imshow('Or', img)
# cv2.imshow('laplacian', abs_filterdimg)
# cv2.waitKey()
# cv2.destroyAllWindows()

###highh boost filter formula

# blured = cv2.GaussianBlur(img, (5,5), 0)
# A=.5
#high_boost = cv2.addWeighted(img,A,blured, (A-1) ,0)
# cv2.imshow('Or', img)
# cv2.imshow('blured', blured)
# cv2.imshow('high_boost', high_boost)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###sobel edge detection

# blured = cv2.GaussianBlur(img, (5,5), 0)
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# sobelx_abs = cv2.convertScaleAbs(sobelx)
# sobely_abs = cv2.convertScaleAbs(sobely)
# sobel = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)
# cv2.imshow('Or', img)
# cv2.imshow('sobelx', sobelx)
# cv2.imshow('sobely', sobel)
# cv2.imshow('sobely', sobely)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####inverse fourier transform

# f=np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift)+1)
# magnitude_spectrum = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
# fishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(fishift)
# img_back = np.abs(img_back)
# img_back = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
# cv2.imshow('Or', img)
# cv2.imshow('magnitude_spectrum', magnitude_spectrum)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ######buttersworth filter
# rows, cols = img.shape[:2]
# crow, ccol = rows//2, cols//2
# mask = np.zeros((rows, cols), np.float32)
# cutoff = 30
# order = 2
# for i in range(rows):
#     for j in range(cols):
#         d = np.sqrt((i-crow)**2 + (j-ccol)**2)
#         mask[i, j] = 1/(1+(d/cutoff)**(2*order))
# filterdft=fshift*mask
# f_ishift = np.fft.ifftshift(filterdft)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)

# cv2.imshow('Or', img)
# cv2.imshow('img_back', img_back)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##Gaussian high pass filter
# gaussian_high_pass =  np.zeros((rows, cols), np.float32)
# for i in range(rows):
#     for j in range(cols):
#         d = np.sqrt((i-crow)**2 + (j-ccol)**2)
#         gaussian_hpf_mask = 1-np.exp(-1*(d**2)/(2*(30**2)))
# filterdft=fshift*gaussian_hpf_mask
# # Convert filterdft to magnitude spectrum for display
# magnitude_spectrum = 20 * np.log(np.abs(filterdft) + 1)
# magnitude_spectrum = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))

# cv2.imshow('Or', img)
# cv2.imshow('magnitude_spectrum', magnitude_spectrum)
# cv2.imshow('gaussian_hpf_mask', gaussian_high_pass)
# cv2.waitKey()
# cv2.destroyAllWindows()


# #############the mean filter
# nois_sigma = 25
# gaussian_noise = np.random.normal(0, nois_sigma, img.shape).astype(np.float32)
# noisy_img = img + gaussian_noise
# filter_size = 15
# dft = np.fft.fft2(noisy_img)
# dft_shift = np.fft.fftshift(dft)
# row, cols = img.shape[:2]
# crow, ccol = row//2, cols//2
# mask = np.zeros((row, cols), np.float32)
# mask[crow-filter_size//2:crow+filter_size//2+1, ccol-filter_size//2:ccol+filter_size//2+1] = 1/((filter_size)**2)
# filterdft = dft_shift*mask
# img_back = np.fft.ifft2(filterdft)
# img_back = np.abs(img_back)
# img_back = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
# cv2.imshow('Or', img)
# cv2.imshow('noisy_img', noisy_img)
# cv2.imshow('img_back', img_back)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Set the filter size
# filter_size = 20

# dft = np.fft.fft2(noisy_img)
# dft_shift = np.fft.fftshift(dft)

# magnitude_spectrum = np.abs(dft_shift)
# phase_spectrum = np.angle(dft_shift)

# max_filtered_magnitude = maximum_filter(magnitude_spectrum, size=filter_size)
# max_filtered_phase = minimum_filter(phase_spectrum, size=filter_size)

# max_filter_dft = max_filtered_magnitude * np.exp(1j * max_filtered_phase)
# min_filter_dft = magnitude_spectrum * np.exp(1j * phase_spectrum)

# min_filter_dft_magnitude = np.abs(min_filter_dft)
# min_filter_dft_magnitude = np.uint8(cv2.normalize(min_filter_dft_magnitude, None, 0, 255, cv2.NORM_MINMAX))

# cv2.imshow('lol', min_filter_dft_magnitude)
# cv2.imshow('Or', img)
# cv2.imshow('noisy_img', noisy_img)
# cv2.imshow('MAX',max_filtered_magnitude )

# cv2.waitKey(0)
# cv2.destroyAllWindows()


######## Band reject filter
# def bandreject_filter(shape, d0=160, w=20, ftype='butterworth', n=2):

# 	P, Q = shape
# 	# Initialize filter with ones
# 	H = np.ones((P, Q))

# 	# Traverse through filter
# 	for u in range(0, P):
# 		for v in range(0, Q):
# 			# Get euclidean distance from point D(u,v) to the center
# 			D_uv = np.sqrt((u - (P/2))**2 + (v - (Q/2))**2)
# 			if ftype == 'ideal':

# 				if (d0 - (w/2)) <= D_uv <= (d0 + (w/2)):
# 					H[u, v] = 0.0

# 			elif ftype == 'butterworth':

# 				if D_uv == d0: 
# 					H[u, v] = 0
# 				else:
# 					H[u, v] = 1/(1 + ((D_uv*w)/(D_uv**2 - d0**2))**(2*n))

# 			elif ftype == 'gaussian':

# 				if D_uv == 0:
# 					H[u, v] = 1
# 				else:
# 					H[u, v] = 1.0 - np.exp(-((D_uv**2 - d0**2) / (D_uv * w))**2)

# 	return H
# # Load the image

# P, Q = img.shape[:2]

# # Fourier Transform and shift
# dft = np.fft.fft2(img)
# dft_shifted = np.fft.fftshift(dft)

# # Create an ideal band reject filter
# d0, w = 160, 20
# H = np.ones(dft_shifted.shape)
# for u in range(dft_shifted.shape[0]):
#     for v in range(dft_shifted.shape[1]):
#         D_uv = np.sqrt((u - (dft_shifted.shape[0] / 2))**2 + (v - (dft_shifted.shape[1] / 2))**2)
#         if (d0 - (w / 2)) <= D_uv <= (d0 + (w / 2)):
#             H[u, v] = 0.0

# # Apply filter in frequency domain
# filtered_dft = dft_shifted * H
# filtered_dft_1 = np.fft.ifftshift(filtered_dft)

# # Inverse Fourier Transform to get the filtered image
# filtered_img = np.abs(np.fft.ifft2(filtered_dft_1))


# plt.figure(figsize=(20, 15))
# plt.subplot(1, 4, 1), plt.imshow(img, cmap='gray'), plt.title("Original Image")
# plt.subplot(1, 4, 2), plt.imshow(np.log(1 + np.abs(H)), cmap='gray'), plt.title("Band Reject Filter")
# plt.subplot(1, 4, 3), plt.imshow(np.log(1 + np.abs(dft_shifted)), cmap='gray'), plt.title("Frequency Image")
# plt.subplot(1, 4, 4), plt.imshow(filtered_img, cmap='gray'), plt.title("Filtered Image")
# plt.show()

#######erosion and dilation open and close

#_, binary_image = cv2.threshold(img, 127,  255 ,cv2.THRESH_BINARY_INV)
# kernel = np.ones((5, 5), np.uint8) 

# iter = 7
# res = cv2.erode(binary_image, kernel, iterations=iter) 

# plt.figure(figsize=(8, 5))

# plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
# plt.axis("off")
# plt.subplot(122), plt.imshow(res, cmap='gray'), plt.title('Image after erosion')
# plt.axis("off")
# plt.show()

# #=========dilation
# kernel = np.ones((5, 5), np.uint8) 

# img_dilation = cv2.dilate(img, kernel, iterations=1) 

# plt.figure(figsize=(7, 5))
# plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
# plt.axis("off")
# plt.subplot(122), plt.imshow(img_dilation, cmap='gray'), plt.title('Image after dilation')
# plt.axis("off")
# plt.show()

# #=========opening
# kernel = np.ones((5, 5), np.uint8) 

# img_erosion = cv2.erode(img, kernel, iterations= 4) 
# img_dilation = cv2.dilate(img_erosion, kernel, iterations=5) 

# plt.figure(figsize=(15, 5))
# plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
# plt.axis("off")
# plt.subplot(132), plt.imshow(img_erosion, cmap='gray'), plt.title('image after erosion')
# plt.axis("off")
# plt.subplot(133), plt.imshow(img_dilation, cmap='gray'), plt.title(' Image after opening')
# plt.axis("off")
# plt.show()

# #=========closing


# kernel = np.ones((5, 5), np.uint8) 

# img_dilation = cv2.dilate(img, kernel, iterations=5) 
# img_erosion = cv2.erode(img_dilation, kernel, iterations= 4) 


# plt.figure(figsize=(15, 5))
# plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
# plt.axis("off")
# plt.subplot(132), plt.imshow(img_dilation, cmap='gray'), plt.title('image after dilation')
# plt.axis("off")
# plt.subplot(133), plt.imshow(img_erosion, cmap='gray'), plt.title(' Image after closing')
# plt.axis("off")
# plt.show()


#####svm


# data = pd.read_csv( ".csv")
# X = data.drop(columns=["Label"])
# y = data['Label']
# # 3-Visualize some digits
# fig, axes = plt.subplots(1, 2, figsize=(5, 3))
# for i, ax in enumerate(axes):
#     ax.imshow(X.iloc[i].to_numpy().reshape(128, 128),cmap='gray')
#     ax.axis('off')
# plt.show()
# # 4- Split data into traine and teste 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# # Initialize the SVM classifier
# svm_classifier = SVC(kernel='linear')

# # Train the classifier on the training data
# svm_classifier.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = svm_classifier.predict(X_test)

# # Evaluate the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy of SVM classifier: {accuracy * 100:.2f}%")


#########neural network
# X=np.array([[1,3,4,5,6],
#     [2,1,0,6,7],
#     [3,1,1,7,8],
#     [2,3,1,4,5],
#     [4,0,0,1,5]])
# y=np.array([[1],[0],[0],[1],[1]])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Build the neural network model
# model = models.Sequential([
#     layers.Dense(4, activation='relu'),   # Hidden layer with 4 neurons
#     layers.Dense(1, activation='sigmoid'),
# ])

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=20)

# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f'\nTest accuracy: {test_acc:.4f}')
# data = pd.read_csv("data.csv")
# X1 = data.drop(columns=['Label'])
# y1 = data['Label']
# X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3)

# # Build the neural network model
# model = models.Sequential([
#     layers.Dense(28, activation='relu'),   
#     layers.Dense(28, activation='relu'),   
#     layers.Dense(1, activation='sigmoid'),  
# ])
# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=20)

# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f'\nTest accuracy: {test_acc:.4f}')

# # Load MNIST dataset
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# # Preprocess the data
# X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize to [0, 1]

# model = models.Sequential([
#     layers.Flatten(input_shape=(28, 28)),
#     layers.Dense(36, activation='relu'),   
#     layers.Dense(36, activation='relu'),   
#     layers.Dense(10, activation='softmax'),  
# ])

# # Compile the model
# model.compile(optimizer='adam',                        # Optimizer
#               loss='sparse_categorical_crossentropy',  # Loss function for integer labels (0-9)
#               metrics=['accuracy'])                    # Track accuracy during training

# # Train the model
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # Evaluate the model on test data
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_acc:.4f}")

