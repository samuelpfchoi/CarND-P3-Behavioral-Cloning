import csv
import cv2
import numpy as np


#### Loading Training Data
#
images = []
measurements = []

# 1st dataset
lines = []
with open('./data_01/driving_log.csv') as csvfile:
    csvreader = csv.reader(csvfile)

    header_line = next(csvreader)

    for line in csvreader:
        lines.append(line)

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data_01/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# 2th dataset (training case: driving back to lane center)
lines = []
with open('./data_02/driving_log.csv') as csvfile:
    csvreader = csv.reader(csvfile)

    header_line = next(csvreader)

    for line in csvreader:
        lines.append(line)

for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './data_02/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#### Model Architecture (NVIDIA Architecture)
#
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#### Training the Model
#
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

#### Saving Model
#
model.save('model.h5')

