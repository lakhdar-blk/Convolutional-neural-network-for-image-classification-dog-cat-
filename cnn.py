import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os

from sklearn.preprocessing import LabelEncoder


# Load and preprocess the images
def load_images(directory, img_size):
   
    images = []
    labels = []

    for label in os.listdir(directory):

        label_path = os.path.join(directory, label)

        for image_name in os.listdir(label_path):

            image_path = os.path.join(label_path, image_name)    
            image = cv2.imread(image_path)
             
            try:

                image = cv2.resize(image, (img_size, img_size))
                image2 = cv2.flip(image, 0)
                
                image2 = np.array(image2)
                image = np.array(image)

                images.append(image)
                labels.append(label)

                images.append(image2)
                labels.append(label)
            
            except:

                os.remove(image_path)
                continue

    return np.array(images), np.array(labels)

# Load and preprocess the data
img_size = 84
directory = "dataset2" # Update with your own dataset directory
images, labels = load_images(directory, img_size)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = np_utils.to_categorical(labels, num_classes=2)

images = images/255



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))

model.add(Dense(2, activation='softmax')) # Update with 2 output units for cat and dog

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Loss: ", loss)
print("Accuracy: ", accuracy)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.show()

#model.save("cnn_imgc2.model")