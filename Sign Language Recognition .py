#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt


# In[5]:


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Filter data for the specific hand signs representing letters (A to Z), numbers (0 to 25), and emotions
# For example, you can create custom datasets for emotions or use existing emotion datasets.

# Normalize the pixel values to range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Add a channel dimension for grayscale images
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)


# In[8]:


# Define the number of classes based on the combined categories
num_letters = 26
num_numbers = 26
num_emotions = 10  # For example, if you have 10 emotion classes (happy, sad, surprised, etc.)
num_classes = num_letters + num_numbers + num_emotions

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()



# In[9]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_split=0.1)


# In[10]:


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy:.4f}')


# In[11]:


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:





# In[23]:


num_random_samples = 5
random_indices = np.random.randint(0, len(test_images), num_random_samples)
for index in random_indices:
    random_image = test_images[index]
    random_label = test_labels[index]

    prediction = np.argmax(model.predict(np.expand_dims(random_image, axis=0)))

    plt.imshow(random_image.squeeze(), cmap='gray')
    plt.title(f'Predicted: {chr(prediction + 55) if prediction >= 10 else prediction}, True Label: {chr(random_label + 55) if random_label >= 10 else random_label}')
    plt.axis('off')
    plt.show()


# In[ ]:




