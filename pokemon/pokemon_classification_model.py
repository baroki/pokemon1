
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

#  Veri yolları

train_dir = 'C:/Users/utkus/Desktop/pokemon/train'
test_dir = 'C:/Users/utkus/Desktop/pokemon/test'

#  Veri ön işleme ve artırma

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

#  Model oluşturma

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

#  Modeli derleme

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)



Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)


y_true = test_generator.classes


cm = confusion_matrix(y_true, y_pred)

#  Sensitivity ve Specificity hesaplama

TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (FP + FN + TP)

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f'Sensitivity: {np.mean(sensitivity)}')
print(f'Specificity: {np.mean(specificity)}')
