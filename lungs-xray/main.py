import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64

# OS path for data set
DATA_PATH = 'C:\data_set'

meta_data = pd.read_csv(os.path.join(DATA_PATH, 'metadata', 'chest_xray_test_dataset.csv'))

for index, row in meta_data.iterrows():

    if not isinstance(row.X_ray_image_name, str):
        break

    if row.Label == 'Normal':
        shutil.move(os.path.join(DATA_PATH, 'test', row.X_ray_image_name), os.path.join(DATA_PATH,'test', CATEGORIES[0], row.X_ray_image_name))
    elif row.Label_1_Virus_category == 'Virus':
        shutil.move(os.path.join(DATA_PATH,'test', row.X_ray_image_name),
                    os.path.join(DATA_PATH,'test', CATEGORIES[1], row.X_ray_image_name))
    else:
        shutil.move(os.path.join(DATA_PATH,'test', row.X_ray_image_name),
                    os.path.join(DATA_PATH,'test', CATEGORIES[2], row.X_ray_image_name))






train_datagen = ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)


training_ds = train_datagen.flow_from_directory(
os.path.join(DATA_PATH, 'train'),
target_size=(64, 64),
batch_size=32,
class_mode='categorical')


validation_ds = test_datagen.flow_from_directory(
os.path.join(DATA_PATH, 'validation'),
target_size=(64,64),
batch_size=32,
class_mode='categorical')

test_ds = test_datagen.flow_from_directory(
os.path.join(DATA_PATH, 'test'),
target_size=(64,64),
batch_size=32,
class_mode='categorical')



# Arhitektura resenja
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Conv2D(64, (3, 3),strides=1, padding='same', activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),strides=1, padding='same', activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.5))

model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])

history = model.fit(
training_ds,
epochs=40,
validation_data=validation_ds)


test_loss, test_acc = model.evaluate(validation_ds, verbose=2)
print('Rezultati za test ds: ')
ls, acc = model.evaluate(test_ds, verbose=2)

model.save('./CNNModel')
model.summary()