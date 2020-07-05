import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
CATEGORIES = ['normal', 'virus', 'bacteria']


# OS path for data set
DATA_PATH = 'C:\data_set'

#Loading the data, normalization and later will add data augmentation

meta_data = pd.read_csv(os.path.join(DATA_PATH, 'metadata', 'chest_xray_test_dataset.csv'))

# Ovo je bio kod za razvrstavanje slika po folderima (klasama) za lakse ucitavanje i labele kasnije
# za 23 slike nisu nadjeni meta podaci, a 2 slike vezane za pusenje su takodje maknute
#
# for index, row in meta_data.iterrows():
#
#     if not isinstance(row.X_ray_image_name, str):
#         break
#
#     if row.Label == 'Normal':
#         shutil.move(os.path.join(DATA_PATH, 'test', row.X_ray_image_name), os.path.join(DATA_PATH,'test', CATEGORIES[0], row.X_ray_image_name))
#     elif row.Label_1_Virus_category == 'Virus':
#         shutil.move(os.path.join(DATA_PATH,'test', row.X_ray_image_name),
#                     os.path.join(DATA_PATH,'test', CATEGORIES[1], row.X_ray_image_name))
#     else:
#         shutil.move(os.path.join(DATA_PATH,'test', row.X_ray_image_name),
#                     os.path.join(DATA_PATH,'test', CATEGORIES[2], row.X_ray_image_name))



model = keras.models.load_model('./CNNModel')
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_PATH, 'test'),
    labels='inferred',
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_PATH, 'data'),
    labels='inferred',
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_PATH, 'data'),
    validation_split=0.2,
    labels='inferred',
    subset="validation",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


model.summary()
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
ls, acc = model.evaluate(test_ds, verbose=2)
print(acc, ' je accuracy')



# ucitavanje slika pomocu image loadera iz kerasa

x_train = []
y_train = []





from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
'chest_xraybinary/data',
target_size=(64, 64),
batch_size=32,
class_mode='categorical')
test_set = test_datagen.flow_from_directory(
'chest_xraybinary/test',
target_size=(64,64),
batch_size=32,
class_mode='categorical')
classifier.fit_generator(
training_set,
epochs=25,
validation_data=test_set)
# Funkcija je dostupna u tensor flow nighlty buildu.
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     os.path.join(DATA_PATH, 'data'),
#     labels='inferred',
#     validation_split=0.2,
#     subset="training",
#     seed=1337,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     os.path.join(DATA_PATH, 'data'),
#     validation_split=0.2,
#     labels='inferred',
#     subset="validation",
#     seed=1337,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

#train_ds = train_ds.map(lambda images, labels : print(images[0]))

#val_ds = val_ds.map(lambda images, labels: images.map(lambda x: x/255.))

# dovodimo piksele u range 0 - 1


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

plt.show()


# Arhitektura resenja
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3),strides=1, padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),strides=1, padding='same', activation='relu'))

model.add(layers.Conv2D(80, (3, 3),strides=1, padding='same', activation='relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(3, activation='softmax'))

# koristimo advanced adam optimizator, da nam u prvoj iteraciji ne bude premala vrijednost zbog momenta koji je inicijalno 0
# koristimo SparseCategoricalCrossentropy jer cemo klase predstaviti kao integere, i imamo 3 klase

model.compile(optimizer='adamax',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])



history = model.fit(train_ds, epochs=30,
                    validation_data=val_ds)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print('aa sad test')
ls, acc = model.evaluate(test_ds, verbose=2)

plt.plot(history.history['loss'], label=' (training data)')
plt.plot(history.history['val_loss'], label=' (validation data)')
plt.ylabel(' value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

model.save('./CNNModel')
model.summary()