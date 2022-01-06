#import requests


#url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
#r = requests.get(url, allow_redirects=True)

#open('C:\\Users\\Anu\\Desktop\\couresera\\catsVsDogs.zip', 'wb').write(r.content)


#import zipfile
#with zipfile.ZipFile('C:\\Users\\Anu\\Desktop\\couresera\\catsVsDogs.zip', 'r') as zip_ref:
#    zip_ref.extractall('C:\\Users\\Anu\\Desktop\\couresera')

import tensorflow as tf
import tensorflow.keras.preprocessing.image as image

train_image_generator=image.ImageDataGenerator(rescale=1./255)
#train_datagen = ImageDataGenerator(
#      rescale=1./255,
#      rotation_range=40,
#      width_shift_range=0.2,
#      height_shift_range=0.2,
#      shear_range=0.2,
#      zoom_range=0.2,
#      horizontal_flip=True,
#      fill_mode='nearest')

print(type(train_image_generator))
validation_image_generator=image.ImageDataGenerator(rescale=1./255)

train=train_image_generator.flow_from_directory('C:\\Users\\Anu\\Desktop\\couresera\\cats_and_dogs_filtered\\train',target_size=(150,150),class_mode='binary',batch_size=20)
validation=validation_image_generator.flow_from_directory('C:\\Users\\Anu\\Desktop\\couresera\\cats_and_dogs_filtered\\validation',target_size=(150,150),class_mode='binary',batch_size=20)
print(type(train))
model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),input_shape=(150,150,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D((2,2)),
                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D((2,2)),
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D((2, 2)),
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(units=512,activation='relu'),
                                 # tf.keras.layers.Dense(units=64,activation='relu'),
                                  tf.keras.layers.Dense(units=1,activation='sigmoid')
                                  ])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics = ['accuracy'])
model.summary()
model.fit_generator(train,steps_per_epoch=10,epochs=10,validation_data=validation)
