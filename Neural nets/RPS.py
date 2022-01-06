import matplotlib.pyplot as plt

import tensorflow as tf

traindir='C:\\Users\\Anu\\Desktop\\rps\\rps\\rps'
testdir='C:\\Users\\Anu\\Desktop\\rps\\rps-test-set\\rps-test-set'

train_imageDatagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_imageDatagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_image_data=train_imageDatagen.flow_from_directory(traindir,class_mode='categorical',
                          batch_size=10,target_size=(150,150))

test_image_data=test_imageDatagen.flow_from_directory(testdir,class_mode='categorical',
                          batch_size=10,target_size=(150,150))
input=tf.keras.layers.Input(shape=(150,150,3))
x=tf.keras.layers.Conv2D(32,(3,3),activation='relu')(input)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.MaxPool2D((2,2))(x)

x=tf.keras.layers.Conv2D(64,(3,3),activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.MaxPool2D((2,2))(x)

x=tf.keras.layers.Conv2D(128,(3,3),activation='relu')(x)
x=tf.keras.layers.MaxPool2D((2,2))(x)

x=tf.keras.layers.Flatten()(x)

x=tf.keras.layers.Dense(units=128,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(units=128,activation='relu')(x)
output=tf.keras.layers.Dense(units=3,activation='softmax')(x)

model=tf.keras.Model(inputs=[input],outputs=[output])

model.compile(loss='categorical_crossentropy',metrics=['acc'])
model.summary()
history=model.fit_generator(train_image_data,validation_data=test_image_data,epochs=20,steps_per_epoch=84)

acc=history.history['acc']
loss=history.history['val_acc']
loss     = history.history['loss']
val_loss = history.history['val_loss' ]
epochs   = range(len(acc))

plt.plot(acc,epochs)
plt.plot(loss,epochs)
plt.figure()

plt.plot(loss,epochs)
plt.plot(val_loss,epochs)
