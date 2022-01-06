import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

timestep=[]
sunspots=[]

with open('Sunspots.csv') as csvfile:
    reader=csv.reader(csvfile)
    next(reader)
    for line in reader:
        timestep.append(int(line[0]))
        sunspots.append(float(line[2]))


#plt.plot(timestep,sunspots,'--')
#plt.interactive(False)
#plt.show()


timestep=np.array(timestep)
sunspots=np.array(sunspots)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
 #   series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

batch_size=32
window_size=50
shuffle_buffer=4000
trainsize=int(len(sunspots)*0.8)
sunspots_train=sunspots[:trainsize]
timestep_train=timestep[:trainsize]
sunspots_test=sunspots[trainsize:]
timestep_test=timestep[trainsize:]

dataset_train=windowed_dataset(sunspots_train,window_size,batch_size,shuffle_buffer)

model=tf.keras.models.Sequential([tf.keras.layers.Dense(10,input_shape=[window_size],activation='relu'),
                                  tf.keras.layers.Dense(10,activation='relu'),
                                  tf.keras.layers.Dense(1),
                                  tf.keras.layers.Lambda(lambda x:x*200)])
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(optimizer=optimizer,loss=tf.keras.losses.Huber(),metrics=['mae'])
model.fit(dataset_train,batch_size=batch_size,epochs=100)
model.save('sunspotmodel')
dataset_test=windowed_dataset(sunspots_test,window_size,batch_size,shuffle_buffer)
model.evaluate(dataset_test)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

rnn_forecast = model_forecast(model, sunspots, window_size)
rnn_forecast = rnn_forecast[trainsize - window_size:-1, -1]

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


plt.figure(figsize=(10, 6))
plot_series(timestep_test, sunspots_test)
plot_series(timestep_test, rnn_forecast)
plt.show()