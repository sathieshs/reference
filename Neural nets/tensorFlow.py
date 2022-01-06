import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_house=160
np.random.seed(42)
house_size=np.random.randint(low=1000,high=3500,size=num_house)

house_price=house_size*100+np.random.randint(low=20000,high=70000,size=num_house)

plt.plot(house_size,house_price,'bx')
plt.ylabel('Price')
plt.xlabel('size')
plt.show()

def normalize(data):
    return (data-data.mean())/data.std()

num_train_samples=math.floor(num_house*0.7)
train_house_size=house_size[:num_train_samples]
train_house_price=house_size[:num_train_samples]

train_house_size_norm=normalize(train_house_size)
train_house_price_norm=normalize(train_house_price)

test_house_size=house_size[num_test_samples:]
test_house_price=house_size[num_test_samples:]

test_house_size_norm=normalize(test_house_size)
test_house_price_norm=normalize(test_house_price)


tf_house_size=tf.placeholder('float',name='house_size')
tf_house_price=tf.placeholder('float',name='price')
