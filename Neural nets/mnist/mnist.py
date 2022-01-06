import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_input_data("mnist/",one_hot=True)

x=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,[None,10])