import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

print(tf.__version__)

imdb,info=tfds.load('imdb_reviews',with_info=True,as_supervised=True)

train_data,test_dfata=imdb['train'],imdb['test']

train_sentences=[]
train_labels=[]

test_sentences=[]
test_labels=[]

for s,l in train_data:
    train_sentences.append(str(s.numpy()))
    train_labels.append(l.numpy())

for s,l in test_dfata:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

OOV_str='<OOV>'
dim=3
vocab_size=10000
max_len=120
trunc_type='post'

from  tensorflow.keras.preprocessing.text import Tokenizer
from  tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer=Tokenizer(num_words=vocab_size,oov_token=OOV_str)
tokenizer.fit_on_texts(train_sentences)
seq=tokenizer.texts_to_sequences(train_sentences)
word_index = tokenizer.word_index

padded=pad_sequences(seq,maxlen=max_len,truncating=trunc_type)


test_seq=tokenizer.texts_to_sequences(test_sentences)
test_pad=pad_sequences(test_seq,maxlen=max_len,truncating=trunc_type)

#model=tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size,dim,input_length=max_len),
#                           tf.keras.layers.Flatten(),
#                           tf.keras.layers.Dense(units=6,activation='relu'),
#                           tf.keras.layers.Dense(1,activation='sigmoid')])

#model=tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size,dim,input_length=max_len),
 #                           tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#                           tf.keras.layers.Dense(units=6,activation='relu'),
#                           tf.keras.layers.Dense(1,activation='sigmoid')])

#model=tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size,dim,input_length=max_len),
#                           tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
#                           tf.keras.layers.Dense(units=6,activation='relu'),
#                           tf.keras.layers.Dense(1,activation='sigmoid')])

model=tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size,dim,input_length=max_len),
                           tf.keras.layers.Conv1D(128,5,activation='relu'),
                           tf.keras.layers.GlobalAveragePooling1D(),
                           tf.keras.layers.Dense(units=6,activation='relu'),
                           tf.keras.layers.Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x=padded,y=train_labels,batch_size=32,epochs=10,validation_data=(test_pad,test_labels))

