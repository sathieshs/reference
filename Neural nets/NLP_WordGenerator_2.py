import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam
import numpy as np

tokenizer = Tokenizer()

data=open('poetry.txt').read()
corpus=data.lower().split('\n')


tokenizer.fit_on_texts(corpus)
word_index=tokenizer.word_index
vocab_size=len(word_index)+1

sequence=[]

for line in corpus:
    seq = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(seq)):
        sequence.append(seq[:i+1])


max_len=max(len(x) for x in sequence )
input_sequence=np.array(pad_sequences(sequence,maxlen=max_len))

xs ,labels=input_sequence[:,:-1],input_sequence[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)

#model=tf.keras.Sequential()
#model.add(Embedding(vocab_size,100,input_length=max_len-1))
#model.add(Bidirectional(LSTM(150)))
#model.add(Dense(vocab_size,activation='softmax'))
#model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer=tf.keras.optimizers.Adam(lr=0.01))
#history=model.fit(xs,ys,epochs=100,verbose=1)

import matplotlib.pyplot as plt



def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

#plot_graphs(history, 'acc')


#model.save('nlp_model')
model=tf.keras.models.load_model('nlp_model')
seed='Battered is hadnt'
num_pred=20
for i in range(num_pred):
    token_list=tokenizer.texts_to_sequences([seed])[0]
    print('ddd '+seed )
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed += " " + output_word
print(seed)

