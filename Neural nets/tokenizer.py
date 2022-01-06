from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences=['I love my dog','I love , my cat','You love my dog']

tokenizer=Tokenizer(100,oov_token='<oov>')
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
print(word_index)
sentence_new=['my dog loves my mantee great']

print('-------------------------------------')

seq=tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(seq )
print(padded)
print('-------------------------------------')
seq=tokenizer.texts_to_sequences(sentence_new)
padded = pad_sequences(seq )
print(padded)

print('-------------------------------------')
seq=tokenizer.texts_to_sequences(sentence_new+sentences)
padded = pad_sequences(seq )
print(padded)
print('-------------------------------------')


seq=tokenizer.texts_to_sequences(sentence_new+sentences)
padded = pad_sequences(seq ,maxlen=5)
print(padded)
print('-------------------------------------')

seq=tokenizer.texts_to_sequences(sentence_new+sentences)
padded = pad_sequences(seq ,maxlen=5,padding='post')
print(padded)
print('-------------------------------------')
seq=tokenizer.texts_to_sequences(sentence_new+sentences)
padded = pad_sequences(seq ,maxlen=5,padding='post',truncating='post')
print(padded)
print('-------------------------------------')
seq=tokenizer.texts_to_sequences(sentence_new+sentences)
padded = pad_sequences(seq ,maxlen=10,padding='post',truncating='post')
print(padded)
print('-------------------------------------')