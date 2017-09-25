#coding=utf-8
''' Import theano and numpy '''
from __future__ import print_function
from past.builtins import execfile
import numpy as np
execfile('00_readingInput.py')

''' Import l1,l2 (regularizer) '''
from keras.regularizers import l1,l2, l1_l2

''' set the size of mini-batch and number of epochs'''
batch_size = 16
epochs = 50

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

def build_model_with_regulizers(l1n = 0., l2n = 0.):
  print('Building a model with regularizer L1: %g, L2: %g' % (l1n, l2n))
  model = Sequential()
  model.add(Dense(128, input_dim=200, kernel_regularizer=l1_l2(l1 = l1n, l2 = l2n) ))
  model.add(Activation('relu'))
  model.add(Dense(256, kernel_regularizer=l1_l2(l1 = l1n, l2 = l2n) ))
  model.add(Activation('relu'))
  model.add(Dense(5, kernel_regularizer=l1_l2(l1 = l1n, l2 = l2n)))
  model.add(Activation('softmax'))
  return model

''' Setting optimizer as Adam '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
model_l2 = build_model_with_regulizers(l1n = 0, l2n = 0.005)
model_l2.compile(loss= 'categorical_crossentropy',
              	optimizer='Adam',
              	metrics=['accuracy'])

'''Fit models and use validation_split=0.1 '''
history_l2 = model_l2.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)
loss_l2 = history_l2.history.get('loss')
acc_l2 = history_l2.history.get('acc')
val_loss_l2 = history_l2.history.get('val_loss')
val_acc_l2 = history_l2.history.get('val_acc')


model_l2a = build_model_with_regulizers(l1n = 0, l2n = 0.05)
model_l2a.compile(loss= 'categorical_crossentropy',
              	optimizer='Adam',
              	metrics=['accuracy'])

'''Fit models and use validation_split=0.1 '''
history_l2a = model_l2a.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)
loss_l2a = history_l2a.history.get('loss')
acc_l2a = history_l2a.history.get('acc')
val_loss_l2a = history_l2a.history.get('val_loss')
val_acc_l2a = history_l2a.history.get('val_acc')

##
model_l1 = build_model_with_regulizers(l1n = 0.005, l2n = 0.)
model_l1.compile(loss= 'categorical_crossentropy',
              	optimizer='Adam',
              	metrics=['accuracy'])

'''Fit models and use validation_split=0.1 '''
history_l1 = model_l1.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)
loss_l1 = history_l1.history.get('loss')
acc_l1 = history_l1.history.get('acc')
val_loss_l1 = history_l1.history.get('val_loss')
val_acc_l1 = history_l1.history.get('val_acc')


# reference
print('model without regularizers')
model_adam = build_model_with_regulizers(l1n = 0., l2n = 0.)
''' Setting optimizer as Adam '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
model_adam.compile(loss= 'categorical_crossentropy',
              		optimizer='Adam',
              		metrics=['accuracy'])

'''Fit models and use validation_split=0.1 '''
history_adam = model_adam.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)
''' Access the performance on validation data '''
loss_adam = history_adam.history.get('loss')
acc_adam = history_adam.history.get('acc')
val_loss_adam = history_adam.history.get('val_loss')
val_acc_adam = history_adam.history.get('val_acc')




''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0)
plt.subplot(221)
plt.plot(range(len(loss_adam)), loss_adam,label='Training')
plt.plot(range(len(val_loss_adam)), val_loss_adam,label='Validation')
plt.title('Loss - Original')
plt.legend(loc='upper left')
plt.subplot(222)
plt.plot(range(len(loss_l2)), loss_l2,label='Training')
plt.plot(range(len(val_loss_l2)), val_loss_l2,label='Validation')
plt.title('Loss - With L2 Regularizer: 0.005')
plt.subplot(223)
plt.plot(range(len(loss_l2a)), loss_l2a,label='Training')
plt.plot(range(len(val_loss_l2a)), val_loss_l2a,label='Validation')
plt.title('Loss - With L2 Regularizer: 0.05')
plt.subplot(224)
plt.plot(range(len(loss_l1)), loss_l1,label='Training')
plt.plot(range(len(val_loss_l1)), val_loss_l1,label='Validation')
plt.title('Loss - With L1 Regularizer: 0.005')
plt.tight_layout()

plt.savefig('06_regularizer.png',dpi=300,format='png')
plt.close()
print('Result saved into 06_regularizer.png')

plt.figure(0)
plt.subplot(121)
plt.plot(range(len(acc_adam)), acc_adam, label = 'without regularizer')
plt.plot(range(len(acc_adam)), acc_l1, label = 'L1 regularizer: 0.005')
plt.plot(range(len(acc_adam)), acc_l2, label = 'L2 regularizer: 0.005')
plt.plot(range(len(acc_adam)), acc_l2a, label = 'L2 regularizer: 0.5')
plt.legend(fontsize = 8, loc = 'lower right')
plt.subplot(122)
plt.plot(range(len(val_acc_adam)), val_acc_adam, label = 'without regularizer')
plt.plot(range(len(val_acc_adam)), val_acc_l1, label = 'L1 regularizer: 0.005')
plt.plot(range(len(val_acc_adam)), val_acc_l2, label = 'L2 regularizer: 0.005')
plt.plot(range(len(val_acc_adam)), val_acc_l2a, label = 'L2 regularizer: 0.5')
plt.legend(fontsize = 8, loc = 'lower right')

plt.savefig('06_regularizer_accuracy.png',dpi=300,format='png')
plt.close()
print('Result saved into 06_regularizer_accuracy.png')
