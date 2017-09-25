#coding=utf-8
''' Import theano and numpy '''
from __future__ import print_function
from past.builtins import execfile
import numpy as np
execfile('00_readingInput.py')

''' set the size of mini-batch and number of epochs'''
batch_size = 16
epochs = 100

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad

def create_model_withDrp(ratio_dropout):
	
	print('Building current best model with Dropout = %g' % ratio_dropout)
	model_adam = Sequential()
	model_adam.add(Dense(128, input_dim=200))
	model_adam.add(Activation('relu'))
	model_adam.add(Dropout(ratio_dropout))
	model_adam.add(Dense(256))
	model_adam.add(Activation('relu'))
	model_adam.add(Dropout(ratio_dropout))
	model_adam.add(Dense(5))
	model_adam.add(Activation('softmax'))
	##
	model_adam.compile(loss= 'categorical_crossentropy',
              		optimizer='Adam',
              		metrics=['accuracy'])

	return model_adam

model_adam_drp10 = create_model_withDrp(0.1)
model_adam_drp20 = create_model_withDrp(0.2)
model_adam_drp40 = create_model_withDrp(0.4)

'''Fit models and use validation_split=0.1 '''
history_adam_drp10 = model_adam_drp10.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=2,
							shuffle=True,
                    		validation_split=0.1)

history_adam_drp20 = model_adam_drp20.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=2,
							shuffle=True,
                    		validation_split=0.1)

history_adam_drp40 = model_adam_drp40.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=2,
							shuffle=True,
                    		validation_split=0.1)

def get_result(history_model):
	train_loss = history_model.history.get('loss')
	train_acc = history_model.history.get('acc')
	valid_loss = history_model.history.get('val_loss')
	valid_acc = history_model.history.get('val_acc')
	return train_loss, train_acc, valid_loss, valid_acc

loss_adam_drp10, acc_adam_drp10, val_loss_adam_drp10, val_acc_adam_drp10 = get_result(history_adam_drp10)
loss_adam_drp20, acc_adam_drp20, val_loss_adam_drp20, val_acc_adam_drp20 = get_result(history_adam_drp20)
loss_adam_drp40, acc_adam_drp40, val_loss_adam_drp40, val_acc_adam_drp40 = get_result(history_adam_drp40)

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
skp = 10
plt.figure(0)
#plt.subplot(121)
plt.plot(range(len(loss_adam_drp10)), loss_adam_drp10,label='Training_drp10')
plt.plot(range(len(val_loss_adam_drp10)), val_loss_adam_drp10,label='Validation_drp10')

plt.plot(range(len(loss_adam_drp20)), loss_adam_drp20,label='Training_drp20')
plt.plot(range(len(val_loss_adam_drp20)), val_loss_adam_drp20,label='Validation_drp20')

plt.plot(range(len(loss_adam_drp40)), loss_adam_drp40,label='Training_drp40')
plt.plot(range(len(val_loss_adam_drp40)), val_loss_adam_drp40,label='Validation_drp40')

plt.legend()
plt.title('Loss')
plt.savefig('08_dropout_loss.png',dpi=300,format='png')
plt.close()

plt.plot(range(len(acc_adam_drp10)), acc_adam_drp10,label='Training_drp10')
plt.plot(range(len(val_acc_adam_drp10)), val_acc_adam_drp10,label='Validation_drp10')

plt.plot(range(len(acc_adam_drp20)), acc_adam_drp20,label='Training_drp20')
plt.plot(range(len(val_acc_adam_drp20)), val_acc_adam_drp20,label='Validation_drp20')

plt.plot(range(len(acc_adam_drp40)), acc_adam_drp40,label='Training_drp40')
plt.plot(range(len(val_acc_adam_drp40)), val_acc_adam_drp40,label='Validation_drp40')

plt.legend(loc = 4)
plt.title('Accuracy')

plt.savefig('08_dropout_accuracy.png',dpi=300,format='png')
plt.close()

print('Result saved into 08_dropout_loss/acc.png')
