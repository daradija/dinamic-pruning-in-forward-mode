import datetime
import time
import numpy as np
from drnumba import *
import tensorflow as tf
from tensorflow import keras

def generate_time_series(companies,indexGlobal,indexDate,matrix,n_steps):
	serie=[]
	for i in range(1,len(indexDate)):
		serie2=[]
		for j in range(len(indexGlobal)):
			if matrix[i][j]!=None and matrix[i-1][j]!=None:
				serie2.append(np.log(matrix[i][j]["o"]/matrix[i-1][j]["o"]))
			else:
				serie2.append(0)
		yield np.array(serie2,dtype=np.float16)


def main():
	n_steps = 220
	train_size=20
	validate_size=0
	epochs=50
	prediccion=n_steps

	canales=5
	test_size=1
	
	series=generate_time_series(b.companies,b.indexGlobal,aux["indexDate"],aux["matrix"],n_steps+1)
	# La primera fecha tiene índice 1 el comienzo, y termina n_steps+prediccion días después

	model = keras.models.Sequential()
	model.add(keras.layers.InputLayer(input_shape=[n_steps, dimension]))
	for rate,size in ((1,5),(5,4),(20,3)):
		model.add(keras.layers.Conv1D(filters=canales, kernel_size=size,padding="causal", activation=None, dilation_rate=rate))
	model.add(keras.layers.Conv1D(filters=dimension, kernel_size=1))

	def last_time_step_mse(Y_true, Y_pred):
		return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])
	
	optimizer = keras.optimizers.Adam(lr=0.01)
	model.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])

	#model.compile(loss="mape", optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))
	
	#model.compile(loss=custom_ar, optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))
	#model.compile(loss=custom_arSinSort,  optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))
	
	model.summary()
	
	numerador=0
	denominador=0

	for since in range(0,100):
		X_train, Y_train = series[since:since+train_size, :n_steps], series[since:since+train_size, -prediccion:]
		X_valid, Y_valid = series[since+train_size:since+train_size+validate_size, :n_steps], series[since+train_size:since+train_size+validate_size, -prediccion:]
		X_test, Y_test = series[since+train_size+validate_size:since+train_size+validate_size+test_size, :n_steps], series[since+train_size+validate_size:since+train_size+validate_size+test_size, -prediccion:]

		start=time.time()
		  
		model.fit(X_train, Y_train, epochs=epochs,verbose=0 if validate_size==0 else 2,validation_data=(X_valid, Y_valid))
		print('Time:',time.time()-start)

		p=model.predict(X_test)[:,-1:].flatten()
		Y_test=Y_test[:,-1,:].flatten()
		loss = np.mean(keras.losses.mean_squared_error(Y_test, p))
		print("loss",loss)
		# tf.keras.backend.clear_session()
		
		ar=ARCoeficient()
		ar.calculateByNumpy(Y_test,p,None) #np.array(b.indexGlobal))
		#print(ar.mean)
		numerador+=ar.mean

		# custom_mse2=custom_mse(Y_test,p)
		# comp=1-custom_mse2.numpy()
		# assert comp==ar.mean

		denominador+=1
		print("ar",numerador/denominador)
