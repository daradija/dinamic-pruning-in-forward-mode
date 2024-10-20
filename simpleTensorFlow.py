# print python version
import sys
print(sys.version)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from numbann import NumbaNN
import os
import keras
import time


#np.random.seed(42)  
#tf.random.set_seed(42)

ancho=4
X = np.random.rand(200, ancho)  
y = np.sum(X, axis=1, keepdims=True)  


def createModel(capas=1):
    model=Sequential()
    model.add(Dense(5, input_dim=ancho, activation='sigmoid'))
    for c in range(capas):
        model.add(Dense(5, activation='sigmoid'))
    #model.add(Dense(5, activation='sigmoid'))
    #model.add(Dense(5))
    # model.add(Dense(5))
    model.add(Dense(1))
    model.compile(optimizer=SGD(), loss='mean_squared_error')
    model.summary()
    print(model.get_weights())
    return model


# modelRef = createModel(0)
# nnRef=NumbaNN(modelRef)
# y=nnRef.predict(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=42)

nns=[]

# import tensorflow as tf
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# print(f"Tamaño del conjunto de entrenamiento: {x_train.shape}")
# print(f"Tamaño del conjunto de prueba: {x_test.shape}")



if os.path.exists('model.h5') and False:
    model = tf.keras.models.load_model('model.h5')
else:
    model = createModel()
    # weights = model.get_weights()	
    # print(weights)	

    epochs=50
    
    its=[]
    for prunning in [2, 4, 8, 16, 32, 64, 0]:
        nn=NumbaNN(model,prunning=prunning)
        nns.append(nn)
        its.append(nn.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=1,validation_data=(X_test,y_test))())
   
    start=time.time()
    cont=True
    while cont:
        try:
            for it in its:
                next(it)
        except StopIteration:
            cont=False
    print("Tiempo de entrenamiento1: ",time.time()-start)
    start=time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=1,validation_data=(X_test, y_test))
    print("Tiempo de entrenamiento2: ",time.time()-start)

    model.save('model.h5')

nn=NumbaNN(model)

loss = model.evaluate(X_test, y_test)
print(f'Pérdida keras: {loss}')

# lossb=np.mean(keras.losses.mean_squared_error(y_test, model.predict(X_test)).numpy())
# print(f'Pérdida kerasB: {lossb}')

lost2 = nn.evaluate(X_test, y_test)
print(f'Pérdida numba: {lost2}')

# lost2b=np.mean(keras.losses.mean_squared_error(y_test, nn.predict(X_test)).numpy())
# print(f'Pérdida numbaB: {lost2b}')

lost3 = nn2.evaluate(X_test, y_test)
print(f'Pérdida numba2: {lost3}')

# lost3b=np.mean(keras.losses.mean_squared_error(y_test, nn2.predict(X_test)).numpy())
# print(f'Pérdida numba2B: {lost3b}')

yp=nn2.predict(x)

predicciones = model.predict(X_test)
for i in range(len(yp)):
    print(f'Numba: {yp[i][0].value} \nKeras: {predicciones[i][0]} \nReal: {y_test[i][0]}\n')
