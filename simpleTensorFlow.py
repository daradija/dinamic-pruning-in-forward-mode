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
import matplotlib.pyplot as plt


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
    # print(model.get_weights())
    return model

class Report:
    def __init__(self,xs,names):
        self.xs=xs
        self.names=names
        self.ys=[[] for _ in range(len(names))]
        self.voyy=0

    def add(self,y):
        self.ys[self.voyy].append(y)
        self.voyy+=1
        if self.voyy==len(self.names):
            self.voyy=0

    def print(self):
        # Crear la gráfica
        plt.figure(figsize=(8, 5))
        # Recorrer las series en ys y graficar cada una con su respectivo nombre
        for y_series, label in zip(self.ys, self.names):
            plt.plot(self.xs, y_series, label=label, marker='o')

        # Añadir títulos y etiquetas
        plt.title('Series Plot')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.legend()
        plt.grid(True)
        plt.xticks(self.xs)

        # Mostrar la gráfica
        plt.show()

#np.random.seed(42)  
#tf.random.set_seed(42)

ancho=4
X = np.random.rand(400, ancho) # 200, 50, 100, 400  
y = np.sum(X, axis=1, keepdims=True)  




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

    epochs=250

    registers=[2, 4, 8, 16, 32, 64, 0]
    #registers=[16,0]
    # loss=[0]*len(registers)
    # val_loss=[0]*len(registers)
    # trasversal_1=[]
    # trasversal_2=[]

    rhorizontal=None
    oldrhorizontal=None

    registers2=list(registers)
    registers2[-1]=128

    rtrasversal=Report(range(1,epochs+1),registers)
    
    its=[]
    for prunning in registers:
        nn=NumbaNN(model,prunning=prunning)
        nns.append(nn)
        its.append(nn.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=1,validation_data=(X_test,y_test),shuffle=True)())
    start=time.time()
    cont=True
    while cont:
        try:
            rhorizontal=Report(registers2,["Loss","Validation Loss"])
            for i,it in enumerate(its):
                epoch,loss_,val_loss_=next(it)
                rhorizontal.add(loss_)
                rhorizontal.add(val_loss_)
                rtrasversal.add(val_loss_)

                # loss[i]=loss_
                # val_loss[i]=val_loss_
                # if 16==registers[i]:
                #     trasversal_1.append(val_loss_)
                # if 0==registers[i]:
                #     trasversal_2.append(val_loss_)
            oldrhorizontal=rhorizontal
        except StopIteration:
            rhorizontal=oldrhorizontal
            cont=False
    print("Tiempo de entrenamiento1: ",time.time()-start)
    
    rtrasversal.print()
    rhorizontal.print()

    report(range(epochs),trasversal_1,trasversal_2)

    registers[-1]=128
    report(registers,loss,val_loss)
        
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

