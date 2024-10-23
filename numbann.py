
# Fragmento importado desde racecar3.py de p24/ddl-sp
# Se hizo una derivación porque esta versión era estable.

from drnumba import *
import math
from autofore import AutoFore

drnumba=DrNumba("kernel.py")

class NumbaNN:
	def __init__(self,model, pruning=0):
		self.resolution=20

		self.dr=drnumba.dr(self)
		self.nn=AutoFore(pruning=pruning)

		weights = model.get_weights()

		maxWidth=model.layers[0].input.shape[1]+1
		maxHeight=0
		for layer in model.layers:
			maxHeight+=1
			if layer.input.shape[1]>maxWidth:
				maxWidth=layer.input.shape[1]+1

		self.h=maxHeight
		self.w=maxHeight
		self.weights = np.zeros((maxHeight,maxWidth,maxWidth),dtype=np.float16)
	
		self.nn_weights = []
		for i in range(maxHeight):
			nn1=[]
			for j in range(maxWidth):
				nn2=[]
				for k in range(maxWidth):
					nn2.append(self.nn.var().derivable())
				nn1.append(nn2)
			self.nn_weights.append(nn1)
			
		self.wids=np.zeros((maxHeight,maxWidth,maxWidth),dtype=np.int16)
		self.wid2tuple=[]

		self.wid2tuple=[None]
	
		self.data=np.zeros(maxWidth,dtype=np.float16)
		self.nn_data=[]
		for i in range(maxWidth):
			self.nn_data.append(self.nn.var())

		self.gdata=np.zeros((maxWidth,self.resolution),dtype=np.float16)
		self.gid=np.zeros((maxWidth,self.resolution),dtype=np.int16)

	 	#par entrada salida
		#impar salida
		for i,h in enumerate(weights):
			for j,w in enumerate(h):
				if i%2==0:
					for k,v in enumerate(w):
						self.weights[i//2][k][j]=v
						self.wids[i//2][k][j]=len(self.wid2tuple)
						self.wid2tuple.append((i//2,k,j))

						self.nn_weights[i//2][k][j].value=v
				else:
					self.weights[i//2][j][maxWidth-1]=w
					self.wids[i//2][j][maxWidth-1]=len(self.wid2tuple)
					self.wid2tuple.append((i//2,j,maxWidth-1))

					self.nn_weights[i//2][j][maxWidth-1].value=w

		#self.dr.data("h","w","w","weights")
		#self.dr.data("w","data",param=["predict"])

		# identifica si hay que hacer activación
		self.activation=np.zeros(maxHeight,dtype=np.int8)

		code=["linear","sigmoid","relu","softmax","tanh"]

		for i,layer in enumerate(model.layers):
			# print(f"Capa: {layer.name}")
			# print(f"Tipo de capa: {layer.__class__.__name__}")
			# print(f"Entradas: {layer.input.shape}")
			self.lastWidh=layer.output.shape[1]
			# print(f"Función de activación: {layer.activation if hasattr(layer, 'activation') else 'No tiene'}")
			# # get function name:
			# print(layer.activation.__name__)
			self.activation[i]=code.index(layer.activation.__name__)
			# print("\n")
		#self.dr.data("h","activation")
		#self.dr.function("predict2","w")

	def predict(self, xs):
		r=[]
		for x in xs:
			r.append([v.value for v in self.predict2(x)])
		return r


	def predict2(self, xs): # version numpy
		# copy xs to self.data
		data=np.zeros(self.data.shape,dtype=np.float16)
		data2=np.zeros(self.data.shape,dtype=np.float16)
		data[:len(xs)]=xs


		# Resto a 0
		for i in range(len(self.nn_data)):
			if i<len(xs):
				self.nn_data[i]=self.nn.val(xs[i])
			else:
				self.nn_data[i]=self.nn.val(0)
		

		sigmoide=np.vectorize( lambda x: (1/(1+np.exp(-x))) if -11<x else 0 if x<11 else 1  )

		# vectorial product
		for i in range(self.weights.shape[0]):
			data[-1]=1

			self.nn_data[-1]=self.nn.val(1)

			nn_data2=[]

			for idx in range(self.weights.shape[1]):
				# scalar product
				c=np.dot(self.weights[i][idx],data)

				nn_c=self.nn.val(0)
				for j in range(self.weights.shape[2]):
					nn_c+=self.nn_weights[i][idx][j]*self.nn_data[j]

				# for i,grad in enumerate(data):
				# 	# grad=grad*1
				# 	id=self.gid[i][idx][i]

				
				data2[idx]=c
				nn_c.pruning()
				nn_data2.append(nn_c)

			if self.activation[i]==1: # sigmoid
				data2=sigmoide(data2)
				for idx in range(self.weights.shape[1]):
					nn_data2[idx]=nn_data2[idx].sigmoid()
			data[:]=data2

			# Tiene tamaño completo? si
			# son iguales? si
			for i,d2 in enumerate(nn_data2):
				self.nn_data[i]=d2	

		return self.nn_data[:self.lastWidh]
		return data[:self.lastWidh]
	
	def addDelta(self, error):
		for i in range(self.weights.shape[0]):
			for j in range(self.weights.shape[1]):
				for k in range(self.weights.shape[2]):
					w=self.nn_weights[i][j][k]
					w.delta+=error.get(w)

	def applyDelta(self, epsilon):
		totalvalue=0
		totaldelta=0
		topDelta=[0]*10
		num=0
		for i in range(self.weights.shape[0]):
			for j in range(self.weights.shape[1]):
				for k in range(self.weights.shape[2]):
					w=self.nn_weights[i][j][k]
					totalvalue+=abs(w.value)
					totaldelta+=abs(w.delta)
					adelta=abs(w.delta)
					if 0<adelta:
						num+=1
					for m,td in enumerate(topDelta):
						if td<adelta:
							aux=topDelta[m]
							topDelta[m]=adelta
							adelta=aux

		
		minimoTop=topDelta[-1]
		learnTasa=epsilon/topDelta[0]
		#print(learnTasa)
		for i in range(self.weights.shape[0]):
			for j in range(self.weights.shape[1]):
				for k in range(self.weights.shape[2]):
					w=self.nn_weights[i][j][k]
					#if abs(w.delta)>minimoTop:
					w.value-=w.delta*learnTasa
					w.delta=0
					# w.value-=w.delta*learnTasa
					# w.delta=0


	def fit(self, X_train, y_train, epochs=100, batch_size=10, verbose=1,shuffle=False,validation_data=None):
		def f():
			nonlocal X_train, y_train
			for epoch in range(epochs):
				if shuffle:
					indices=np.random.permutation(len(X_train))
					X_train=X_train[indices]
					y_train=y_train[indices]
				loss2=0
				divisor0=0
				for i in range(0,len(X_train),batch_size):
					X_batch=X_train[i:i+batch_size]
					y_batch=y_train[i:i+batch_size]

					loss=self.nn.val(0)
					divisor=0
					for X,y in zip(X_batch,y_batch):
						yp=self.predict2(X)
						for y2,yp2 in zip(y,yp):
							error=yp2-y2
							loss+=error*error
							divisor+=1

					loss=loss/divisor
					self.addDelta(loss)
					self.applyDelta(0.01)
					loss2+=loss.value
					divisor0+=1
				if verbose==1:
					loss2=loss2/divisor0
					if validation_data is not None:
						loss3=self.evaluate(validation_data[0],validation_data[1])
						print(f"Pruning: {self.nn.pruning}, Epoch: {epoch+1}, Loss: {loss2}, Validation Loss: {loss3}")
					else:
						print(f"Pruning: {self.nn.pruning}, Epoch: {epoch+1}, Loss: {loss2}")
				if validation_data is not None:
					yield (epoch+1,loss2,loss3)
				else:
					yield (epoch+1,loss2)
		return f

	def evaluate(self, X_test, y_test):
		loss=0
		for X,y in zip(X_test,y_test):
			yp=self.predict2(X)
			for y2,yp2 in zip(y,yp):
				error=y2-yp2.value
				loss+=error*error
		return loss/len(X_test)

	def predictVNumba(self):
		idx=cuda.grid(1)
		if idx>=self.weights.shape[1]:
			return
		
		for i in range(self.weights.shape[0]):
			c=np.float16(idx==self.weights.shape[1]-1)
			for j in range(self.weights.shape[2]):
				c+=self.weights[i][idx][j]*self.data[j]

			if self.activation[i]==1: # sigmoid
				c=1/(1+math.exp(-c))
			
			cuda.syncthreads()
			self.data[j]=c
			cuda.syncthreads()

