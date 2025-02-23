
# Deriva de numbann.py y es darle la capacidad de hacer las derivadas de forma paralela usando numpy y con dos registros de datos
# el fin es chequear los pasos, si son correctos con la pesa y correcta forma de python.

from drnumba import *
import math
from autofore import AutoFore

drnumba=DrNumba("kernel.py")

class NumbaNN:
	def __init__(self,model, pruning=0):
		self.resolution=pruning
		self.concurrencia=20 # Máximas reedes neuronales que resuelven. Ideal que sea superior al batch.
		self.buffer=2

		self.dr=drnumba.dr(self)
		self.nn=AutoFore(pruning=pruning)

		weights = model.get_weights()

		maxWidth=model.layers[0].input.shape[1]+1
		maxHeight=0
		for layer in model.layers:
			maxHeight+=1
			if layer.input.shape[1]+1>maxWidth:
				maxWidth=layer.input.shape[1]+1

		self.h=maxHeight
		self.w=maxHeight
		self.weights = np.zeros((maxHeight,maxWidth,maxWidth),dtype=np.float32)
		self.posweights = np.zeros((maxHeight,maxWidth,maxWidth),dtype=np.int16)	
	
		self.nn_weights = []
		for i in range(maxHeight):
			nn1=[]
			for j in range(maxWidth):
				nn2=[]
				for k in range(maxWidth):
					self.posweights[i][j][k]=k
					nn2.append(self.nn.var().derivable())
				nn1.append(nn2)
			self.nn_weights.append(nn1)
			
		self.wids=np.zeros((maxHeight,maxWidth,maxWidth),dtype=np.int16)

		self.wid2tuple=[None]
		self.data=np.zeros((self.concurrencia,self.buffer,maxWidth),dtype=np.float32)
		
		self.nn_data=[]
		for i in range(maxWidth):
			self.nn_data.append(self.nn.var())

		self.gdata=np.zeros((self.concurrencia,self.buffer,maxWidth,self.resolution),dtype=np.float32)
		self.gid=np.zeros((self.concurrencia,self.buffer,maxWidth,self.resolution),dtype=np.int16)

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

		# Sort self.weights 
		for i in range(maxHeight):
			for j in range(maxWidth):
				# index sort, descending
				indices=np.argsort(np.abs(self.weights[i][j]))[::-1]
				self.weights[i][j]=self.weights[i][j][indices]
				self.posweights[i][j]=self.posweights[i][j][indices]

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
		concurrencia=0
		fromBuffer=0
		toBuffer=1

		epsilon=0.000001


		# copy xs to self.data
		data=self.data[concurrencia][fromBuffer]
		data2=self.data[concurrencia][toBuffer]
		gdata=self.gdata[concurrencia][fromBuffer]
		gdata2=self.gdata[concurrencia][toBuffer]
		gdata3=np.zeros(gdata2.shape,dtype=np.float32)
		gid=self.gid[concurrencia][fromBuffer]
		gid2=self.gid[concurrencia][toBuffer]
		gid3=np.zeros(gid2.shape,dtype=np.int16)

		# set to 0
		gdata[:]=0
		gdata2[:]=0
		gid[:]=0
		data[:]=0
		data2[:]=0
		data[:len(xs)]=xs

		# Resto a 0
		for i in range(len(self.nn_data)):
			if i<len(xs):
				self.nn_data[i]=self.nn.val(xs[i])
			else:
				self.nn_data[i]=self.nn.val(0)
		

		sigmoide=np.vectorize( lambda x: (1/(1+np.exp(-x))) if -11<x else 0 if x<11 else 1  )

		# vectorial product
		for i in range(self.weights.shape[0]): # capa
			# if i==1:
			# 	print("zancadilla")
			data[-1]=1
			# reset gdata ids too
			for k in range(self.weights.shape[2]):
				gid[-1][k]=0
				gdata[-1][k]=0

			self.nn_data[-1]=self.nn.val(1)

			nn_data2=[]


			for idx in range(self.weights.shape[1]): # neurona
				posweights=self.posweights[i][idx]
				# scalar product
				c=np.dot(self.weights[i][idx],data[posweights])
				print(self.weights[i][idx])
				print(data[posweights])
				data2[idx]=c

	

				# Copia los gradientes
				w2=self.weights[i][idx][0]
				pos=posweights[0]
	
				nn_c=self.nn.val(0)
				w=self.nn_weights[i][idx][pos]
				nn_c+=w*self.nn_data[pos]
				print(w.value,self.nn_data[pos].value)
	
				x=data[pos]
				id=self.wids[i][idx][pos]
				delta=0
				for m in range(self.resolution):
					gdata2[idx][m]=w2*gdata[pos][m-delta]
					gid2[idx][m]=gid[pos][m-delta]
					if id>0 and np.abs(x)>np.abs(gdata2[idx][m]):
						delta+=1
						gdata2[idx][m]=x
						gid2[idx][m]=id
						id=0

					tu=self.wid2tuple[gid2[idx][m]]
					if tu!=None:
						gw=self.nn_weights[tu[0]][tu[1]][tu[2]]
						g2=nn_c.get(gw)
						if abs(gdata2[idx][m]-g2)>epsilon:
							print("gdata2[idx][m]",gdata2[idx][m])
							print("g2",g2)
							print("Error en",gid2[idx][m])
						assert abs(gdata2[idx][m]-g2)<epsilon

				# Programa la verificación, es muy sencilla.

				errors={}
				for j in range(1,self.weights.shape[2]): # peso
					pos=posweights[j]
					x=data[pos]
					id=self.wids[i][idx][pos]
					delta=0

					w=self.nn_weights[i][idx][pos]
					nn_c+=w*self.nn_data[pos]
					print(w.value,self.nn_data[pos].value)
	

					w2=self.weights[i][idx][j]
					m1=0
					m2=0
					m3=0
					for m in range(self.resolution):
						new=False

						while not new:
							entro=False
							new=True
							_x=0
							_id=0

							if m1==0:
								_id=id
								_x=x
								m1_=1
								entro=True

							m2_=m2
							if abs(_x)<abs(gdata2[idx][m2]):
								m1_=m1

								_x=gdata2[idx][m2]
								_id=gid2[idx][m2]
								m2_+=1
								entro=True
							
							m3_=m3
							if abs(_x)<abs(w2*gdata[pos][m3]):
								m1_=m1
								m2_=m2

								_x=w2*gdata[pos][m3]
								_id=gid[pos][m3]
								m3_+=1
								entro=True

							if not entro:
								continue

							if m1==m1_: # El cambio proviene de m2 o m3, hay que hacer una burbuja sort inversa
								for m4 in range(m-1,0,-1):
									if _id==gid3[idx][m4]:
										gdata3[idx][m4]+=_x
										new=False
										#print(_id,"id bubuja")

										# inverse bubble sort, look up
										for m5 in range(m4-1,0,-1):
											if abs(gdata3[idx][m5])<abs(gdata3[idx][m5+1]):
												_x=gdata3[idx][m5]
												_id=gid3[idx][m5]
												gdata3[idx][m5]=gdata3[idx][m5+1]
												gid3[idx][m5]=gid3[idx][m5+1]
												gdata3[idx][m5+1]=_x
												gid3[idx][m5+1]=_id
											else:
												break
										break
							m1=m1_
							m2=m2_
							m3=m3_

						if entro:
							gdata3[idx][m]=_x
							gid3[idx][m]=_id

							#print(_id,_x)
						else:
							# set rest to 0 0
							for m4 in range(m,self.resolution):
								gdata3[idx][m4]=0
								gid3[idx][m4]=0

					#print("Check")
					for m in range(self.resolution):
						gdata2[idx][m]=gdata3[idx][m]
						gid2[idx][m]=gid3[idx][m]
						
						tu=self.wid2tuple[gid2[idx][m]]
						if tu!=None:
							gw=self.nn_weights[tu[0]][tu[1]][tu[2]]
							g2=nn_c.get(gw)
							# if abs(gdata2[idx][m]-g2)>0.001:
							# 	print("Error en",gid2[idx][m])
							# 	assert abs(gdata2[idx][m]-g2)<0.001
							errors[gid2[idx][m]]=abs(gdata2[idx][m]-g2)>epsilon
			
					for m,e in errors.items():
						if e:
							print("m1",m1,"m2",m2,"m3",m3)
							print("gid",gid[idx][m2],"gid2",gid2[idx][m3]);
							print("Error en",m)
						
						
				nn_c.pruning()
				nn_data2.append(nn_c)

				if (abs(data2[idx]-nn_c.value)>epsilon):
					print("data2[idx]",data2[idx])
					print("nn_c.value",nn_c.value)
					print("Error en data2")

				assert abs(data2[idx]-nn_c.value)<epsilon



			if self.activation[i]==1: # sigmoid
				data2[:]=sigmoide(data2)
				for idx in range(self.weights.shape[1]):
					nn_data2[idx]=nn_data2[idx].sigmoid()

					assert(abs(data2[idx]-nn_data2[idx].value)<0.001)

					link=(4 * np.cosh(data2[idx] / 2)**2)
					# Calcular el gradiente
					for j in range(len(gdata2[idx])):
						gdata2[idx][j]=gdata2[idx][j]/link
						# como hacer el assert?
						# id -> localiza gradiente
						tupla=self.wid2tuple[gid2[idx][j]]
						if not tupla is None:
							w=self.nn_weights[tupla[0]][tupla[1]][tupla[2]]
							assert abs(gdata2[idx][j]-nn_data2[idx].get(w))<0.001
					# verificarlo
					
					# link=(4 * np.cosh(v.value / 2)**2)
					# v.forward[name]+=value / link
					# for k,g in enumerate(gdata2[i]):
					# 	gdata2[i][k]=g/link
						
					# 	# como hacer el assert?
					# 	# id -> localiza gradiente
					# 	tupla=self.wid2tuple[gid2[i][k]]
					# 	w=self.nn_weights[tupla[0]][tupla[1]][tupla[2]]
					# 	print(gdata2[i][k],nn_data2[idx].get(w))
					# 	assert abs(gdata2[i][k]-nn_data2[idx].get(w))<0.001


			
			# pon data, data2, gdata, gdata2, gid, gid2 para ello crea variable.
			fromBuffer,toBuffer=toBuffer,fromBuffer
			data=self.data[concurrencia][fromBuffer]
			data2=self.data[concurrencia][toBuffer]
			gdata=self.gdata[concurrencia][fromBuffer]
			gdata2=self.gdata[concurrencia][toBuffer]
			gid=self.gid[concurrencia][fromBuffer]
			gid2=self.gid[concurrencia][toBuffer]

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
				yield 
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
			c=np.float32(idx==self.weights.shape[1]-1)
			for j in range(self.weights.shape[2]):
				c+=self.weights[i][idx][j]*self.data[j]

			if self.activation[i]==1: # sigmoid
				c=1/(1+math.exp(-c))
			
			cuda.syncthreads()
			self.data[j]=c
			cuda.syncthreads()

