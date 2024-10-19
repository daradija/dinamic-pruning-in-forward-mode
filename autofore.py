# Es una copia de carpole
# Que a su vez es neroprogramming4.py renombrada como autofore
# Pero con avances.

import random 
import math
import numpy as np

class AutoFore:
	def __init__(self,gaf=None,prunning=0):
		self.var2id={}
		self.id2var={}
		self.nominative=[]
		self.gaf=gaf
		self.prunning=prunning

	def vector(self,x,y):
		if isinstance(x,Variable):
			xx=x
		else:
			xx=self.val(x)
		if isinstance(y,Variable):
			yy=y
		else:
			yy=self.val(y)
		return (xx,yy)

	def nominativeComparison(self,ref,epsilon):
		
		for n,p in enumerate(self.nominative):
			error=-p+ref.nominative[n].value
			if n==1:
				print("error",error.value)
			for key,grad in enumerate(p.forward):
				if grad!=0:
					self.id2var[key].delta+=error.value*grad*epsilon
				p.forward[key]=0
			p.value=ref.nominative[n].value
		self.applyDelta()

	def applyDelta(self):
		for p in self.id2var.values():
			p.value+=p.delta
			p.delta=0

	def get(self,name):
		return self.params[name]

	def derivable(self):
		for k in self.keys:
			p=self.params[k]
			p.name=k
			p.derivable()

	def val(self,value):
		v=Variable(self)
		v.value=value
		v.forward=[]
		return v
	
	def var(self):
		v=Variable(self)
		v.forward=[]
		return v
	
	def param(self,valueFrom, valueTo):
		v=self.val(random.uniform(valueFrom,valueTo))
		v.valueFrom=valueFrom
		v.valueTo=valueTo
		return v.derivable()
	
	def control(self,valueFrom, valueTo):
		v=self.val(random.uniform(valueFrom,valueTo))
		v.valueFrom=valueFrom
		v.valueTo=valueTo
		return v.nomination()

	def midVar(self):
		v=Variable(self)
		v.forward=[0]*len(self.var2id)
		return v
	
class Variable:
	def __init__(self, nn):
		self.nn=nn
		self.value = 0

	def prunning(self):
		if self.nn.prunning==0:
			return 
		topDelta=[0]*self.nn.prunning
		for delta in self.forward:
			adelta=abs(delta)
			for m,td in enumerate(topDelta):
				if td<adelta:
					aux=topDelta[m]
					topDelta[m]=adelta
					adelta=aux
		for i,delta in enumerate(self.forward):
			adelta=abs(delta)
			if adelta<topDelta[-1]:
				self.forward[i]=0

	def clone(self):
		v=Variable(self.nn)
		v.value=self.value
		v.forward=list(self.forward)
		return v

	def set(self,v):
		if self.valueFrom<=v.value and v.value<=self.valueTo:
			self.value=v.value
			self.forward=v.forward
		return self

	def nomination(self):
		if self.nn.gaf.population[0]!=self.nn:
			self.value=self.nn.gaf.population[0].nominative[len(self.nn.nominative)].value
		self.nn.nominative.append(self)
		return self

	def get(self,v):
		id=self.nn.var2id[v]
		return self.forward[id]

	def derivable(self):
		self.id=len(self.nn.var2id)
		self.nn.var2id[self]=self.id
		self.nn.id2var[self.id]=self
		self.forward=[0]*len(self.nn.var2id)
		self.forward[self.id]=1
		self.delta=0
		return self
		
	def __add__(self, other):
		v=self.nn.midVar()
		if not isinstance(other, Variable):
			v.value=self.value+other
		else:
			v.value=self.value+other.value

		for child in (self, other):
			if isinstance(child, Variable):
				for name,value in enumerate(child.forward):
					v.forward[name]+=value
		return v

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		v=self.nn.midVar()
		if not isinstance(other, Variable):
			v.value=self.value*other
		else:
			v.value=self.value*other.value

		children=(self, other)
		for i,child in enumerate(children):
			if isinstance(child, Variable):
				for name,value in enumerate(child.forward):
					if isinstance(children[1-i], Variable):
						link=children[1-i].value
					else:
						link=children[1-i]
					v.forward[name]+=link*value 
		return v
	
	def __pow__(self, exponent):
		# Crear una nueva variable para el resultado
		v = self.nn.midVar()
		
		# Calcular el valor de la potencia
		v.value = self.value ** exponent
		
		# Calcular la pasada forward para los gradientes
		for name, value in enumerate(self.forward):
			v.forward[name] = exponent * (self.value ** (exponent - 1)) * value
		
		return v

	def __neg__(self):
		v=self.nn.midVar()
		v.value=-self.value
		child=self
		for name,value in enumerate(child.forward):
			link=-1
			v.forward[name]+=link*value 
		return v

	def sin(self):
		v=self.nn.midVar()
		v.value=math.sin(self.value)
		child=self
		for name,value in enumerate(child.forward):
			link=math.cos(child.value)
			v.forward[name]+=link*value 
		return v

	def cos(self):
		v=self.nn.midVar()
		v.value=math.cos(self.value)
		child=self
		for name,value in enumerate(child.forward):
			link=-math.sin(child.value)
			v.forward[name]+=link*value 
		return v

	def sigmoid(self):
		v=self.nn.midVar()
		v.value=1 / (1 + np.exp(-self.value))
		for name,value in enumerate(self.forward):
			
			link=(4 * np.cosh(v.value / 2)**2)
			v.forward[name]+=value / link
		return v

	def __sub__(self, other):
		v=self.nn.midVar()
		if isinstance(other, Variable):
			v.value=self.value-other.value
		else:
			v.value=self.value-other
		for i,child in enumerate((self, other)):
			if isinstance(child, Variable):
				for name,value in enumerate(child.forward):
				
					link=1-2*i
					v.forward[name]+=link*value 
		return v

	def __truediv__(self, other):
		v=self.nn.midVar()
		if isinstance(other, Variable):
			v.value=self.value/other.value
		else:
			v.value=self.value/other
		for i,child in enumerate((self, other)):
			if isinstance(child, Variable):
				for name,value in enumerate(child.forward):
					if i==0:
						if isinstance(other, Variable):
							link=1/other.value
						else:
							link=1/other
					else:
						link=-v.value/(other.value**2)
					v.forward[name]+=link*value 
		return v


class GeneticAutoFore:
	def __init__(self,populationSize):
		self.population=[AutoFore(gaf=self) for i in range(populationSize)]

if __name__ == '__main__':
	nn=AutoFore()

	# A*B=C
	# z*x * x*y = z*y
	x=2
	y=4
	z=4
	
	def f0(*args):
		return sum(args)
	def f1(a,b):
		return 1*a+2*b
	def f3(a,b):
		return 10*a+2*b
	def f4(a,b):
		return 2*a+5*b
	
	fs=[f1,f0,f3,f4]
	assert len(fs)==y
	
	A=[[random.random() for j in range(x)] for i in range(z)]

	Ct=[[fs[yy](*A[zz]) for zz in range(z)] for yy in range(y)]
	# C= transpose(Ct)
	C=[[Ct[j][i] for j in range(y)] for i in range(z)]

	B=[[nn.val(random.random()).derivable() for j in range(y)] for i in range(x)]
	
	totalPendientes=y
	completado=[False]*y

	# A   * B   = C
	# z*x * x*y = z*y
	while True:
		for yy in range(y):
			if completado[yy]:
				continue
			for b1 in B:
				b1[yy].delta=0
			errorTotal=0

			for zz in range(z):
				c=C[zz][yy]
				a=A[zz]
				cp=0
				for xx in range(x):
					cp+=B[xx][yy]*a[xx]

				#print("c",c.value)
				error=c-cp.value
				errorTotal+=abs(error)

				for b1 in B:
					b=b1[yy]
					b.delta+=cp.get(b)*error

			
			epsilon=0.01
			for b1 in B:
				b=b1[yy]
				b.value+=b.delta*epsilon
			# 	print(b.value,end=" ")
			# print()

			if errorTotal<0.001:
				completado[yy]=True
				totalPendientes-=1
				break
		if totalPendientes==0:
			break
		
	