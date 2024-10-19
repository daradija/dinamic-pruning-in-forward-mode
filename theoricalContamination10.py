
# Copiada del proyecto Kinetics
# Deriva de la versión 6, JL me pide que escale el entorno sin complicar la tecnología
# el sistema converge

import random
from neuronalprogrammig4 import NeuralNetwork

random.seed(0)

class Cell:
	def __init__(self,g,x,y):
		self.g=g
		self.x=x
		self.y=y

		self.car=None
		self.sensor=None

		self.contamination=[]
		self.contamination2=[]
		for nn in self.g.p.nn:
			self.contamination.append(nn.var())
			self.contamination2.append(nn.var())
		
class Grid:
	def __init__(self,p,tamx,tamy):
		self.p=p
		self.tamx=tamx
		self.tamy=tamy
		self.grid=[]
		for i in range(tamx):
			self.grid.append([])
			for j in range(tamy):
				self.grid[i].append(Cell(self,i,j))
		self.cars=[]	
		self.sensors=[]

	def putCar(self):
		x=random.randint(0,self.tamx-1)
		y=random.randint(0,self.tamy-1)
		cell=self.grid[x][y]
		car=Car(p)
		car.move(cell)
		self.cars.append(car)

	def putSensor(self):
		while True:
			x=random.randint(0,self.tamx-1)
			y=random.randint(0,self.tamy-1)
			sensor=Sensor(p)
			if self.grid[x][y].sensor==None:
				self.grid[x][y].sensor=sensor
				self.sensors.append(sensor)
				break

	def run(self):
		# Move cars
		for car in self.cars:
			car.x=random.randint(0,self.tamx-1)
			car.y=random.randint(0,self.tamy-1)
			cell=self.grid[car.x][car.y]
			car.move(cell)
		
		for i,nn in enumerate(self.p.nn):
			for car in self.cars:

				cell=car.cell
				# Aquí tengo problemas, porque el coche no tiene un lugar fijo, se mueve,
				# pruebo a irme a foward
				cell.contamination[i]+=nn.generator

		# loop all cells
		for k,nn in enumerate(self.p.nn):
			for row in self.grid:
				for cell in row:
					if nn.propagationx.value>0:
						cell.contamination2[k]-=cell.contamination[k]*nn.propagationx
						if cell.x<self.tamx-1:
							self.grid[cell.x+1][cell.y].contamination2[k]+=cell.contamination[k]*nn.propagationx
					else:
						cell.contamination2[k]+=cell.contamination[k]*nn.propagationx
						if cell.x>0:
							self.grid[cell.x-1][cell.y].contamination2[k]-=cell.contamination[k]*nn.propagationx

					if nn.propagationy.value>0:
						cell.contamination2[k]-=cell.contamination[k]*nn.propagationy
						if cell.y<self.tamy-1:
							self.grid[cell.x][cell.y+1].contamination2[k]+=cell.contamination[k]*nn.propagationy
					else:
						cell.contamination2[k]+=cell.contamination[k]*nn.propagationy
						if cell.y>0:
							self.grid[cell.x][cell.y-1].contamination2[k]-=cell.contamination[k]*nn.propagationy
					
					# cell.contamination2[k]+=cell.contamination[k]-cell.contamination[k]*(nn.propagation)
					# for (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]:
					# 	if 0<=cell.x+i<self.tamx and 0<=cell.y+j<self.tamy:
					# 		self.grid[cell.x+i][cell.y+j].contamination2[k]+=cell.contamination[k]*nn.propagation
	

		for inn,nn in enumerate(self.p.nn): #aqui está el problema, cambia la celda
			for row in self.grid:
				for cell in row:
					cell.contamination[inn]=cell.contamination2[inn]
					cell.contamination2[inn]=nn.val(cell.contamination[inn].value)

		p=self.p

		generador0=p.nn[0].generator
		propagationx0=p.nn[0].propagationx
		propagationy0=p.nn[0].propagationy

		for inn in range(1,len(p.nn)):
			nn=self.p.nn[inn]
			generador=nn.generator
			generador.delta=0
			propagationx=nn.propagationx
			propagationx.delta=0
			propagationy=nn.propagationy
			propagationy.delta=0

			for row in self.grid:
				for cell in row:
					if cell.sensor!=None:
						error=cell.contamination[0].value-cell.contamination[inn].value
						#print(cell.x,cell.y,cell.contamination[0].value,cell.contamination[inn].value,error)
						# if cell.contamination[0].value-cell.contamination[1].value<0:
						# 	error=-0.1
						for (key,grad) in enumerate(cell.contamination[inn].forward):
							if grad!=0:
								nn.id2var[key].delta+=error*grad								

					cell.contamination[i].foward={}

			epsilon=0.01

			# medir distancia
			# seleccionar candidatos a morir
			# seleccionar candidatos a reproducirse
			print(inn)
			for k,v,v0 in (("generator",generador,generador0),("propagationx",propagationx,propagationx0),("propagationy",propagationy,propagationy0)):
				print(" ",k,"correccion",v.delta)
				print(" ","es",v0.value)

				if -1<v.value+v.delta*epsilon and v.value+v.delta*epsilon<1:
					print(" ",k,v.value,"->",v.value+v.delta*epsilon)
					v.value+=v.delta*epsilon
			print()			

	

class Param:
	def __init__(self,tamNN=500):
		self.tamNN=tamNN
		self.nn=[]
		for i in range(self.tamNN):
			self.nn.append(NeuralNetwork())
		for i,nn in enumerate(self.nn):
			nn.generator=nn.val(random.random())
			nn.propagationx=nn.val(random.random()/2-0.5)
			nn.propagationy=nn.val(random.random()/2-0.5)
			if 0<i:
				nn.generator.derivable()
				nn.propagationx.derivable()
				nn.propagationy.derivable()
			

class Car:
	def __init__(self,p):
		self.p=p
		self.cell=None
	
	def move(self,cell):
		if self.cell!=None:
			self.cell.car=None
		self.cell=cell
		cell.car=self

class Sensor:
	def __init__(self,p):
		self.p=p

if __name__ == '__main__':
	p=Param(100)
	grid=Grid(p,4*10,4*10)
	for i in range(10):
		grid.putCar()
	for i in range(4*10):
		grid.putSensor()
	for i in range(100):
		grid.run()
		# tirar dos datos y mejor matar a peor
		for i in range(p.tamNN//4):
			dado1=random.randint(1,p.tamNN-1)
			dado2=random.randint(1,p.tamNN-1)
			dado3=random.random()
			if dado3<0.333333:
				h1=p.nn[dado1].generator.delta**2
				h2=p.nn[dado2].generator.delta**2	
				if h1<h2:
					p.nn[dado2].generator.value=p.nn[dado1].generator.value
				else:
					p.nn[dado1].generator.value=p.nn[dado2].generator.value
			elif dado3<0.666666:
				h1=p.nn[dado1].propagationx.delta**2	
				h2=p.nn[dado2].propagationx.delta**2	
				if h1<h2:
					p.nn[dado2].propagationx.value=p.nn[dado1].propagationx.value
				else:
					p.nn[dado1].propagationx.value=p.nn[dado2].propagationx.value
				
			else:
				h1=p.nn[dado1].propagationy.delta**2	
				h2=p.nn[dado2].propagationy.delta**2	
				if h1<h2:
					p.nn[dado2].propagationy.value=p.nn[dado1].propagationy.value
				else:
					p.nn[dado1].propagationy.value=p.nn[dado2].propagationy.value
				
			# h1=p.nn[dado1].generator.delta**2+p.nn[dado1].propagation.delta**2	
			# h2=p.nn[dado2].generator.delta**2+p.nn[dado2].propagation.delta**2	
			# if h1<h2:
			# 	p.nn[dado2].generator.value=p.nn[dado1].generator.value
			# 	p.nn[dado2].propagation.value=p.nn[dado1].propagation.value
			# else:
			# 	p.nn[dado1].generator.value=p.nn[dado2].generator.value
			# 	p.nn[dado1].propagation.value=p.nn[dado2].propagation.value
	
			
