# Taller de Redes Neuronales.

Automatic Differentiation: Dinamic Pruning in Forward Mode

Fecha: Jueves 24 de 10:30 a 12:30 
Lugar: Seminario de ATC 
Impartido por: David Ragel Díaz-Jara

# Hipótesis
- Es posible desarrollar redes neuronales mas eficientes mediante el paralelismo que proporciona Dinamic Pruning in Forward Mode.
- Es posible crear un hardware específico para entrenar. Nota: ahora esta de moda hardware especifico de inferencia.
- El modo forward permite una dinámica en las conexiones que no permite la backpropagation.

# Metodología práctica.
- Traeté tu portátil, a ser posible con nvidia.
- Todo el código en github.
- Propondré retos concretos a los que quieran colaborar.
- Alcance: Hasta donde permitan estas dos horas.
- Si gusta, repetiré periódicamente la experiencia.
- Crearé videos, para el que no pueda asistir o el que quiera repasar pueda repetir el taller.

# Temario 
- Es más fácil entender una red neuronal usando autodiferenciación y modo forward.
  - Sobrecarga de operadores en python.
  - Topología básica de una red neuronal.
    - Producto escalar.
    - Activación sigmoide.
  - Batchs, gradientes y minimizacion del error cuadrático medio.

- Prunning.
  - Latencia de tensorflow.
  - Entender el potencial paralelizable y la infrautilizacion de recursos.
  - En funciones sencillas el prunning es beneficioso.

- Desarrollo sin errores.
  - Simbolico.
  - Python -> numpy.
  - numpy -> numba.

- Harware (busco colaboradores)  
  - Potencial de una FPGA.
  - Tensor RT y latencia.
  - Soñemos con una alternativa mas eficiente que NVIDIA.
  - Potencial de artículos. Revisión de todas las arquitecturas de redes neuronales.
  - Construcción de una librería.


# Es más fácil entender una red neuronal usando autodiferenciación y modo forward.
- A continuación explicamos algo equivalente al algoritmo de backpropagation que requiere dos pasadas.
- El modo forward requiere una pasada. Pero la estructura de datos es mas compleja.

Es el eterno dilema entre memoria y tiempo de programación dinámica.

"Si quieres algo mas rápido gasta memoria".

¿Qué memoria debemos incorporar?

![Screenshot-2024-10-20_04_33_42](Screenshot-2024-10-20_04_33_42.png)
- Cada neurona tiene un vector de pesos **w**.
- Realiza un producto escalar entre la entrada **x** y los pesos **w**
- Hay una función de activación, que de momento podemos ignorar el objetivo es ver la estructura de memoria que necesitamos, no que contiene.
- EL OBJETIVO ES SABER LAS DERIVADAS DE LOS PESOS PRECEDENTES
![Screenshot-2024-10-20_05_08_03](Screenshot-2024-10-20_05_08_03.png)
- Para la explicación no me interesa cómo es la derivada/gradiente, sino cuantos cálculos he de guardar.
## ¿Cómo se utilizan lo gradientes?
* En un entrenamiento al final tenemos **y'** e **y**. 
* Si agrupamos varios resultados (batch) podemos calcular una función de pérdida (L)

$$
L = \frac{1}{N} \sum_{i=1}^{N} \left( y'_i - y_i \right)^2
$$

## Sobrecarga de operadores en python.
