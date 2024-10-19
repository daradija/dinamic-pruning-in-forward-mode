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
