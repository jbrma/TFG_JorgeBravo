<h2> Optimización Heurística en el Juego de la Vida de Conway: Algoritmos Genéticos y Aprendizaje Automático para la Simulación de Sistemas Astrofísicos </h2>

<p align="center">
  <img src="./simulaciones/gif_10Kgen.gif" width="60%"/>
</p>

### Descripción

Este Trabajo de Fin de Grado presenta el diseño y desarrollo de un universo simulado mediante un autómata celular modificado, inspirado en el clásico Juego de la Vida de Conway. A partir de una cuadrícula de células que representan distintos tipos de materia —como estrellas, planetas y agujeros negros— se han implementado reglas astrofísicas que permiten observar la evolución dinámica del sistema a lo largo de miles de generaciones.

### Objetivos

- Objetivo Principal: Simular estructuras análogas a las observadas en sistemas astrofísicos mediante algoritmos genéticos para optimizar configuraciones iniciales y reglas de transición en variantes del Juego de la Vida

- Análisis de Patrones: Analizar los patrones obtenidos a través de técnicas de aprendizaje automático para clasificar comportamiento y estructura

- Paralelismos Astrofísicos: Identificar similitudes entre los patrones generados y fenómenos astrofísicos y cosmológicos reales

### Tecnologías Utilizadas
- Python 3.11.4
- NumPy 1.24.3: Estructuras de datos matriciales
- Pygame: Visualización interactiva del autómata celular
- Matplotlib 3.10.3: Análisis gráfico de resultados
- TensorFlow/Keras: Redes neuronales y aprendizaje automático
- Tkinter: Interfaz gráfica avanzada

### Metodología Científica
#### Algoritmos Genéticos

- Población: 10-30 individuos según el experimento
- Selección: Elitista, conservando el 50% superior
- Cruzamiento: Punto único respetando coherencia espacial
- Mutación: Tasa del 1% para mantener diversidad

#### Aprendizaje Automático
- Extracción de características: 35 parámetros cuantificando distribuciones globales y regionales
- Red neuronal: Arquitectura con capas densas (128→64→8 neuronas)
- Autoencoder: Compresión de imágenes 200×200×3 a vectores de 64 dimensiones
- Clustering: K-Means para identificar 8 tipos de patrones estructurales

### Analogías Astrofísicas
El sistema demuestra similitudes estructurales con fenómenos cósmicos reales:
- Red cósmica: Patrones filamentosos similares a la distribución de materia a gran escala
- Formación jerárquica: Evolución de energía → asteroides → planetas → estrellas → agujeros negros
- Autoorganización: Emergencia espontánea de estructuras complejas sin control externo


Jorge Bravo Mateos

Grado en Ingeniería de Computadores

Universidad Complutense de Madrid

Curso académico 2024-25

Director: Rafael del Vado Vírseda

La memoria completa del trabajo está disponible en el repositorio: [TFG_JorgeBravoMateos](https://github.com/jbrma/TFG_JorgeBravo/blob/main/memoria/TFG_JorgeBravoMateos.pdf)

Este proyecto se desarrolla con fines académicos en el marco del Trabajo de Fin de Grado de la Universidad Complutense de Madrid.


