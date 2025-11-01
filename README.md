<h2> Heuristic Optimization in Conway's Game of Life: Genetic Algorithms and Machine Learning for the Simulation of Astrophysical Systems </h2>

<p align="center">
  <img src="./simulaciones/gif_10Kgen.gif" width="60%"/>
</p>

### Description

This Final Degree Project presents the design and development of a simulated universe using a modified cellular automaton, inspired by Conway's classic Game of Life. Based on a grid of cells representing different types of matter—such as stars, planets, and black holes—astrophysical rules have been implemented that allow us to observe the dynamic evolution of the system over thousands of generations.

### Objectives

- Main Objective: Simulate structures analogous to those observed in astrophysical systems using genetic algorithms to optimize initial configurations and transition rules in variants of the Game of Life

- Pattern Analysis: Analyze the patterns obtained through machine learning techniques to classify behavior and structure

- Astrophysical Parallels: Identify similarities between the patterns generated and real astrophysical and cosmological phenomena.

### Technologies Used
- Python 3.11.4
- NumPy 1.24.3: Matrix data structures
- Pygame: Interactive visualization of the cellular automaton
- Matplotlib 3.10.3: Graphical analysis of results
- TensorFlow/Keras: Neural networks and machine learning
- Tkinter: Advanced graphical interface

### Scientific Methodology
#### Genetic Algorithms

- Population: 10-30 individuals depending on the experiment
- Selection: Elitist, retaining the top 50%
- Crossover: Single point respecting spatial consistency
- Mutation: 1% rate to maintain diversity

#### Machine Learning
- Feature extraction: 35 parameters quantifying global and regional distributions
- Neural network: Architecture with dense layers (128→64→8 neurons)
- Autoencoder: Compression of 200×200×3 images to 64-dimensional vectors
- Clustering: K-Means to identify 8 types of structural patterns

### Astrophysical Analogies
The system demonstrates structural similarities with real cosmic phenomena:
- Cosmic web: Filamentary patterns similar to the large-scale distribution of matter
- Hierarchical formation: Evolution of energy → asteroids → planets → stars → black holes
- Self-organization: Spontaneous emergence of complex structures without external control

Jorge Bravo Mateos

Bachelor's Degree in Computer Engineering

Complutense University of Madrid

Academic year 2024-25

Director: Rafael del Vado Vírseda

The complete thesis is available in the repository: [TFG_JorgeBravoMateos](https://github.com/jbrma/TFG_JorgeBravo/blob/main/memoria/TFG_JorgeBravoMateos.pdf)
