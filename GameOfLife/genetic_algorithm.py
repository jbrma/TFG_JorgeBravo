import numpy as np
import random
from tqdm import tqdm 


def create_population(size, cellsX, cellsY):
    """ Creates a random population """
    return [np.random.randint(2, size=(cellsX, cellsY)) for _ in range(size)]


def calculate_fitness(board, cellsX, cellsY):
    """Calculate the fitness function of the board simulating some generations """

    generations = 10  # number of iterations
    gameState = np.copy(board)
    fitness = 0

    for _ in range(generations):
        newGameState = np.copy(gameState)
        for y in range(cellsY):
            for x in range(cellsX):
                num_neigh = (gameState[(x-1) % cellsX, (y-1) % cellsY] +
                             gameState[(x) % cellsX, (y-1) % cellsY] +
                             gameState[(x+1) % cellsX, (y-1) % cellsY] +
                             gameState[(x-1) % cellsX, (y) % cellsY] +
                             gameState[(x+1) % cellsX, (y) % cellsY] +
                             gameState[(x-1) % cellsX, (y+1) % cellsY] +
                             gameState[(x) % cellsX, (y+1) % cellsY] +
                             gameState[(x+1) % cellsX, (y+1) % cellsY])

                if gameState[x, y] == 0 and num_neigh == 3:
                    newGameState[x, y] = 1
                elif gameState[x, y] == 1 and (num_neigh < 2 or num_neigh > 3):
                    newGameState[x, y] = 0

        fitness += np.sum(newGameState)  # Its the living cell quantity 
        gameState = newGameState

    return fitness


def select_parents(population, fitness_scores):
    """ Select the best individuals to proceed into the next generation """
    subgroup = 5
    selected = random.sample(list(zip(population, fitness_scores)), subgroup)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]  # Returns the best ones


def crossover(parent1, parent2, cellsX):
    """ Crossover between two parents """
    point = random.randint(1, cellsX - 1)
    child = np.vstack((parent1[:point], parent2[point:]))
    return child

def mutate(board, mutation_rate, cellsX, cellsY):
    """ Aplies mutations into the board (child) """
    for x in range(cellsX):
        for y in range(cellsY):
            if random.random() < mutation_rate:
                board[x, y] = 1 - board[x, y]  # Changes between 0 and 1
    return board

def run_genetic_algorithm(cellsX, cellsY, population_size=50, num_generations=100, mutation_rate=0.1):
    """Ejecuta el algoritmo genético y retorna el mejor tablero encontrado."""
    population = create_population(population_size, cellsX, cellsY)

    for generation in tqdm(range(num_generations), desc="Simulando generaciones", unit="gen"):
        fitness_scores = [calculate_fitness(individual, cellsX, cellsY) for individual in population]
        new_population = []

        for _ in range(population_size // 2):
            parent1 = select_parents(population, fitness_scores)
            parent2 = select_parents(population, fitness_scores)
            child1 = crossover(parent1, parent2, cellsX)
            child2 = crossover(parent2, parent1, cellsX)
            child1 = mutate(child1, mutation_rate, cellsX, cellsY)
            child2 = mutate(child2, mutation_rate, cellsX, cellsY)
            new_population.extend([child1, child2])

        population = new_population

    best_individual = max(population, key=lambda board: calculate_fitness(board, cellsX, cellsY))
    return best_individual