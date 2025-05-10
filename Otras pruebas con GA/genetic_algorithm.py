import numpy as np
import random
from tqdm import tqdm
import csv


def create_population(size, cellsX, cellsY):
    """ Creates a random population """
    return [np.random.randint(2, size=(cellsX, cellsY)) for _ in range(size)]


def calculate_fitness_cell(board, x, y):
    """Calculate the fitness for a single cell based on its neighborhood"""
    cellsX, cellsY = board.shape
    num_neigh = (
        board[(x-1) % cellsX, (y-1) % cellsY] +
        board[(x) % cellsX, (y-1) % cellsY] +
        board[(x+1) % cellsX, (y-1) % cellsY] +
        board[(x-1) % cellsX, (y) % cellsY] +
        board[(x+1) % cellsX, (y) % cellsY] +
        board[(x-1) % cellsX, (y+1) % cellsY] +
        board[(x) % cellsX, (y+1) % cellsY] +
        board[(x+1) % cellsX, (y+1) % cellsY]
    )

    if board[x, y] == 1:
        return 2 <= num_neigh <= 3  # Survives if 2-3 neighbors
    elif board[x, y] == 0:
        return num_neigh == 3  # Born if exactly 3 neighbors
    return 0

def select_mate(neighborhood, fitness_scores):
    """Select a mate for a cell based on fitness scores within its neighborhood"""
    candidates = [(i, j) for i, row in enumerate(neighborhood) for j, _ in enumerate(row)]
    candidates_fitness = [fitness_scores[i, j] for i, j in candidates]
    best_index = np.argmax(candidates_fitness)
    return candidates[best_index]


def crossover_and_mutate(cell, mate, mutation_rate):
    """Perform crossover and mutation for a cell"""
    new_state = cell if random.random() > 0.5 else mate
    if random.random() < mutation_rate:
        new_state = 1 - new_state  # Flip the state
    return new_state



""" CSV """

def save_generation_data(filename, generation, population, fitness_scores):
    """ Save the data of each generation in a csv """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if generation == 0:
            headers = ['Generation', 'Individual', 'Fitness', 'Density', 'Avg_Neighbors']
            writer.writerow(headers)

        for i, (board, fitness) in enumerate(zip(population, fitness_scores)):
            
            density = np.sum(board) / (board.shape[0] * board.shape[1])
            avg_neighbors = calculate_avg_neighbors(board)
            writer.writerow([generation, i, fitness, density, avg_neighbors])

def calculate_avg_neighbors(board):
    
    cellsX, cellsY = board.shape
    total_neighbors = 0
    for y in range(cellsY):
        for x in range(cellsX):
            num_neigh = (board[(x-1) % cellsX, (y-1) % cellsY] +
                         board[(x) % cellsX, (y-1) % cellsY] +
                         board[(x+1) % cellsX, (y-1) % cellsY] +
                         board[(x-1) % cellsX, (y) % cellsY] +
                         board[(x+1) % cellsX, (y) % cellsY] +
                         board[(x-1) % cellsX, (y+1) % cellsY] +
                         board[(x) % cellsX, (y+1) % cellsY] +
                         board[(x+1) % cellsX, (y+1) % cellsY])
            total_neighbors += num_neigh
    return total_neighbors / (cellsX * cellsY)