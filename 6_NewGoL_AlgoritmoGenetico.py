import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from collections import Counter
import random

# Parámetros del universo
SIZE = 50
PROB_MASS_LOSS = 0.15
MASS_LOSS_RATE = 0.12

POP_SIZE = 10 # Poblacion
SIM_STEPS = 20 # Numero de pasos para evaluar cada universo
MUTATION_RATE = 0.01
GEN_STEPS = 20  # Numero de generaciones para el algoritmo genético

# Tipos de celda
SPACE, STAR, PLANET, ASTEROID, ENERGY, BLACK_HOLE, ANTIMATTER = 0, 1, 2, 3, 4, 5, 6

# Configuración de colores
COLORS = {
    SPACE: (0, 0, 0),
    STAR: (1, 1, 0),
    PLANET: (0, 0.7, 0.7),
    ASTEROID: (0, 0, 1),
    ENERGY: (1, 0.2, 0),
    BLACK_HOLE: (0.1, 0.1, 0.4),
    ANTIMATTER: (0.3, 0.7, 1)
}

MASSES = {
    SPACE: 0,
    STAR: 10,
    PLANET: 4,
    ASTEROID: 1,
    ENERGY: 0.5,
    BLACK_HOLE: 30,
    ANTIMATTER: 4
}

# Inicialización aleatoria
def create_universe():
    return np.random.choice(
        [SPACE, STAR, PLANET, ASTEROID, ENERGY, BLACK_HOLE, ANTIMATTER],
        size=(SIZE, SIZE),
        p=[0.77, 0.03, 0.07, 0.07, 0.05, 0.002, 0.008]
    )

def initialize_masses(grid):
    masses = np.zeros_like(grid, dtype=float)
    for t, v in MASSES.items():
        masses[grid == t] = v
    return masses

def local_gravity(grid, x, y):
    radius = 2
    fx, fy = 0.0, 0.0
    for i in range(max(0, x-radius), min(SIZE, x+radius+1)):
        for j in range(max(0, y-radius), min(SIZE, y+radius+1)):
            if i == x and j == y:
                continue
            mass = MASSES[grid[i, j]]
            dx, dy = i - x, j - y
            dist = np.sqrt(dx**2 + dy**2) + 0.1
            f = mass / (dist**2)
            fx += f * dx / dist
            fy += f * dy / dist
    return fx, fy

def update_universe(grid, masses):
    new_grid = grid.copy()
    new_masses = masses.copy()
    for x in range(SIZE):
        for y in range(SIZE):
            cell = grid[x, y]
            mass = masses[x, y]
            neighbors = grid[max(0, x-1):x+2, max(0, y-1):y+2]
            neighbor_masses = masses[max(0, x-1):x+2, max(0, y-1):y+2]
            
            same_type = (neighbors == cell)
            if cell != SPACE and np.sum(same_type) > 1:
                total_mass = np.sum(neighbor_masses[same_type])
                new_masses[x, y] = total_mass

            if cell != SPACE and ANTIMATTER in neighbors:
                new_grid[x, y] = ENERGY
                new_masses[x, y] = MASSES[ENERGY]
                continue
            if cell == ANTIMATTER and np.any((neighbors > SPACE) & (neighbors != ANTIMATTER)):
                new_grid[x, y] = ENERGY
                new_masses[x, y] = MASSES[ENERGY]
                continue

            if cell == SPACE:
                if np.count_nonzero(neighbors == ENERGY) >= 3 and np.random.rand() < 0.02:
                    new_grid[x, y] = ASTEROID
                    new_masses[x, y] = MASSES[ASTEROID]
                if np.count_nonzero(neighbors == ENERGY) >= 2 and np.count_nonzero(neighbors == ASTEROID) >= 2 and np.random.rand() < 0.01:
                    new_grid[x, y] = ASTEROID
                    new_masses[x, y] = MASSES[ASTEROID]
                if np.count_nonzero(neighbors == ASTEROID) >= 4 and np.random.rand() < 0.01:
                    new_grid[x, y] = PLANET
                    new_masses[x, y] = MASSES[PLANET]
                if np.count_nonzero(neighbors == PLANET) >= 4 and np.random.rand() < 0.01:
                    new_grid[x, y] = STAR
                    new_masses[x, y] = MASSES[STAR]
                if np.count_nonzero(neighbors == STAR) >= 5 and np.random.rand() < 0.005:
                    new_grid[x, y] = BLACK_HOLE
                    new_masses[x, y] = MASSES[BLACK_HOLE]
                if np.count_nonzero(neighbors == STAR) >= 2 and np.random.rand() < 0.01:
                    new_grid[x, y] = ENERGY
                    new_masses[x, y] = MASSES[ENERGY]
                if np.count_nonzero(neighbors == BLACK_HOLE) >= 2 and np.random.rand() < 0.01:
                    new_grid[x, y] = BLACK_HOLE
                    new_masses[x, y] = MASSES[BLACK_HOLE]

            elif cell == STAR:
                if mass > 25 and np.random.rand() < 0.01:
                    new_grid[x, y] = BLACK_HOLE
                    new_masses[x, y] = MASSES[BLACK_HOLE]
                if np.random.rand() < 0.02:
                    for i in range(max(0, x-1), min(SIZE, x+2)):
                        for j in range(max(0, y-1), min(SIZE, y+2)):
                            if grid[i, j] == SPACE and np.random.rand() < 0.2:
                                new_grid[i, j] = ENERGY
                                new_masses[i, j] = MASSES[ENERGY]

            elif cell == PLANET:
                if mass > 10 and np.random.rand() < 0.01:
                    new_grid[x, y] = STAR
                    new_masses[x, y] = MASSES[STAR]
                if np.random.rand() < 0.001:
                    new_grid[x, y] = ASTEROID
                    new_masses[x, y] = MASSES[ASTEROID]

            elif cell == ASTEROID:
                if np.random.rand() < 0.002:
                    new_grid[x, y] = SPACE
                    new_masses[x, y] = 0

            elif cell == ENERGY:
                if np.count_nonzero(neighbors == ENERGY) >= 3 and np.random.rand() < 0.01:
                    new_grid[x, y] = ASTEROID
                    new_masses[x, y] = MASSES[ASTEROID]
                if np.random.rand() < 0.01:
                    new_grid[x, y] = SPACE
                    new_masses[x, y] = 0

            elif cell == BLACK_HOLE:
                if np.random.rand() < PROB_MASS_LOSS:
                    mass_lost = mass * MASS_LOSS_RATE
                    new_masses[x, y] -= mass_lost

                for i in range(max(0, x-1), min(SIZE, x+2)):
                    for j in range(max(0, y-1), min(SIZE, y+2)):
                        if (i != x or j != y) and grid[i, j] not in [SPACE, BLACK_HOLE]:
                            if np.random.rand() < 0.5:
                                new_grid[i, j] = SPACE
                                new_masses[i, j] = 0
                if np.random.rand() < 0.001:
                    for i in range(max(0, x-1), min(SIZE, x+2)):
                        for j in range(max(0, y-1), min(SIZE, y+2)):
                            if grid[i, j] == SPACE and np.random.rand() < 0.2:
                                new_grid[i, j] = ENERGY
                                new_masses[i, j] = MASSES[ENERGY]

                if new_masses[x, y] < 1:
                    remaining_energy = new_masses[x, y]
                    energy_per_cell = remaining_energy / 8
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < SIZE and 0 <= ny < SIZE:
                                if new_grid[nx, ny] == SPACE:
                                    new_grid[nx, ny] = ENERGY
                                    new_masses[nx, ny] = energy_per_cell
                                elif new_grid[nx, ny] == ENERGY:
                                    new_masses[nx, ny] += energy_per_cell
                    
                    new_grid[x, y] = SPACE
                    new_masses[x, y] = 0

    return new_grid, new_masses

def render(grid):
    img = np.zeros((SIZE, SIZE, 3))
    for t, color in COLORS.items():
        img[grid == t] = color
    return img

def count_cells(grid):
    counts = Counter(grid.flatten())
    return [counts.get(t, 0) for t in range(7)]

# =============== ALGORITMO GENÉTICO ====================

def fitness(final_grid, final_masses):
    total_mass = np.sum(final_masses)
    stars = np.sum(final_grid == STAR)
    planets = np.sum(final_grid == PLANET)
    #black_holes = np.sum(final_grid == BLACK_HOLE)
    return total_mass + 10 * (stars + planets)

def evolve_population(population):
    best_universe = None
    best_score = -np.inf
    current_step = 0
    
    for _ in range(GEN_STEPS):
        scores = []
        
        for universe in population:
            # Simular universo
            current_grid = universe.copy()
            current_mass = initialize_masses(current_grid)
            for _ in range(SIM_STEPS):
                current_grid, current_mass = update_universe(current_grid, current_mass)
            
            # Evaluar fitness
            score = fitness(current_grid, current_mass)
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_universe = current_grid.copy()

        print(f"Gen {current_step} -> Mejor fitness: {best_score}, Promedio: {np.mean(scores)}")
        current_step += 1
        best_score = 0
        
        # Selección y reproducción
        sorted_indices = np.argsort(scores)[::-1]
        selected = [population[i] for i in sorted_indices[:POP_SIZE//2]]
        
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        
        population = new_pop
    
    return best_universe, best_score

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, SIZE-1)
    child = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def mutate(universe):
    mutation_mask = np.random.rand(SIZE, SIZE) < MUTATION_RATE
    random_values = np.random.randint(0, 7, size=(SIZE, SIZE))
    mutated = np.where(mutation_mask, random_values, universe)
    return mutated

# ========== Interfaz gráfica y animación ========

class UniverseSimulator:
    def __init__(self):
        self.fig = plt.figure(figsize=(8, 6))
        self.gridspec = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])
        
        # Panel izquierdo: Mejor universo evolucionado
        self.ax_best = self.fig.add_subplot(self.gridspec[0, 0])
        self.ax_best.set_title("Mejor Universo Inicial")
        self.ax_best.axis('off')
        
        # Panel derecho: Simulación en vivo
        self.ax_sim = self.fig.add_subplot(self.gridspec[0, 1])
        self.ax_sim.set_title("Simulación")
        self.ax_sim.axis('off')
        
        # Panel inferior: Estadísticas
        self.ax_stats = self.fig.add_subplot(self.gridspec[1, :])
        self.ax_stats.axis('off')
        
        # Evolucionar población inicial
        initial_pop = [create_universe() for _ in range(POP_SIZE)]
        self.best_universe, score = evolve_population(initial_pop)
        print(f"\nMejor fitness obtenido: {score}")
        
        # Inicializar simulación
        self.grid = self.best_universe.copy()
        self.masses = initialize_masses(self.grid)
        self.frame = [0]
        self.paused = True
        
        # Elementos gráficos
        self.img_best = self.ax_best.imshow(render(self.best_universe), interpolation='nearest')
        self.img_sim = self.ax_sim.imshow(render(self.grid), interpolation='nearest')
        
        # Estadísticas
        self.stats_text = self.ax_stats.text(0.5, 0.5, "", ha='center', va='center')
        
        # Botones
        self.btn_ax = plt.axes([0.45, 0.05, 0.1, 0.04])
        self.btn = Button(self.btn_ax, 'Iniciar', color='lightgoldenrodyellow')
        self.btn.on_clicked(self.toggle_sim)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
    
    def toggle_sim(self, event):
        self.paused = not self.paused
        self.btn.label.set_text('Pausar' if not self.paused else 'Reanudar')
    
    def update_stats(self):
        counts = count_cells(self.grid)
        stats = (
            f"Generación: {self.frame[0]} | "
            f"Estrellas: {counts[STAR]} | "
            f"Planetas: {counts[PLANET]} | "
            f"Agujeros Negros: {counts[BLACK_HOLE]}"
        )
        self.stats_text.set_text(stats)
    
    def update_frame(self, i):
        if not self.paused:
            self.grid, self.masses = update_universe(self.grid, self.masses)
            self.img_sim.set_data(render(self.grid))
            self.frame[0] += 1
            self.update_stats()
        return self.img_sim, self.stats_text
    
    def start(self):
        anim = animation.FuncAnimation(self.fig, self.update_frame, interval=100, blit=False)
        plt.show()

# Iniciar simulación
sim = UniverseSimulator()
sim.start()