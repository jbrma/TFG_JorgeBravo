
'''

Usaremos un Algoritmo Genético (GA) para evolucionar patrones 
iniciales en el Juego de la Vida de forma que sobrevivan el 
mayor tiempo posible o generen el mayor número de células 
vivas tras cierto número de generaciones.

Qué hace este código:

- Crea una población de patrones aleatorios (matrices 30x30).

- Cada uno se evalúa según cuántas células quedan vivas tras 20 pasos.

- El algoritmo selecciona los mejores, hace crossover y mutaciones.

- Repite esto por 50 generaciones.

- Muestra el mejor patrón inicial encontrado y lo simula.

'''


import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from matplotlib.widgets import Button

# Parámetros del Juego de la Vida
GRID_SIZE = 30
MAX_GENERATIONS = 20

# Parámetros del Algoritmo Genético
POPULATION_SIZE = 30
CHROMOSOME_SIZE = GRID_SIZE * GRID_SIZE
MUTATION_RATE = 0.01
NUM_EVOLUTIONS = 50  # Cuántas veces se ejecuta el algoritmo genético

def random_individual():
    """Crea un individuo aleatorio (una cuadrícula de 0s y 1s)."""
    return np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE))

def fitness(individual):
    """Evalúa cuántas células vivas sobreviven después de X generaciones."""
    grid = individual.copy()
    for _ in range(MAX_GENERATIONS):
        grid = game_of_life_step(grid)
    return np.sum(grid)

def fitness2(individual):
    """Evalúa si el patrón se mueve, oscila o se mantiene dinámico."""
    grid = individual.copy()
    history = []
    movement_score = 0
    oscillation_score = 0

    last_com = None  # centro de masa anterior

    for gen in range(MAX_GENERATIONS):
        grid = game_of_life_step(grid)
        history.append(grid.copy())

        # Medir movimiento del centro de masa
        y, x = np.nonzero(grid)
        if len(x) > 0:
            com = (np.mean(y), np.mean(x))
            if last_com:
                dy = com[0] - last_com[0]
                dx = com[1] - last_com[1]
                movement_score += np.sqrt(dx**2 + dy**2)
            last_com = com

        # Medir oscilación: comparando con generaciones anteriores
        for prev in history[:-1]:
            if np.array_equal(prev, grid):
                oscillation_score += 1
                break

    # Células vivas al final (actividad)
    alive_cells = np.sum(grid)

    # Ponderar cada componente
    score = alive_cells + movement_score * 2 + oscillation_score * 5
    return score

def game_of_life_step(grid):
    """Realiza un paso del juego de la vida."""
    new_grid = np.zeros_like(grid)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            total = np.sum(grid[max(0,i-1):i+2, max(0,j-1):j+2]) - grid[i,j]
            if grid[i,j] == 1 and (2 <= total <= 3):
                new_grid[i,j] = 1
            elif grid[i,j] == 0 and total == 3:
                new_grid[i,j] = 1
    return new_grid

def crossover(parent1, parent2):
    """Cruza dos padres para generar un hijo."""
    point = random.randint(0, CHROMOSOME_SIZE-1)
    flat1 = parent1.flatten()
    flat2 = parent2.flatten()
    child_flat = np.concatenate((flat1[:point], flat2[point:]))
    return child_flat.reshape((GRID_SIZE, GRID_SIZE))

def mutate(individual):
    """Muta un individuo con cierta probabilidad."""
    mutation_mask = np.random.rand(GRID_SIZE, GRID_SIZE) < MUTATION_RATE
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual

# Inicializa población
population = [random_individual() for _ in range(POPULATION_SIZE)]

best_fitness = 0
best_individual = None

for generation in range(NUM_EVOLUTIONS):
    # Evaluar
    scored_population = [(fitness2(ind), ind) for ind in population]
    scored_population.sort(reverse=True, key=lambda x: x[0])

    if scored_population[0][0] > best_fitness:
        best_fitness = scored_population[0][0]
        best_individual = scored_population[0][1]

    print(f"Gen {generation}: Mejor fitness = {scored_population[0][0]}")

    # Selección: top 50%
    selected = [ind for _, ind in scored_population[:POPULATION_SIZE // 2]]

    # Reproducción
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)

    population = new_population

class Simulation:
    def __init__(self, best_individual):
        self.fig = plt.figure(figsize=(10, 6))
        
        self.gridspec = plt.GridSpec(3, 2, 
                                   width_ratios=[1, 1],  
                                   height_ratios=[8, 8, 2],
                                   hspace=0.3)
        
        # Panel izquierdo: Mejor patrón inicial
        self.ax_initial = self.fig.add_subplot(self.gridspec[0:2, 0])
        self.ax_initial.set_title("Mejor Patrón Inicial", pad=20)
        self.ax_initial.imshow(best_individual, cmap='Greys')
        self.ax_initial.axis('off')
        
        # Panel derecho: Simulación en vivo
        self.ax_sim = self.fig.add_subplot(self.gridspec[0:2, 1])
        self.ax_sim.set_title("Simulación Evolutiva", pad=20)
        self.ax_sim.axis('off')
        
        # Info
        self.info_ax = self.fig.add_subplot(self.gridspec[2, :])
        self.info_ax.axis('off')
        
        self.grid = best_individual.copy()
        self.sim_running = False
        self.current_step = 0
        
        # Gráficos
        self.img = self.ax_sim.imshow(self.grid, cmap='Greys', interpolation='nearest')
        self.info_text = self.info_ax.text(0.5, 0.5, 
                                          "", 
                                          ha='center', 
                                          va='center',
                                          fontsize=12)
        
        # Boton
        self.ax_play = plt.axes([0.45, 0.08, 0.1, 0.05]) 
        self.btn_play = Button(self.ax_play, 'Empezar', color='lightblue', hovercolor='skyblue')
        self.btn_play.on_clicked(self.toggle_simulation)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)

    def toggle_simulation(self, event=None):
        self.sim_running = not self.sim_running
        self.btn_play.label.set_text('Reanudar' if not self.sim_running else 'Pausar')
        self.btn_play.label.set_backgroundcolor((1, 1, 1, 0.1))

    def update_info(self):
        info = (
            f"Generación: {self.current_step} | "
            f"Células vivas: {np.sum(self.grid)} "
        )
        self.info_text.set_text(info)

    def update_frame(self, frame):
        if self.sim_running:
            self.grid = game_of_life_step(self.grid)
            self.img.set_data(self.grid)
            self.current_step += 1
            self.update_info()
        return [self.img, self.info_text]

    def start(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update_frame, 
            frames=1000,
            interval=200, 
            blit=True
        )
        plt.show()

# Iniciar la simulación con el mejor individuo
sim = Simulation(best_individual)
sim.start()


'''

Vamos a adaptar el algoritmo genético para que evolucione patrones 
que tengan comportamientos interesantes como:

- Osciladores: Patrones que vuelven a un estado anterior tras algunos 
pasos.

- Naves espaciales (como gliders): Patrones que se desplazan en la 
cuadrícula.

-Patrones dinámicos: Que mantengan actividad (no mueran o se estabilicen).

Objetivo modificado:
    
Evaluaremos los individuos no solo por cuántas células sobreviven, 
sino por si:

- El patrón cambia con el tiempo (no es estático).
- Se desplaza (detectaremos movimiento del centro de masa).
- Tiene ciclo (patrón se repite cada cierto número de pasos).

Estrategia de fitness avanzada:
    
Vamos a diseñar una función de fitness más inteligente:

- Detecta si el patrón cambia con el tiempo.
- Calcula la cantidad de movimiento (centro de masa entre generaciones).
- Detecta si hay un ciclo (comparando estados anteriores).

La nueva función "fitness2":
    
Suma puntos por:

- Oscilación (el patrón se repite).
- Movimiento real (com se desplaza).
- Actividad viva (no se extingue).

Así, favorecemos patrones como el glider, que:
    
- Se mueven.
- Oscilan.
- Siguen vivos.

Recomendaciones adicionales:
    
Para encontrar patrones tipo glider:

- Reduce el tamaño del grid a 10x10 o 15x15 para acelerar la evolución.
- Aumenta MAX_GENERATIONS a 50 para darles más tiempo a los patrones.
- Guarda los mejores patrones para visualizarlos luego.

'''


def fitness2(individual):
    """Evalúa si el patrón se mueve, oscila o se mantiene dinámico."""
    grid = individual.copy()
    history = []
    movement_score = 0
    oscillation_score = 0

    last_com = None  # centro de masa anterior

    for gen in range(MAX_GENERATIONS):
        grid = game_of_life_step(grid)
        history.append(grid.copy())

        # Medir movimiento del centro de masa
        y, x = np.nonzero(grid)
        if len(x) > 0:
            com = (np.mean(y), np.mean(x))
            if last_com:
                dy = com[0] - last_com[0]
                dx = com[1] - last_com[1]
                movement_score += np.sqrt(dx**2 + dy**2)
            last_com = com

        # Medir oscilación: comparando con generaciones anteriores
        for prev in history[:-1]:
            if np.array_equal(prev, grid):
                oscillation_score += 1
                break

    # Células vivas al final (actividad)
    alive_cells = np.sum(grid)

    # Ponderar cada componente
    score = alive_cells + movement_score * 2 + oscillation_score * 5
    return score
