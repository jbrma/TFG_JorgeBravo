
'''

Vamos a añadir un sistema de detección de patrones específicos como:

- Block (estático)
- Blinker (oscilador)
- Glider (nave espacial que se mueve diagonalmente)

¿Cómo los vamos a detectar?

- Usaremos detección por plantilla: buscamos submatrices dentro del 
grid que coincidan con patrones conocidos.

- Con suficientes generaciones, el algoritmo genético puede descubrir 
automáticamente gliders, blinker u osciladores.

- Si lo dejas correr varias veces, deberías empezar a ver gliders 
emerger naturalmente porque:

  Se mueven.
  Oscilan.
  Dan un alto puntaje.

Vamos a llevar esto al siguiente nivel: detectar glider guns y 
osciladores de período mayor a 2.

Objetivo:

- Detectar glider guns: estructuras que periódicamente generan gliders.

- Detectar osciladores con período arbitrario (3, 4, ...).

'''


import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from matplotlib.widgets import Button

'''

Paso 1: Patrones conocidos

Primero, definimos las plantillas de patrones:
        
'''


# Diccionario de patrones conocidos (rotaciones incluidas opcionalmente)

KNOWN_PATTERNS = {
    "glider": [
        np.array([
            [0,1,0],
            [0,0,1],
            [1,1,1]
        ])
    ],
    "blinker": [
        np.array([
            [1,1,1]
        ]),
        np.array([
            [1],
            [1],
            [1]
        ])
    ],
    "block": [
        np.array([
            [1,1],
            [1,1]
        ])
    ],
    "beehive": [
        np.array([
            [0,1,1,0],
            [1,0,0,1],
            [0,1,1,0]
        ]),
        np.array([
            [0,1,0],
            [1,0,1],
            [1,0,1],
            [0,1,0]
        ])
    ]
}



'''

Paso 2: Función para buscar patrones en el grid


'''


def match_pattern(grid, pattern):
    """Busca un patrón dentro del grid."""
    for i in range(grid.shape[0] - pattern.shape[0] + 1):
        for j in range(grid.shape[1] - pattern.shape[1] + 1):
            subgrid = grid[i:i+pattern.shape[0], j:j+pattern.shape[1]]
            if np.array_equal(subgrid, pattern):
                return True
    return False

def detect_known_patterns(grid):
    """Detecta si hay algún patrón conocido en el grid."""
    matches = []
    for name, templates in KNOWN_PATTERNS.items():
        for pat in templates:
            if match_pattern(grid, pat):
                matches.append(name)
    return matches


'''

Detección de osciladores de período N:

Durante la simulación de generaciones en fitness(), guardamos un 
historial de los estados. Si el patrón se repite después de N 
generaciones, entonces es un oscilador de período N.

'''


def detect_oscillator_period(history):
    """Detecta el período de oscilación del patrón (si existe)."""
    for i in range(1, len(history)):
        if np.array_equal(history[0], history[i]):
            return i  # patrón se repite tras i generaciones
    return None  # no hay ciclo detectado


'''

Detección de glider guns:

Una glider gun es un patrón estacionario que emite gliders periódicamente. 

Lo podemos detectar observando que:

- El patrón principal permanece en su lugar.

- Aparecen nuevos gliders que se alejan consistentemente del centro.

Estrategia simple para detectar un glider gun:
    
- Comparar el grid inicial con las últimas generaciones.

- Si hay gliders detectados lejos del centro (con suficiente movimiento), 
  y el patrón base no cambia mucho → es probablemente una glider gun.

'''


def detect_glider_gun(history):
    """Detecta si el patrón actúa como una glider gun."""
    if len(history) < 10:
        return False

    # Requiere que al menos un glider aparezca lejos del centro
    last = history[-1]
    glider_detected = False

    for name, templates in KNOWN_PATTERNS.items():
        if name != "glider":
            continue
        for pat in templates:
            if match_pattern(last, pat):
                glider_detected = True

    if not glider_detected:
        return False

    # Verificamos si el patrón base (centro) sigue similar
    center_start = history[0][5:15, 5:15]
    center_end = history[-1][5:15, 5:15]
    static_center = np.sum(np.abs(center_start - center_end)) < 5  # casi sin cambio

    return static_center and glider_detected




# Parámetros del Juego de la Vida

GRID_SIZE = 30
MAX_GENERATIONS = 100

# Parámetros del Algoritmo Genético

POPULATION_SIZE = 30
CHROMOSOME_SIZE = GRID_SIZE * GRID_SIZE
MUTATION_RATE = 0.01
NUM_EVOLUTIONS = 50  # Cuántas veces se ejecuta el algoritmo genético


def random_individual():
    """Crea un individuo aleatorio (una cuadrícula de 0s y 1s)."""
    return np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE))


def fitness(individual):
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
    
    # BONUS si se detecta un patrón específico
    detected = detect_known_patterns(grid)
    if "glider" in detected:
        score += 100  # gran recompensa
    elif "blinker" in detected:
        score += 40
    elif "block" in detected:
        score += 20
        
    # Detectar oscilador de período N
    period = detect_oscillator_period(history)
    if period and period > 2:
        score += 50 + period * 2  # recompensa creciente

    # Detectar glider gun
    if detect_glider_gun(history):
        score += 300  # recompensa alta

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
    scored_population = [(fitness(ind), ind) for ind in population]
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
    def __init__(self, best_individual, max_generations):
        self.fig = plt.figure(figsize=(10, 8))
        self.max_generations = max_generations
        
        # Configuración de la cuadrícula
        self.gridspec = plt.GridSpec(3, 2, 
                                    width_ratios=[1, 1],  
                                    height_ratios=[8, 8, 5], 
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
        self.history = [self.grid.copy()]
        
        # Gráficos
        self.img = self.ax_sim.imshow(self.grid, cmap='Greys', interpolation='nearest')
        self.info_text = self.info_ax.text(0.5, 0.5, "", ha='center', va='center', fontsize=12)
        self.patches = []  # Para resaltar patrones
        
        # Boton
        self.ax_play = plt.axes([0.45, 0.08, 0.1, 0.05])
        self.btn_play = Button(self.ax_play, 'Empezar', color='lightblue', hovercolor='skyblue')
        self.btn_play.on_clicked(self.toggle_simulation)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)

    def toggle_simulation(self, event=None):
        if self.current_step >= self.max_generations:
            return
        self.sim_running = not self.sim_running
        self.btn_play.label.set_text('Reanudar' if not self.sim_running else 'Pausar')

    def update_info(self):
        info = (
            f"Generación: {self.current_step}/{self.max_generations} | "
            f"Células vivas: {np.sum(self.grid)}\n"
            f"Patrones detectados: {detect_known_patterns(self.grid)}"
        )
        if self.current_step >= self.max_generations:
            final_history = []
            grid = best_individual.copy()
            for _ in range(MAX_GENERATIONS):
                grid = game_of_life_step(grid)
                final_history.append(grid.copy())

            period = detect_oscillator_period(self.history)
            is_gun = detect_glider_gun(self.history)
            patterns_found = detect_known_patterns(final_history[-1])

            info += ("\nAnálisis del mejor patrón:")

            if period:
                info += (f"\n→ Oscilador de período {period}")
            if is_gun:
                info += ("\n→ ¡Glider gun detectada!")
            if patterns_found:
                info += (f"\n→ Contiene: {', '.join(patterns_found)}")
            
        self.info_text.set_text(info)

    def update_frame(self, frame):
        if self.sim_running and self.current_step < self.max_generations:
            self.grid = game_of_life_step(self.grid)
            self.history.append(self.grid.copy())
            self.current_step += 1
            
            # Dibujar patrones detectados
            for patch in self.patches:
                patch.remove()
            self.patches.clear()
            
            for name, templates in KNOWN_PATTERNS.items():
                for pat in templates:
                    for i in range(self.grid.shape[0] - pat.shape[0] + 1):
                        for j in range(self.grid.shape[1] - pat.shape[1] + 1):
                            if np.array_equal(self.grid[i:i+pat.shape[0], j:j+pat.shape[1]], pat):
                                rect = plt.Rectangle((j-0.5, i-0.5), pat.shape[1], pat.shape[0], 
                                                    linewidth=1, edgecolor='red', facecolor='none')
                                self.ax_sim.add_patch(rect)
                                self.patches.append(rect)
        
        self.img.set_data(self.grid)
        self.update_info()
        return [self.img, self.info_text] + self.patches

    def start(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update_frame, 
            frames=self.max_generations,
            interval=200, 
            blit=True
        )
        plt.show()

sim = Simulation(best_individual, MAX_GENERATIONS)
sim.start()