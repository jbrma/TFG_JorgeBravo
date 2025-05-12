import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from collections import Counter

# Parámetros del universo
SIZE = 200
PROB_MASS_LOSS = 0.15      # 15% de probabilidad de perder masa por paso
MASS_LOSS_RATE = 0.12      # 12% de masa perdida en cada evento

# Tipos de celda
SPACE = 0
STAR = 1
PLANET = 2
ASTEROID = 3
ENERGY = 4
BLACK_HOLE = 5
ANTIMATTER = 6

# Colores para visualización
COLORS = {
    SPACE: (0, 0, 0),
    STAR: (1, 1, 0),
    PLANET: (0, 0.7, 0.7),
    ASTEROID: (0, 0, 1),
    ENERGY: (1, 0.2, 0),
    BLACK_HOLE: (0.1, 0.1, 0.4),
    ANTIMATTER: (0.3, 0.7, 1)
}

# Masas relativas para cada tipo
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
    # Inicializa el universo con una distribución aleatoria de celdas
    return np.random.choice(
        [SPACE, STAR, PLANET, ASTEROID, ENERGY, BLACK_HOLE, ANTIMATTER],
        size=(SIZE, SIZE),
        p=[0.77, 0.03, 0.07, 0.07, 0.05, 0.002, 0.008]
    )

# Inicializa la matriz de masas
def initialize_masses(grid):
    # Asigna la masa inicial según el tipo de celda
    masses = np.zeros_like(grid, dtype=float)
    for t, v in MASSES.items():
        masses[grid == t] = v
    return masses

# Calcula la gravedad local simplificada
def local_gravity(grid, x, y):
    # Calcula la fuerza gravitatoria local en la celda (x, y)
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

# Actualiza el universo según las reglas
def update_universe(grid, masses):
    # Aplica las reglas de evolución para cada celda
    new_grid = grid.copy()
    new_masses = masses.copy()
    for x in range(SIZE):
        for y in range(SIZE):
            cell = grid[x, y]
            mass = masses[x, y]
            neighbors = grid[max(0, x-1):x+2, max(0, y-1):y+2]
            neighbor_masses = masses[max(0, x-1):x+2, max(0, y-1):y+2]
            
            # Fusión de estructuras del mismo tipo
            same_type = (neighbors == cell)
            if cell != SPACE and np.sum(same_type) > 1:
                total_mass = np.sum(neighbor_masses[same_type])
                new_masses[x, y] = total_mass

            # Aniquilación materia-antimateria
            if cell != SPACE and ANTIMATTER in neighbors:
                new_grid[x, y] = ENERGY
                new_masses[x, y] = MASSES[ENERGY]
                continue
            if cell == ANTIMATTER and np.any((neighbors > SPACE) & (neighbors != ANTIMATTER)):
                new_grid[x, y] = ENERGY
                new_masses[x, y] = MASSES[ENERGY]
                continue

            # Reglas para cada tipo
            if cell == SPACE:
                # Energía puede formar asteroides
                if np.count_nonzero(neighbors == ENERGY) >= 3 and np.random.rand() < 0.02:
                    new_grid[x, y] = ASTEROID
                    new_masses[x, y] = MASSES[ASTEROID]
                # Formación de asteroides por energía y otros asteroides
                if np.count_nonzero(neighbors == ENERGY) >= 2 and np.count_nonzero(neighbors == ASTEROID) >= 2 and np.random.rand() < 0.01:
                    new_grid[x, y] = ASTEROID
                    new_masses[x, y] = MASSES[ASTEROID]
                # Formación de planeta a partir de asteroides
                if np.count_nonzero(neighbors == ASTEROID) >= 4 and np.random.rand() < 0.01:
                    new_grid[x, y] = PLANET
                    new_masses[x, y] = MASSES[PLANET]
                # Formación de estrella a partir de planetas
                if np.count_nonzero(neighbors == PLANET) >= 4 and np.random.rand() < 0.01:
                    new_grid[x, y] = STAR
                    new_masses[x, y] = MASSES[STAR]
                # Formación de agujero negro por colapso estelar
                if np.count_nonzero(neighbors == STAR) >= 5 and np.random.rand() < 0.005:
                    new_grid[x, y] = BLACK_HOLE
                    new_masses[x, y] = MASSES[BLACK_HOLE]
                # Emisión de energía por estrellas
                if np.count_nonzero(neighbors == STAR) >= 2 and np.random.rand() < 0.01:
                    new_grid[x, y] = ENERGY
                    new_masses[x, y] = MASSES[ENERGY]
                # Colisión de agujeros negros
                if np.count_nonzero(neighbors == BLACK_HOLE) >= 2 and np.random.rand() < 0.01:
                    new_grid[x, y] = BLACK_HOLE
                    new_masses[x, y] = MASSES[BLACK_HOLE]

            elif cell == STAR:
                # Colapso a agujero negro si la masa es grande
                if mass > 25 and np.random.rand() < 0.01:
                    new_grid[x, y] = BLACK_HOLE
                    new_masses[x, y] = MASSES[BLACK_HOLE]
                # Las estrellas emiten energía
                if np.random.rand() < 0.02:
                    for i in range(max(0, x-1), min(SIZE, x+2)):
                        for j in range(max(0, y-1), min(SIZE, y+2)):
                            if grid[i, j] == SPACE and np.random.rand() < 0.2:
                                new_grid[i, j] = ENERGY
                                new_masses[i, j] = MASSES[ENERGY]

            elif cell == PLANET:
                # El planeta puede convertirse en estrella si la masa es grande
                if mass > 10 and np.random.rand() < 0.01:
                    new_grid[x, y] = STAR
                    new_masses[x, y] = MASSES[STAR]
                # El planeta puede fragmentarse en asteroides
                if np.random.rand() < 0.001:
                    new_grid[x, y] = ASTEROID
                    new_masses[x, y] = MASSES[ASTEROID]

            elif cell == ASTEROID:
                # El asteroide se dispersa
                if np.random.rand() < 0.002:
                    new_grid[x, y] = SPACE
                    new_masses[x, y] = 0

            elif cell == ENERGY:
                # La energía puede formar asteroides
                if np.count_nonzero(neighbors == ENERGY) >= 3 and np.random.rand() < 0.01:
                    new_grid[x, y] = ASTEROID
                    new_masses[x, y] = MASSES[ASTEROID]
                # La energía se disipa
                if np.random.rand() < 0.01:
                    new_grid[x, y] = SPACE
                    new_masses[x, y] = 0

            elif cell == BLACK_HOLE:
                # Pérdida de masa gradual
                if np.random.rand() < PROB_MASS_LOSS:
                    mass_lost = mass * MASS_LOSS_RATE
                    new_masses[x, y] -= mass_lost

                # El agujero negro absorbe objetos cercanos
                for i in range(max(0, x-1), min(SIZE, x+2)):
                    for j in range(max(0, y-1), min(SIZE, y+2)):
                        if (i != x or j != y) and grid[i, j] not in [SPACE, BLACK_HOLE]:
                            if np.random.rand() < 0.5:
                                new_grid[i, j] = SPACE
                                new_masses[i, j] = 0
                # El agujero negro puede emitir energía (radiación Hawking)
                if np.random.rand() < 0.001:
                    for i in range(max(0, x-1), min(SIZE, x+2)):
                        for j in range(max(0, y-1), min(SIZE, y+2)):
                            if grid[i, j] == SPACE and np.random.rand() < 0.2:
                                new_grid[i, j] = ENERGY
                                new_masses[i, j] = MASSES[ENERGY]

                # Desintegración final
                if new_masses[x, y] < 1:
                    # Liberar energía residual
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

# Renderiza el universo
def render(grid):
    # Convierte el grid en una imagen RGB
    img = np.zeros((SIZE, SIZE, 3))
    for t, color in COLORS.items():
        img[grid == t] = color
    return img

# Cuenta los tipos de celda
def count_cells(grid):
    # Devuelve un diccionario con el conteo de cada tipo de celda
    counts = Counter(grid.flatten())
    return [counts.get(t, 0) for t in range(7)]


# --- Interfaz gráfica y animación ---

fig = plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)

# Crear subplots
ax_title = plt.subplot2grid((12, 1), (0, 0), rowspan=1)
ax_grid = plt.subplot2grid((12, 1), (1, 0), rowspan=8)
ax_stats = plt.subplot2grid((12, 1), (9, 0), rowspan=3)

ax_title.axis('off')
ax_grid.axis('off')
ax_stats.axis('off')

# Inicialización del universo
grid = create_universe()
masses = initialize_masses(grid)
frame = [0] 

# Elementos gráficos
img = ax_grid.imshow(render(grid), interpolation='nearest')
title_text = ax_title.text(0.5, 0.5, f'Generación: {frame[0]}', 
                          fontsize=16, ha='center', va='center', 
                          transform=ax_title.transAxes)

cell_types = ["Espacio (negro)", "Estrella (amarillo)", "Planeta (verde)", "Asteroide (azul)", 
             "Energía (rojo)", "Agujero Negro (azul oscuro)", "Antimateria"]
counts = count_cells(grid)
stats_text = "\n".join([f"{name}: {count}" for name, count in zip(cell_types, counts)])

count_text = ax_stats.text(0.05, 0.7, stats_text, 
                          fontsize=12, va='top', 
                          transform=ax_stats.transAxes)

# Botón de control
btn_ax = plt.axes([0.8, 0.05, 0.15, 0.04])
btn_play = Button(btn_ax, 'Pausar', color='lightgoldenrodyellow')

# Control de animación
paused = [True]
history = []

def update_display():
    img.set_data(render(grid))
    counts = count_cells(grid)
    count_text.set_text("\n".join([f"{name}: {count}" for name, count in zip(cell_types, counts)]))
    title_text.set_text(f'Generación: {frame[0]}')
    fig.canvas.draw_idle()

def animate(i):
    if not paused[0]:
        new_grid, new_masses = update_universe(grid, masses)
        grid[:] = new_grid
        masses[:] = new_masses
        frame[0] += 1
        update_display()
    return img, count_text

def toggle_play(event):
    paused[0] = not paused[0]
    btn_play.label.set_text('Reanudar' if paused[0] else 'Pausar')

btn_play.on_clicked(toggle_play)

# Inicializar animación
ani = animation.FuncAnimation(fig, animate, interval=100, blit=False)

plt.show()