import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.ndimage import label
from sklearn.cluster import KMeans
from tensorflow.keras import layers, Model, backend as K

# Parámetros del universo
SIZE = 200
PROB_MASS_LOSS = 0.15
MASS_LOSS_RATE = 0.12

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
    BLACK_HOLE: 50,
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



# ===== RED NEURONAL =====

class DataCollector:
    def __init__(self, max_samples=500):
        self.images = []
        self.stats = []
        self.max_samples = max_samples
    
    def collect_sample(self, grid, masses):
        if len(self.images) >= self.max_samples:
            return False
        self.images.append(render(grid))
        self.stats.append(self._extract_statistics(grid, masses))
        return True
    
    def _extract_statistics(self, grid, masses):
        counts = [np.sum(grid == i) for i in range(7)]
        regions = 5
        region_size = max(1, SIZE // regions)
        region_stats = []
        for i in range(regions):
            for j in range(regions):
                region = grid[i*region_size:min(SIZE,(i+1)*region_size), 
                             j*region_size:min(SIZE,(j+1)*region_size)]
                region_counts = [np.sum(region == k) for k in range(7)]
                region_stats.extend(region_counts)
        return np.concatenate([counts, region_stats])

def build_autoencoder(input_shape=(20, 20, 3), latent_dim=64):
    input_img = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    
    shape_before_flatten = K.int_shape(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(np.prod(shape_before_flatten[1:]), activation='relu')(encoded)
    x = layers.Reshape(shape_before_flatten[1:])(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    return autoencoder, encoder

class PatternDetector:
    def __init__(self, latent_dim=64, n_clusters=8):
        self.autoencoder, self.encoder = build_autoencoder(
            input_shape=(SIZE, SIZE, 3), 
            latent_dim=latent_dim
        )
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.pattern_counts = {
            'Sistemas Estelares': 0,
            'Regiones Energía': 0,
            'Clusters Agujeros': 0,
            'Vacíos Cósmicos': 0
        }
        
    def train(self, images, stats):
        images_array = np.array(images)
        print(f"Entrenando con {len(images_array)} muestras")
        self.autoencoder.fit(images_array, images_array, epochs=10, batch_size=8, validation_split=0.2, verbose=1)
        latent_vectors = self.encoder.predict(images_array, verbose=0)
        stats_array = np.array(stats)
        combined_features = np.concatenate([latent_vectors, stats_array], axis=1)
        self.kmeans.fit(combined_features)
        print("Entrenamiento completo")
        
    def detect_patterns(self, grid):
        self._update_pattern_counts(grid)
        return self.pattern_counts.copy()
    
    def _extract_statistics(self, grid, masses):
        return DataCollector()._extract_statistics(grid, masses)
    
    def _update_pattern_counts(self, grid):
        # Sistemas estelares (estrella con al menos 1 planeta/asteroide en radio 2x2)
        self.pattern_counts['Sistemas Estelares'] = 0
        stars = np.argwhere(grid == STAR)
        for x, y in stars:
            area = grid[max(0,x-1):min(SIZE,x+2), max(0,y-1):min(SIZE,y+2)]
            if np.sum(np.isin(area, [PLANET, ASTEROID])) >= 1:
                self.pattern_counts['Sistemas Estelares'] += 1
                
        # Regiones de alta energía (mínimo 2 celdas contiguas)
        energy_mask = grid == ENERGY
        labeled, num_features = label(energy_mask)
        self.pattern_counts['Regiones Energía'] = 0
        for i in range(1, num_features+1):
            if np.sum(labeled == i) >= 2: 
                self.pattern_counts['Regiones Energía'] += 1
                
        # Clusters de agujeros negros (2+ en el universo)
        black_holes = np.argwhere(grid == BLACK_HOLE)
        self.pattern_counts['Clusters Agujeros'] = len(black_holes) // 2
            
        # Vacíos cósmicos (área 3x3 con 80% espacio vacío)
        self.pattern_counts['Vacíos Cósmicos'] = 0
        for i in range(0, SIZE-2, 2):
            for j in range(0, SIZE-2, 2):
                window = grid[i:i+3, j:j+3]
                if np.sum(window == SPACE) >= 7:  # 9 celdas * 0.8 = 7.2
                    self.pattern_counts['Vacíos Cósmicos'] += 1     

# Configuración de la interfaz
fig = plt.figure(figsize=(7, 9))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)
ax = plt.gca()
ax.axis('off')

# Variables globales
grid = create_universe()
masses = initialize_masses(grid)
frame = [0]
paused = [True]
detector = PatternDetector()
data_collector = DataCollector(max_samples=50)
training_done = [False]

# Crear tabla de estadísticas
stats_ax = plt.axes([0.1, 0.05, 0.8, 0.15])
stats_ax.axis('off')
table_data = [
    ["Generación", "0"],
    ["Sistemas Estelares", "0"],
    ["Regiones Energía", "0"],
    ["Clusters Agujeros", "0"],
    ["Vacíos Cósmicos", "0"]
]
table = stats_ax.table(
    cellText=table_data,
    colWidths=[0.3, 0.3],
    loc='center',
    cellLoc='center',
    edges='closed'
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Elementos de la interfaz
img = ax.imshow(render(grid), interpolation='nearest')

# Botón de play/pause
btn_ax = plt.axes([0.45, 0.22, 0.1, 0.04])
btn_play = Button(btn_ax, 'Iniciar', color='lightgoldenrodyellow')


def update_display(counts):
    img.set_data(render(grid))
    
    # Actualizar datos de la tabla
    table.get_celld()[(0,1)].get_text().set_text(f"{frame[0]}")
    table.get_celld()[(1,1)].get_text().set_text(f"{counts['Sistemas Estelares']}")
    table.get_celld()[(2,1)].get_text().set_text(f"{counts['Regiones Energía']}")
    table.get_celld()[(3,1)].get_text().set_text(f"{counts['Clusters Agujeros']}")
    table.get_celld()[(4,1)].get_text().set_text(f"{counts['Vacíos Cósmicos']}")
    
    fig.canvas.draw()

def animate(i):
    global grid, masses, detector, training_done
    if not paused[0]:
        grid, masses = update_universe(grid, masses)
        frame[0] += 1
        
        if not training_done[0]:
            if data_collector.collect_sample(grid, masses):
                if len(data_collector.images) >= 50:  # Reducido a 50 muestras
                    detector.train(data_collector.images, data_collector.stats)
                    training_done[0] = True
        
        counts = detector.detect_patterns(grid)
        update_display(counts)
    
    return img,

def toggle_play(event):
    paused[0] = not paused[0]
    btn_play.label.set_text('Pausar' if not paused[0] else 'Reanudar')

btn_play.on_clicked(toggle_play)
ani = animation.FuncAnimation(fig, animate, interval=100, blit=False)
plt.show()
