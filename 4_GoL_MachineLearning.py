
'''

Todo el código junto + integración con ML


'''


import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import sys
import os


# ====== CONFIGURACIÓN ======

GRID_SIZE = 30
MAX_GENERATIONS = 100
POPULATION_SIZE = 30
CHROMOSOME_SIZE = GRID_SIZE * GRID_SIZE
MUTATION_RATE = 0.05
NUM_EVOLUTIONS = 10

# ====== PATRONES CONOCIDOS ======

KNOWN_PATTERNS = {
    "glider": [
        np.array([
            [0,1,0],
            [0,0,1],
            [1,1,1]
        ])
    ],
    "blinker": [
        np.array([[1,1,1]]),
        np.array([[1],[1],[1]])
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


# ====== FUNCIONES BÁSICAS ======

def random_individual():
    return np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE))

def game_of_life_step(grid):
    new_grid = np.zeros_like(grid)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            total = np.sum(grid[max(0,i-1):i+2, max(0,j-1):j+2]) - grid[i,j]
            if grid[i,j] == 1 and (2 <= total <= 3):
                new_grid[i,j] = 1
            elif grid[i,j] == 0 and total == 3:
                new_grid[i,j] = 1
    return new_grid

def match_pattern(grid, pattern):
    for i in range(grid.shape[0] - pattern.shape[0] + 1):
        for j in range(grid.shape[1] - pattern.shape[1] + 1):
            subgrid = grid[i:i+pattern.shape[0], j:j+pattern.shape[1]]
            if np.array_equal(subgrid, pattern):
                return True
    return False

def detect_known_patterns(grid):
    matches = []
    for name, templates in KNOWN_PATTERNS.items():
        for pat in templates:
            if match_pattern(grid, pat):
                matches.append(name)
    return matches

def detect_oscillator_period(history):
    for i in range(1, len(history)):
        if np.array_equal(history[0], history[i]):
            return i
    return None

def detect_glider_gun(history):
    if len(history) < 10:
        return False
    last = history[-1]
    glider_detected = False
    for pat in KNOWN_PATTERNS["glider"]:
        if match_pattern(last, pat):
            glider_detected = True
    center_start = history[0][5:15, 5:15]
    center_end = history[-1][5:15, 5:15]
    static_center = np.sum(np.abs(center_start - center_end)) < 5
    return static_center and glider_detected

def fitness(individual):
    grid = individual.copy()
    history = []
    movement_score = 0
    oscillation_score = 0
    last_com = None

    for _ in range(MAX_GENERATIONS):
        grid = game_of_life_step(grid)
        history.append(grid.copy())

        y, x = np.nonzero(grid)
        if len(x) > 0:
            com = (np.mean(y), np.mean(x))
            if last_com:
                dy = com[0] - last_com[0]
                dx = com[1] - last_com[1]
                movement_score += np.sqrt(dx**2 + dy**2)
            last_com = com

        for prev in history[:-1]:
            if np.array_equal(prev, grid):
                oscillation_score += 1
                break

    alive_cells = np.sum(grid)
    score = alive_cells + movement_score * 2 + oscillation_score * 5

    period = detect_oscillator_period(history)
    if period and period > 2:
        score += 50 + period * 2

    if detect_glider_gun(history):
        score += 300

    detected = detect_known_patterns(grid)
    if "glider_gun" in detected:
        score += 1000
    if "glider" in detected:
        score += 500
    if "beehive" in detected:
        score += 200
    if "blinker" in detected:
        score -= 30
    if "block" in detected:
        score -= 30

    return score


def predict_pattern(grid, model, label_mapping, threshold=0.2):
    feat = extract_features(grid).reshape(1, -1)
    raw_preds = model.predict(feat, verbose=0)[0]
    
    # Filtrar por umbral y obtener probabilidades
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    filtered = {reverse_mapping[idx]: float(prob) 
                for idx, prob in enumerate(raw_preds) if prob >= threshold}
    
    # Calcular porcentajes relativos si hay detecciones
    if filtered:
        total = sum(filtered.values())
        return {k: round(v / total, 2) for k, v in filtered.items()}
    else:
        return {"random": 1.0}  # Caso sin patrones detectados




# ====== ALGORITMO GENÉTICO ======

def crossover(parent1, parent2):
    point = random.randint(0, CHROMOSOME_SIZE - 1)
    flat1 = parent1.flatten()
    flat2 = parent2.flatten()
    child_flat = np.concatenate((flat1[:point], flat2[point:]))
    return child_flat.reshape((GRID_SIZE, GRID_SIZE))

def mutate(individual):
    mutation_mask = np.random.rand(GRID_SIZE, GRID_SIZE) < MUTATION_RATE
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual


# ====== EJECUTAR EVOLUCIÓN ======

population = [random_individual() for _ in range(POPULATION_SIZE)]
population_history = []
fitness_history = []
best_individual = None
best_fitness = 0

for gen in range(NUM_EVOLUTIONS):
    scored_population = [(fitness(ind), ind) for ind in population]
    scored_population.sort(reverse=True, key=lambda x: x[0])

    fitness_values = [f for f, _ in scored_population]
    fitness_history.append((np.mean(fitness_values), np.max(fitness_values)))
    population_history.append([ind.copy() for _, ind in scored_population])

    print(f" Generación {gen}/{NUM_EVOLUTIONS - 1} -- fitness promedio: {fitness_history[-1][0]:.2f}, fitness máximo: {fitness_history[-1][1]:.2f}")

    if scored_population[0][0] > best_fitness:
        best_fitness = scored_population[0][0]
        best_individual = scored_population[0][1]

    selected = [ind for _, ind in scored_population[:POPULATION_SIZE // 2]]
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)
    population = new_population


# ====== HISTORIA DEL MEJOR INDIVIDUO ======

best_history = [best_individual.copy()]
grid = best_individual.copy()
for _ in range(50):
    grid = game_of_life_step(grid)
    best_history.append(grid.copy())


# ====== GUI UNIFICADA ======

class LifeGAApp:
    def __init__(self, root, model, label_mappin):
        self.model = model
        self.label_mapping = label_mapping
        self.root = root
        self.bar_ax = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.title("Algoritmo Genético + Juego de la Vida + Red Neuronal")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=20, pady=20)

        self.create_best_tab()
        self.create_population_tab()
        self.create_fitness_tab()

    def create_best_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Mejor Patrón")

        self.index = 0
        self.fig = plt.figure(figsize=(10, 10))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        self.ax = self.fig.add_subplot(gs[0])
        self.bar_ax = self.fig.add_subplot(gs[1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack()

        self.status = tk.Label(frame, text="Generación 0")
        self.status.pack()

        LARGE_FONT = ('Arial', 18)
        BUTTON_STYLE = {'padx': 15, 'pady': 10}

        controls = tk.Frame(frame)
        controls.pack()
        tk.Button(controls, text="⏮️", command=self.prev_frame, font=LARGE_FONT, **BUTTON_STYLE).pack(side=tk.LEFT)
        tk.Button(controls, text="▶️", command=self.play, font=LARGE_FONT, **BUTTON_STYLE).pack(side=tk.LEFT)
        tk.Button(controls, text="⏭️", command=self.next_frame, font=LARGE_FONT, **BUTTON_STYLE).pack(side=tk.LEFT)
        
        style = ttk.Style()
        style.configure('TNotebook.Tab', font=LARGE_FONT, padding=[16, 10])

        info_frame = ttk.Frame(frame)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        label_style = {'font': ('Arial', 12, 'bold'), 'padding': 10}

        self.lbl_cells = ttk.Label(info_frame, text="Células vivas: 0", **label_style)
        self.lbl_patterns = ttk.Label(info_frame, text="Patrones detectados: ", **label_style)
        self.lbl_period = ttk.Label(info_frame, text="Período oscilación: ", **label_style)
        self.lbl_ml = ttk.Label(info_frame, text="Predicción ML: ", **label_style)
        
        self.lbl_ml.pack(side=tk.BOTTOM, padx=25)
        self.lbl_period.pack(side=tk.BOTTOM, padx=25)
        self.lbl_patterns.pack(side=tk.BOTTOM, padx=25)
        self.lbl_cells.pack(side=tk.BOTTOM, padx=25)

        self.plot_frame(0)


    def plot_frame(self, idx):
        self.ax.clear()
        grid = best_history[idx]

        # Actualizar métricas
        alive = np.sum(grid)
        patterns = detect_known_patterns(grid)
        period = detect_oscillator_period(best_history[:idx+1]) if idx > 0 else 0
        
        # Predicción con ML
        features = extract_features(grid).reshape(1, -1)
        predictions = predict_pattern(grid, self.model, self.label_mapping, threshold=0.2)

        # Actualizar labels
        self.lbl_cells.config(text=f"Células vivas: {alive}")
        self.lbl_patterns.config(text=f"Patrones detectados: {', '.join(patterns) if patterns else 'Ninguno'}")
        self.lbl_period.config(text=f"Período oscilación: {period if period else 'No detectado'}")
        pred_text = ", ".join([f"{k} ({v*100:.0f}%)" for k, v in predictions.items()])
        self.lbl_ml.config(text=f"Patrones: {pred_text}")

        self.ax.imshow(grid, cmap='Greys', interpolation='nearest')
        
        # Resaltar patrones detectados
        for name, templates in KNOWN_PATTERNS.items():
            for pat in templates:
                for i in range(grid.shape[0] - pat.shape[0] + 1):
                    for j in range(grid.shape[1] - pat.shape[1] + 1):
                        if np.array_equal(grid[i:i+pat.shape[0], j:j+pat.shape[1]], pat):
                            rect = plt.Rectangle((j-0.5, i-0.5), pat.shape[1], pat.shape[0],
                                                linewidth=2, edgecolor='red', facecolor='none')
                            self.ax.add_patch(rect)
        
        self.ax.axis('off')
        self.ax.set_title(f"Generación {idx}")
        self.canvas.draw()
        self.index = idx
        self.status.config(text=f"Generación {idx}")

        self.bar_ax.clear()
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, v in sorted_preds]
        values = [v for k, v in sorted_preds]
        
        bars = self.bar_ax.barh(labels, values, color='#9af156')
        self.bar_ax.set_xlim(0, 1)
        self.bar_ax.set_title("Probabilidades de patrones (Red Neuronal)")
        self.bar_ax.bar_label(bars, fmt='%.2f', padding=5)
        self.bar_ax.invert_yaxis()
        
        self.canvas.draw()

    def prev_frame(self):
        if self.index > 0:
            self.plot_frame(self.index - 1)

    def next_frame(self):
        if self.index < len(best_history) - 1:
            self.plot_frame(self.index + 1)

    def play(self):
        if not hasattr(self, 'playing'):
            self.playing = False
            
        if not self.playing:
            self.playing = True
            self.root.after(0, self.animate)
        else:
            self.playing = False

    def animate(self):
        if self.playing and self.index < len(best_history) - 1:
            self.next_frame()
            self.root.after(300, self.animate)
            
    def create_population_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Población Final")

        fig, axes = plt.subplots(6, 5, figsize=(14, 14))
        
        for i, ax in enumerate(axes.flat):
            if i < len(population_history[-1]):
                ax.imshow(population_history[-1][i], cmap='Greys')
            ax.axis('off')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack()

    def create_fitness_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Fitness")

        avg = [f[0] for f in fitness_history]
        max_ = [f[1] for f in fitness_history]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(avg, label="Promedio", linewidth=2.5, color='blue')
        ax.plot(max_, label="Máximo", linewidth=2.5, color='green', linestyle='--')

        if max_:
            y_max = max(max_) * 1.1  # 10% de margen
            ax.set_ylim(0, y_max)

        ax.set_title("Evolución del Fitness", fontsize=14, pad=20)
        ax.set_xlabel("Generación", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack()


    def on_close(self):
        """Maneja el cierre seguro de la aplicación"""
        self.playing = False  
        plt.close('all')      
        self.root.destroy()   
        if sys.platform.startswith('win'):
            os._exit(0)       
        else:
            sys.exit(0)

# ====== LANZAR APP ======

# root = tk.Tk()
# root.geometry("1200x1400")
# app = LifeGAApp(root)
# root.mainloop()


# ====== MODULO ML =======

import numpy as np
from scipy.ndimage import label

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# ====== Feature extraction ======

def extract_features(grid):
    alive = np.sum(grid)
    y, x = np.nonzero(grid)
    com_x = np.mean(x) if len(x) > 0 else 0
    com_y = np.mean(y) if len(y) > 0 else 0
    density = alive / (grid.shape[0] * grid.shape[1])

    variance_x = np.var(x) if len(x) > 0 else 0
    variance_y = np.var(y) if len(y) > 0 else 0
    symmetry_h = np.sum(grid == np.fliplr(grid))/grid.size
    symmetry_v = np.sum(grid == np.flipud(grid))/grid.size

    labeled, num_components = label(grid)
    
    return np.array([alive, com_x, com_y, density, 
            variance_x, variance_y, 
            symmetry_h, symmetry_v, 
            num_components], dtype=np.float32)


# ====== Dataset creation ======

def build_dataset(population_history, steps=20):
    X, y = [], []
    all_labels = set()

    for gen in population_history:
        for individual in gen:
            grid = individual.copy()
            history = [grid.copy()]
            for _ in range(steps):
                grid = game_of_life_step(grid)
                history.append(grid.copy())

            features = extract_features(individual)

            detected_patterns = set()

            for grid_step in history[-1:]:  # mira el ultimo paso
                detected_patterns.update(detect_known_patterns(grid_step))

            # Etiquetado heurístico
            if detect_glider_gun(history):
                detected_patterns.add("glider_gun")

            osc = detect_oscillator_period(history)
            if osc:
                detected_patterns.add("oscillator")

            if np.sum(history[-1]) == 0:
                detected_patterns.add("extinct")

            if not detected_patterns:
                detected_patterns.add("random")

            label = sorted(list(detected_patterns))
            all_labels.update(label)

            X.append(features)
            y.append(list(detected_patterns))

    # Convertir a matriz binaria (one-hot encoding multi-etiqueta)
    label_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
    y_binary = np.zeros((len(y), len(label_mapping)), dtype=int)
    for i, labels in enumerate(y):
        for label in labels:
            y_binary[i, label_mapping[label]] = 1

    return np.array(X), y_binary, label_mapping


# ====== Entrenar modelo ======


def train_model(X, y_binary):
    
    # Modelo de red neuronal
    model = Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(y_binary.shape[1], activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X, y_binary,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    return model


# === Conectar con el módulo ML ===

X, y_binary, label_mapping = build_dataset(population_history)

label_counts = {label: np.sum(y_binary[:, idx]) for label, idx in label_mapping.items()}
print("Conteo de patrones:", label_counts)

model = train_model(X, y_binary)

root = tk.Tk()
app = LifeGAApp(root, model, label_mapping)
root.mainloop()


