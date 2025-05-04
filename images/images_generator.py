import pygame
import numpy as np
import random
import os
from datetime import datetime

WIDTH, HEIGHT = 1200, 675
CELL_SIZE = 12
GRID_WIDTH, GRID_HEIGHT = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE

COLOR_BG = (20, 30, 50)     
COLOR_GRID = (20, 30, 50)   
COLOR_CELL = (255, 255, 255) 
COLOR_TEXT = (200, 200, 200) 

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game of Life - Patrones Aleatorios")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)

# Diccionario de patrones conocidos
PATTERNS = {
    # Still Lifes
    #"block": [(0,0), (0,1), (1,0), (1,1)],
    #"beehive": [(0,1), (1,0), (1,2), (2,0), (2,2), (3,1)],
    #"loaf": [(0,1), (1,0), (1,2), (2,0), (2,3), (3,1), (3,2)],
    
    # Oscillators
    #"blinker": [(0,0), (0,1), (0,2)],
    #"toad": [(0,1), (0,2), (0,3), (1,0), (1,1), (1,2)],
    
    # Spaceships
    #"glider": [(0,1), (1,2), (2,0), (2,1), (2,2)],
    #"lwss": [(0,0), (0,3), (1,4), (2,0), (2,4), (3,1), (3,2), (3,3)],
    
    # Guns
    "gosper_gun": [
        (0,4), (0,5), (1,4), (1,5),
        (10,4), (10,5), (10,6), (11,3), (11,7),
        (12,2), (12,8), (13,2), (13,8), (14,5),
        (15,3), (15,7), (16,4), (16,5), (16,6),
        (17,5),
        (20,2), (20,3), (20,4), (21,2), (21,3), (21,4),
        (22,1), (22,5), (24,0), (24,1), (24,5), (24,6),
        (34,2), (34,3), (35,2), (35,3)
    ]
}

def place_pattern(universe, pattern_name, x_offset, y_offset):
    """Coloca un patrón en la posición especificada"""
    pattern = PATTERNS[pattern_name]
    for dx, dy in pattern:
        x, y = x_offset + dx, y_offset + dy
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            universe[x, y] = 1

def initialize_patterns():
    """Inicializa el universo con patrones aleatorios"""
    universe = np.zeros((GRID_WIDTH, GRID_HEIGHT))
    
    num_patterns = random.randint(1, 1)
    all_patterns = list(PATTERNS.keys())
    
    for _ in range(num_patterns):
        pattern = random.choice(all_patterns)
        x = random.randint(5, GRID_WIDTH - 10)
        y = random.randint(5, GRID_HEIGHT - 10)
        place_pattern(universe, pattern, x, y)
    
    return universe

def draw_universe(universe, show_info=True):
    """Dibuja el universo, con o sin información según parámetro"""
    screen.fill(COLOR_BG)
    
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            if universe[x, y] == 1:
                pygame.draw.rect(screen, COLOR_CELL, 
                               (x * CELL_SIZE, y * CELL_SIZE, 
                                CELL_SIZE - 1, CELL_SIZE - 1))
    if show_info:
        status = "PAUSADO" if paused else "EJECUTANDO"
        status_text = font.render(f"Estado: {status} | Generación: {generation} | Velocidad: {speed}x", True, COLOR_TEXT)
        controls_text = font.render("Controles: ESPACIO=Pausa | R=Reiniciar | C=Limpiar | S=Guardar Imagen | Ratón=Dibujar/Borrar", True, COLOR_TEXT)
        
        screen.blit(status_text, (10, 10))
        screen.blit(controls_text, (10, 30))

def save_clean_image(universe):
    """Guarda una imagen limpia del universo actual"""
    os.makedirs("images", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_patterns/pattern_{timestamp}.png"
    
    temp_surface = pygame.Surface((WIDTH, HEIGHT))
    temp_surface.fill(COLOR_BG)
    
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            if universe[x, y] == 1:
                pygame.draw.rect(temp_surface, COLOR_CELL, 
                               (x * CELL_SIZE, y * CELL_SIZE, 
                                CELL_SIZE - 1, CELL_SIZE - 1))
    
    pygame.image.save(temp_surface, filename)
    print(f"Imagen guardada como {filename}")

def next_generation(universe):
    new_universe = np.zeros_like(universe)
    
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            neighbors = (
                universe[(x-1) % GRID_WIDTH, (y-1) % GRID_HEIGHT] +
                universe[(x)   % GRID_WIDTH, (y-1) % GRID_HEIGHT] +
                universe[(x+1) % GRID_WIDTH, (y-1) % GRID_HEIGHT] +
                universe[(x-1) % GRID_WIDTH, (y)   % GRID_HEIGHT] +
                universe[(x+1) % GRID_WIDTH, (y)   % GRID_HEIGHT] +
                universe[(x-1) % GRID_WIDTH, (y+1) % GRID_HEIGHT] +
                universe[(x)   % GRID_WIDTH, (y+1) % GRID_HEIGHT] +
                universe[(x+1) % GRID_WIDTH, (y+1) % GRID_HEIGHT]
            )
            
            if universe[x, y] == 1:
                new_universe[x, y] = 1 if neighbors in [2, 3] else 0
            else:
                new_universe[x, y] = 1 if neighbors == 3 else 0
    
    return new_universe

def main():
    global paused, generation, speed
    
    running = True
    paused = True
    drawing = False
    erasing = False
    universe = initialize_patterns()
    generation = 0
    speed = 5
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    universe = initialize_patterns()
                    generation = 0
                elif event.key == pygame.K_c:
                    universe = np.zeros((GRID_WIDTH, GRID_HEIGHT))
                    generation = 0
                elif event.key == pygame.K_s:
                    save_clean_image(universe)
                elif event.key == pygame.K_UP:
                    speed = min(20, speed + 1)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
                    x, y = pygame.mouse.get_pos()
                    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                        universe[grid_x, grid_y] = 1 - universe[grid_x, grid_y]
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
        
        if drawing:
            x, y = pygame.mouse.get_pos()
            grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                universe[grid_x, grid_y] = 1 - universe[grid_x, grid_y]
        
        if not paused and pygame.time.get_ticks() % (1000 // speed) == 0:
            universe = next_generation(universe)
            generation += 1
        
        draw_universe(universe, show_info=True)
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()