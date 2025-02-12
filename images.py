import pygame
import numpy as np
import time
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
color_bg = 75, 75, 75 # background

screen.fill(color_bg)
pygame.display.flip()

cellsX, cellsY = 120, 120 # numero de celdas
dimCW = width / cellsX
dimCH = height / cellsY

cell_margin = 0.35

# Estado de las celdas. Vivas = 1, Muertas = 0
gameState = np.zeros((cellsX, cellsY))

# 🔹 STILL LIFES (estructuras estáticas)
# Block
gameState[5, 5] = gameState[5, 6] = gameState[6, 5] = gameState[6, 6] = 1
# Beehive
gameState[10, 10] = gameState[11, 9] = gameState[11, 11] = gameState[12, 9] = gameState[12, 11] = gameState[13, 10] = 1
# Loaf
gameState[15, 15] = gameState[16, 14] = gameState[16, 16] = gameState[17, 14] = gameState[17, 17] = gameState[18, 15] = gameState[17, 16] = 1

# 🔹 OSCILADORES
# Blinker
gameState[25, 25] = gameState[25, 26] = gameState[25, 27] = 1
# Toad
gameState[30, 30] = gameState[30, 31] = gameState[30, 32] = gameState[31, 29] = gameState[31, 30] = gameState[31, 31] = 1
# Beacon
gameState[40, 40] = gameState[40, 41] = gameState[41, 40] = gameState[42, 43] = gameState[43, 42] = gameState[43, 43] = 1

# 🔹 SPACESHIPS
# Glider
gameState[50, 50] = gameState[51, 51] = gameState[51, 52] = gameState[50, 52] = gameState[49, 52] = 1
# Lightweight Spaceship (LWSS)
gameState[60, 60] = gameState[60, 63] = gameState[61, 64] = gameState[62, 60] = gameState[62, 64] = gameState[63, 61] = gameState[63, 62] = gameState[63, 63] = 1

# 🔹 GOSPER GLIDER GUN (Cañón de planeadores)
glider_gun = [
    (1, 5), (1, 6), (2, 5), (2, 6),
    (11, 5), (11, 6), (11, 7), (12, 4), (12, 8),
    (13, 3), (13, 9), (14, 3), (14, 9), (15, 6),
    (16, 4), (16, 8), (17, 5), (17, 6), (17, 7),
    (18, 6),
    (21, 3), (21, 4), (21, 5), (22, 3), (22, 4), (22, 5),
    (23, 2), (23, 6), (25, 1), (25, 2), (25, 6), (25, 7),
    (35, 3), (35, 4), (36, 3), (36, 4)
]

# Control
pause = False
running = True
while running:
    newGameState = np.copy(gameState) # Updated each iteration
    screen.fill(color_bg) # Clean the screen
    time.sleep(0.1)
    for event in pygame.event.get():
        
        if event.type == pygame.KEYDOWN: # Pause the game
            pause = not pause
            
        if event.type == pygame.QUIT: # Exit the game
            running = False
        
        click = pygame.mouse.get_pressed()
        if sum(click) > 0:
            posX, posY = pygame.mouse.get_pos()
            celX, celY = int(np.floor(posX / dimCW)), int(np.floor(posY / dimCH))
            newGameState[celX, celY] = not gameState[celX, celY]
    for y in range(0, cellsX):
        for x in range(0, cellsY):
            if not pause:
                # Calculate number of neighbours around it
                num_neigh = gameState[(x-1) % cellsX, (y-1) % cellsY] + \
                            gameState[(x)   % cellsX, (y-1) % cellsY] + \
                            gameState[(x+1) % cellsX, (y-1) % cellsY] + \
                            gameState[(x-1) % cellsX, (y)   % cellsY] + \
                            gameState[(x+1) % cellsX, (y)   % cellsY] + \
                            gameState[(x-1) % cellsX, (y+1) % cellsY] + \
                            gameState[(x)   % cellsX, (y+1) % cellsY] + \
                            gameState[(x+1) % cellsX, (y+1) % cellsY]
                            
                # Rule #1 -> Born: If a dead cell has exactly 3 living neighbouring cells, it borns
                if gameState[x, y] == 0 and num_neigh == 3:
                    newGameState[x, y] = 1
                # Rule #2 -> Isolation:       If a living cell has only one or no neighbours around, it dies
                #            Overpopulation:  If it has more than 3 neighbours around it, it dies.
                elif gameState[x, y] == 1 and (num_neigh < 2 or num_neigh > 3):
                    newGameState[x, y] = 0
            # Círculo 
            """
            center_x = (x + 0.5) * dimCW
            center_y = (y + 0.5) * dimCH
            radius = min(dimCW, dimCH) / 2 - 1  # Radio ajustado para que los círculos no se toquen
            if newGameState[x,y] == 0:
                pygame.draw.circle(screen, (24, 24, 24), (int(center_x), int(center_y)), int(radius), width=1)
            else:
                pygame.draw.circle(screen, (255, 255, 255), (int(center_x), int(center_y)), int(radius), width=0) 
            """
            # Cuadrado 
            poly = [(x * dimCW, y * dimCH),
                    ((x+1) * dimCW, y * dimCH),
                    ((x+1) * dimCW, (y+1) * dimCH),
                    (x * dimCW, (y+1) * dimCH)]
            
            if newGameState[x,y] == 1:
                rect_x = x * dimCW + cell_margin
                rect_y = y * dimCH + cell_margin
                rect_width = dimCW - cell_margin * 2
                rect_height = dimCH - cell_margin * 2
                pygame.draw.rect(screen, (255, 255, 255), (rect_x, rect_y, rect_width, rect_height))
        
    gameState = np.copy(newGameState)
    pygame.display.flip()
pygame.quit()