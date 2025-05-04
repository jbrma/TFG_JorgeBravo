import pygame
import numpy as np
import time

pygame.init()

width, height = 800, 800
screen = pygame.display.set_mode((height, width))

color_bg = 25, 25, 25 # background

screen.fill(color_bg)

cellsX, cellsY = 80, 80 # numero de celdas

dimCW = width / cellsX
dimCH = height / cellsY

# Estado de las celulas. Vivas = 1, Muertas = 0
gameState = np.zeros((cellsX, cellsY))

# Block
gameState[5, 20] = 1
gameState[5, 21] = 1
gameState[6, 20] = 1
gameState[6, 21] = 1

# Blinker
gameState[5, 3] = 1
gameState[5, 4] = 1
gameState[5, 5] = 1

# Beehive
gameState[10, 25] = 1
gameState[10, 26] = 1
gameState[11, 24] = 1
gameState[11, 27] = 1
gameState[12, 25] = 1
gameState[12, 26] = 1

# Glider
gameState[45, 45] = 1
gameState[46, 46] = 1
gameState[46, 47] = 1
gameState[45, 47] = 1
gameState[44, 47] = 1

# Gosper Glider Gun
ggg_x_offset = 58 
ggg_y_offset = 8

gosper_gun = [
    (5,1), (5,2), (6,1), (6,2),
    (5,11), (6,11), (7,11),
    (4,12), (8,12),
    (3,13), (9,13),
    (3,14), (9,14),
    (6,15),
    (4,16), (8,16),
    (5,17), (6,17), (7,17),
    (6,18),
    (3,21), (4,21), (5,21),
    (3,22), (4,22), (5,22),
    (2,23), (6,23),
    (1,25), (2,25), (6,25), (7,25),
    (3,35), (4,35),
    (3,36), (4,36)
]

for x, y in gosper_gun:
    gameState[ggg_x_offset + x, ggg_y_offset + y] = 1

#Toad
gameState[10, 10] = 1
gameState[10, 11] = 1
gameState[10, 12] = 1
gameState[11, 9] = 1
gameState[11, 10] = 1
gameState[11, 11] = 1

#Beacon
gameState[20, 20] = 1
gameState[20, 21] = 1
gameState[21, 20] = 1
gameState[22, 23] = 1
gameState[23, 22] = 1
gameState[23, 23] = 1

#Pulsar
x0, y0 = 30, 30
for dx in [2, 3, 4, 8, 9, 10]:
    for dy in [0, 5, 7, 12]:
        gameState[x0+dx, y0+dy] = 1
        gameState[x0+dy, y0+dx] = 1

#Diehard
gameState[10, 50] = 1
gameState[11, 44] = 1
gameState[11, 45] = 1
gameState[12, 45] = 1
gameState[12, 49] = 1
gameState[12, 50] = 1
gameState[12, 51] = 1


# Control
pause = False

running = True
while running:

    newGameState = np.copy(gameState) 

    screen.fill(color_bg)
    time.sleep(0.1)

    for event in pygame.event.get():
        
        if event.type == pygame.KEYDOWN:
            pause = not pause
            
        if event.type == pygame.QUIT:
            running = False
        
        click = pygame.mouse.get_pressed()

        if sum(click) > 0:
            posX, posY = pygame.mouse.get_pos()
            celX, celY = int(np.floor(posX / dimCW)), int(np.floor(posY / dimCH))
            newGameState[celX, celY] = not gameState[celX, celY]

    for y in range(0, cellsX):
        for x in range(0, cellsY):

            if not pause:
                
                num_neigh = gameState[(x-1) % cellsX, (y-1) % cellsY] + \
                            gameState[(x)   % cellsX, (y-1) % cellsY] + \
                            gameState[(x+1) % cellsX, (y-1) % cellsY] + \
                            gameState[(x-1) % cellsX, (y)   % cellsY] + \
                            gameState[(x+1) % cellsX, (y)   % cellsY] + \
                            gameState[(x-1) % cellsX, (y+1) % cellsY] + \
                            gameState[(x)   % cellsX, (y+1) % cellsY] + \
                            gameState[(x+1) % cellsX, (y+1) % cellsY]
                            
                # Regla #1 -> Nacimiento: Si una celula muerta tiene exactamente 3 vecinas vivas, nace.
                if gameState[x, y] == 0 and num_neigh == 3:
                    newGameState[x, y] = 1


                # Regla #2 -> Soledad: Si una celula viva tiene menos de 2 vecinas, muere.
                # Regla #3 -> Sobrepoblación: Si tiene más de 3 vecinas, muere.
                elif gameState[x, y] == 1 and (num_neigh < 2 or num_neigh > 3):
                    newGameState[x, y] = 0


            # Círculo
            center_x = (x + 0.5) * dimCW
            center_y = (y + 0.5) * dimCH
            radius = min(dimCW, dimCH) / 2 - 1 

            if newGameState[x,y] == 0:
                pygame.draw.circle(screen, (24, 24, 24), (int(center_x), int(center_y)), int(radius), width=1)
            else:
                pygame.draw.circle(screen, (255, 255, 255), (int(center_x), int(center_y)), int(radius), width=0)

            # Cuadrado 
            """ poly = [(x * dimCW, y * dimCH),
                    ((x+1) * dimCW, y * dimCH),
                    ((x+1) * dimCW, (y+1) * dimCH),
                    (x * dimCW, (y+1) * dimCH)]
            
            if newGameState[x,y] == 0:
                pygame.draw.polygon(screen, (128, 128, 128), poly, width=1)
            else:
                pygame.draw.polygon(screen, (128, 128, 128), poly, width=0) """

    gameState = np.copy(newGameState)

    pygame.display.flip()

pygame.quit()