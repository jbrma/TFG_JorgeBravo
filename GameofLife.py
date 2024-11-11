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

# Estado de las celdas. Vivas = 1, Muertas = 0
gameState = np.zeros((cellsX, cellsY))

# Automata palo
gameState[5, 3] = 1
gameState[5, 4] = 1
gameState[5, 5] = 1

# Automata movil
gameState[45, 45] = 1
gameState[46, 46] = 1
gameState[46, 47] = 1
gameState[45, 47] = 1
gameState[44, 47] = 1

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
            center_x = (x + 0.5) * dimCW
            center_y = (y + 0.5) * dimCH
            radius = min(dimCW, dimCH) / 2 - 1  # Radio ajustado para que los círculos no se toquen

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