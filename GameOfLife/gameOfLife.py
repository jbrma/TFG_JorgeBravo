import pygame
import numpy as np
import time
import genetic_algorithm

cellsX, cellsY = 50, 50 # numero de celdas

print("Ejecutando el algoritmo genético...")
genetic_board = genetic_algorithm.run_genetic_algorithm(cellsX, cellsY)
print("Tablero generado. Comenzando la simulación...")

# Initial configuration
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((height, width))
pygame.display.set_caption("Conway's Game of Life")

color_bg = 25, 25, 25 # background
screen.fill(color_bg)

dimCW = width / cellsX
dimCH = height / cellsY

gameState = np.copy(genetic_board)


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

        if sum(click) > 0: # Detects a click and get its position
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
                            
                # Rule #1 -> Generation: If a dead cell has exactly 3 living neighbouring cells, it borns
                if gameState[x, y] == 0 and num_neigh == 3:
                    newGameState[x, y] = 1


                # Rule #2 -> Isolation:       If a living cell has only one or no neighbours around, it dies
                #            Overpopulation:  If it has more than 3 neighbours around it, it dies.
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