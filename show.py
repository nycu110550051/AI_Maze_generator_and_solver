import pygame, time, csv, argparse
import numpy as np
from time import sleep

if __name__ == "__main__":

  start_t0 = time.time()

  # parsing user input
  # example: python visualize.py --maze_file=maze_0.csv --algorithm=Astar
  parser = argparse.ArgumentParser()
  parser.add_argument("--algorithm", help="Implemented: Astar, bfs, dfs, <empty> (or unsolved)", default="", type=str)
  parser.add_argument("--maze_file", help="filename (csv) of the maze to load.", default="maze_1.csv", type=str)
  args = parser.parse_args()

  # check correctness of inputted arguements
  if args.algorithm == "":
    prefix = "mazes/"

  elif args.algorithm == "bfs":
    prefix = "mazes_solutions/bfs_"

  elif args.algorithm == "dfs":
    prefix = "mazes_solutions/dfs_"

  elif args.algorithm == "Astar":
    prefix = "mazes_solutions/Astar_"

  elif args.algorithm == "idAstar":
    prefix = "mazes_solutions/idAstar_"
  
  else:
      raise Exception("Not valid --algorithm parameter. (e.g Astar, bfs, dfs, idAstar, <empty>")

  address = prefix + args.maze_file
  try:
    grid = np.genfromtxt(address, delimiter=',', dtype=int)

  except:
    raise Exception(f"Maze {address} not found.")

  # define goal and start
  num_rows = len(grid)
  num_columns = len(grid[0])
  # color the start_point and goal.
  grid[-1, 0] = 2
  grid[0, -1] = 3

  # define colors of the grid RGB
  black = (0, 0, 0) # grid == 0, wall.
  white = (255, 255, 255) # grid == 1, road.
  green = (0,255,0) # grid == 2, start_point.
  red = (255,0,0) # grid == 3, goal.
  gray = (211,211,211) # for background, the region in window outside the maze.
  blue = (0,0,255) # grid == 4, where current position is
  magenta = (255,0,255) # grid == 5, solution

  # set the height/width of each location on the grid
  height = 2
  width = height
  margin = 1 # sets margin between grid locations

  # initialize pygame
  pygame.init()

  # congiguration of the window
  WINDOW_SIZE = [1500, 800]
  screen = pygame.display.set_mode(WINDOW_SIZE)

  pygame.display.set_caption(f"Visualizing: {address}")
  clock = pygame.time.Clock() # to manage how fast the screen updates

  idx_to_color = [black, white, green, red, blue, magenta]
  finish = False
        
  screen.fill(gray) # fill background in grey
  # colr the maze depends on the grid.
  for row in range(num_rows):
      for column in range(num_columns):
        color = idx_to_color[grid[row, column]]
        # the function to paint all the block in maze.
        pygame.draw.rect(screen, color, [(margin + width) * column + margin, (margin + height) * row + margin, width, height])
        
  # set limit to 60 frames per second
  clock.tick(60)
        
  # update screen
  pygame.display.flip()
        
  while not finish:
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              finish = True
  pygame.quit() # so that it doesnt "hang" on exit

  print(f"--- finished {time.time()-start_t0:.3f} s ---")
  exit(0)