import pygame, argparse, csv, time
import argparse
import numpy as np
from time import sleep
from numpy.random import randint

start_point = tuple()

def is_in_map(pos, grid_dim):

    # append the value of max_x ,max_y
    (max_x, max_y) = grid_dim     
    (x, y) = pos

    # check whether x,y (current position) is in the map.
    x_in = (x <= max_x) & (x >= 0) 
    y_in = (y <= max_y) & (y >= 0) 
    return bool(x_in & y_in) # only true if both true
# ===========================
def possible_next_steps(grid_dim, current_pos):

    # just the current position.
    x_pos, y_pos = current_pos
    
    # we have four possible actions to take, up, down, right, left.
    # every effective possible step should be two block away, we can find it later.
    possible_steps = []
    operations_1 = [(0,1), (0,-1), (1,0), (-1,0)]
    operations_2 = [(0,2), (0,-2), (2,0), (-2,0)]

    num_operations = len(operations_1)
    for i in range(num_operations):
        op1_x, op1_y = operations_1[i]
        op2_x, op2_y = operations_2[i]
        # the possible step is available.
        if (is_in_map((x_pos + op1_x, y_pos + op1_y), grid_dim)) and (is_in_map((x_pos + op2_x, y_pos + op2_y), grid_dim)):
            possible_steps.append([(x_pos + op1_x, y_pos + op1_y), (x_pos + op2_x, y_pos + op2_y)])
    return possible_steps
# ===========================
def generate_step(grid, current_pos, pos_history, back_step):

    (x, y) = current_pos
    grid[x, y] = 1
    
    grid_dim = (len(grid), len(grid[0]))
    # get the possible steps by pass the cur_pos.
    possible_steps = possible_next_steps(grid_dim, current_pos)
    # to test whether the steps are valid.
    valid_steps = []
    for step in possible_steps:             # here we get all the valid next move.
        (x1, y1) = step[0]       # (x_pos + op1_x, y_pos + op1_y)
        (x2, y2) = step[1]       # (x_pos + op2_x, y_pos + op2_y)
        # if the conseccutive two blocks on your way are not the blocks you have walked ,then that is a valid action to take.
        # Otherwise, we will create the loop, which should not to be in the maze.
        not_white = (grid[x1, y1] != 1) & (grid[x2, y2] != 1)  # the path has been walked
        not_green = (grid[x1, y1] != 2) & (grid[x2, y2] != 2)  # start point
        
        if bool(not_white & not_green):      # * has the same effect as &
            valid_steps.append(step)
    
    #print(f"Valid steps: {valid_steps}")
    
    if (len(valid_steps) == 0): # if it is a dead end
        # go back to the cross road to take the other road.
        current_pos = pos_history[-2 - back_step]
        # if we go back to start point, which means the maze is done.
        if current_pos == start_point:
            done = True
            return grid, current_pos, back_step, done
        back_step += 1
        done = False
        return grid, current_pos, back_step, done
    
    else:
        back_step = 0 # reset it
        # choose a valid step at random
        if (len(valid_steps) == 1):            # you have only one valid step.
            current_pos = valid_steps[0]
            (x1, y1) = current_pos[0]
            (x2, y2) = current_pos[1]
            grid[x1, y1] = 1
            grid[x2, y2] = 4
            current_pos = current_pos[1] # walk two block
            done = False
            return grid, current_pos, back_step, done
        else:
            index = randint(0, len(valid_steps))
            current_pos = valid_steps[index]
            (x1, y1) = current_pos[0]
            (x2, y2) = current_pos[1]
            grid[x1, y1] = 1
            grid[x2, y2] = 4
            current_pos = current_pos[1]
            done = False
            return grid, current_pos, back_step, done
#==============================================================================
#==============================================================================

if __name__ == "__main__":

  start_t0 = time.time()

  # define the two colors of the grid RGB
  black = (0, 0, 0) # grid == 0, wall.
  white = (255, 255, 255) # grid == 1, road.
  green = (50,205,50) # grid == 2, start_point.
  red = (255,99,71) # grid == 3, goal.
  grey = (211,211,211) # for background, the region in window outside the maze.
  blue = (153,255,255) # grid[x][y] == 4, where current position is

  # set the height/width of each location on the grid
  height = 2
  width = height # i want the grid square
  margin = 1 # sets margin between grid locations

  # parsing user input
  # example: python maze_generator.py --display=1 --num_mazes=1, the first maze generated will be maze_0.csv.
  parser = argparse.ArgumentParser()
  parser.add_argument("--display", help="Display generating process 0 for False, 1 for True", default=1, type=int)
  parser.add_argument("--num_mazes", help="Number of mazes to generate.", default=1, type=int)
  args = parser.parse_args()

  for iter_maze in range(args.num_mazes):
    start_t = time.time()

    num_rows = 249             #this can change
    num_columns = 349    #this can change
    # initialize the grid array full of zeros(black, all set to walls)
    grid = np.zeros((num_rows, num_columns))

    if args.display == 1:
      # initialize pygame
      pygame.init()

      # congiguration of the window
      WINDOW_SIZE = [1000, 1000]  # changed by the factors of num_column,block height....., set an enough window size.
      screen = pygame.display.set_mode(WINDOW_SIZE)
      # screen title
      pygame.display.set_caption(f"Generating Maze {iter_maze+1}/{args.num_mazes}...")

      done = False # loop until done
      run = False # when run = True start running the algorithm

      clock = pygame.time.Clock() # to manage how fast the screen updates

      idx_to_color = [black, white, green, red, blue]

      # initialize current_pos variable. Its the starting point for the algorithm
      start_point = (num_rows - 1, 0) #this can change
      current_pos = start_point
      pos_history = []
      pos_history.append(current_pos)
      back_step = 0

      # define start and goal
      grid[-1, 0] = 2   #if change start point, change this two value
      grid[0, -1] = 3

      # main program
      while not done:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            done = True
              
          # wait for user to press RETURN key to start    
          elif event.type == pygame.KEYDOWN:
              if event.key==pygame.K_RETURN:
                  run = True
      
        screen.fill(grey) # fill background in grey
        
        # draw
        grid[-1, 0] = 2
        grid[0, -1] = 3
        for row in range(num_rows):
          for column in range(num_columns):
            color = idx_to_color[int(grid[row, column])]
            # the function to paint all the block in maze.
            pygame.draw.rect(screen, color, [(margin + width) * column + margin, (margin + height) * row + margin,width, height])
        # set limit to 60 frames per second
        clock.tick(60)
      
        # update screen
        pygame.display.flip()
      
        if run == True:
            # feed the algorithm the last updated position and the grid
            grid, current_pos, back_step, done = generate_step(grid, current_pos, pos_history, back_step)
            if current_pos not in pos_history:
                pos_history.append(current_pos)

      close = False
      while not close:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            close = True
            pygame.quit()
        # press down to close.
        if event.type == pygame.KEYDOWN:
            close = True
            pygame.quit()
    else:
      print(f"Generating Maze {iter_maze}/{args.num_mazes}...", end=" ")

      done = False # loop until done

      # initialize current_pos variable. Its the starting point for the algorithm
      start_point = (num_rows - 1, 0)
      current_pos = start_point
      pos_history = []
      pos_history.append(current_pos)
      back_step = 0

      # color the start and goal
      grid[-1, 0] = 2
      grid[0, -1] = 3

      # main program
      while not done:
        # feed the algorithm the last updated position and the grid
        grid, current_pos, back_step, done = generate_step(grid, current_pos, pos_history, back_step)
        if current_pos not in pos_history:
          pos_history.append(current_pos)

    # export maze to .csv file
    with open(f"mazes/maze_{iter_maze}.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerows(grid)
    print(f"{time.time()-start_t:.3f} s")

  print(f"--- finished {time.time()-start_t0:.3f} s ---")
  exit(0)