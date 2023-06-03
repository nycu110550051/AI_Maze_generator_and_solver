import pygame, time, csv, argparse
import numpy as np
from time import sleep
import sys
class Node:
    def __init__(self, parent, cost, position):
        self.parent = parent
        self.cost = cost
        self.position = position

def remove_node_frontier(node, frontier):
    
    pos = node.position
    new_frontier = []
    for item in frontier:
        if item.position != pos:
            new_frontier.append(item)
    return new_frontier

def compute_cost(pos, goal):
    # cost is the distance to goal
    x, y = pos
    x_goal, y_goal = goal
    cost = np.sqrt((x_goal-x)*(x_goal-x) + (y_goal-y)*(y_goal-y))
    return cost

def is_in_map(pos, grid_dim):
    (max_x, max_y) = grid_dim
    (x, y) = pos 
    x_in = (x <= max_x) & (x >= 0)
    y_in = (y <= max_y) & (y >= 0)
    return bool(x_in & y_in)

def compute_successors(current_node, grid, path):
    
    movements = [(1,0), (-1,0), (0,1), (0,-1)]
    x_0, y_0 = current_node.position
    grid_dim = (len(grid)-1, len(grid[0])-1)
            
    successors = []
    for movement in movements:
        dx, dy = (movement[0], movement[1])
        possible_pos = (x_0+dx, y_0+dy)
        if is_in_map(possible_pos, grid_dim):
            case1 = grid[possible_pos[0], possible_pos[1]] != 0 # not wall
            case2 = possible_pos not in [item.position for item in path] # dont go to the visited block
            # compute all the valid block's cost
            if bool(case1 & case2):
                cost = compute_cost(possible_pos, (0, len(grid[0])-1))# remember to modify this as goal
                new_node = Node(current_node, cost, possible_pos)
                successors.append(new_node)
    return successors

def pick_node_least_cost(valid_node, startp):
    # only have one choice
    if len(valid_node) == 1:
        current_node = valid_node[0]
        return current_node
    # choose the node has least cost
    l_cost_node = Node(None, np.inf, (startp-1, 0)) # initialize current node
    for node in valid_node:
        if node.cost < l_cost_node.cost: # pick up the node with least cost
            l_cost_node = node
    return l_cost_node

def generate_step(grid, path, current_node, thresh, goal):
    # arrived
    if current_node.position == goal:
        return grid, path, current_node, -1
    # control the cost we explore this maze, if the cost is bigger than thresh, we just return the current_node.cost.
    if current_node.cost > thresh:
        return grid, path, current_node, current_node.cost
    
    successors = compute_successors(current_node, grid, path)
    
    # add successors to the frontier
    minimum = float("inf")
    for child in successors:
        path.append(child)
        # tmp is the only value we need
        # thresh means a threshold, if the 
        _,_,_,tmp = generate_step(grid, path, child, thresh, goal)
        if tmp == -1:
            current_node = child
            return grid, path, current_node, -1
        if tmp < minimum:
            minimum = tmp
            current_node = child
    x, y = current_node.position
    grid[x, y] = 4 
    return grid, path, current_node, minimum



if __name__ == "__main__":

  start_t0 = time.time()

  # parsing user input
  # example: python aStar_generator.py --display=True --maze_file=maze_1.csv
  parser = argparse.ArgumentParser()
  parser.add_argument("--display", help="Display generating process 0 for False, 1 for True", default=1, type=int)
  parser.add_argument("--maze_file", help="filename (csv) of the maze you want to load.", default="maze_1.csv", type=str)
  args = parser.parse_args()

  address = "mazes/" + args.maze_file
  grid = np.genfromtxt(address, delimiter=',', dtype=int)

  # because this algorithm is based on recursion, so we have to define this variable to deal with complicated cases.
  sys.setrecursionlimit(5000)

  # define goal and start
  num_rows = len(grid)
  num_columns = len(grid[0])
  goal = (0, num_columns-1)  # this can change
  start = (num_rows-1, 0)    # this can change
  grid[-1, 0] = 2
  grid[0, -1] = 3
  save = np.copy(grid)

  # define start node
  start_node = Node(None, 0, start)

  # initialize visit, frontier list
  path = [start_node] # starts empty
   # starts with the start node


  if args.display == 1:

    # define colors of the grid RGB
    black = (0, 0, 0) # grid == 0
    white = (255, 255, 255) # grid == 1
    green = (0,255,0) # grid == 2
    red = (255,0,0) # grid == 3
    gray = (211,211,211) # for background
    blue = (0,0,255) # grid == 4, where current position is
    magenta = (255,0,255) # grid == 5 solution

    # set the height/width of each location on the grid
    height = 8
    width = height
    margin = 1

    # initialize pygame
    pygame.init()

    # congiguration of the window
    WINDOW_SIZE = [600, 600]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    pygame.display.set_caption(f"Astar_Pathfinder. Solving: {address}")
    clock = pygame.time.Clock() # to manage how fast the screen updates

    idx_to_color = [black, white, green, red, blue, magenta]

    # loop until done
    close = False # when user clicks exit
    run = False # when algorithm starts
    finish = False
    done = False

    # main painting loop

    thresh = 0
    while not close:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                close = True
                
            # wait for user to press RETURN key to start    
            elif event.type == pygame.KEYDOWN:
                if event.key==pygame.K_RETURN:
                    run = True
        
        screen.fill(gray) # fill background in gray
        
        for row in range(num_rows):
            for column in range(num_columns):
                color = idx_to_color[grid[row, column]]
                pygame.draw.rect(screen, color, [(margin + width) * column + margin, (margin + height) * row + margin, width, height])
        
        # set limit to 60 frames per second
        clock.tick(60)
        
        # update screen
        pygame.display.flip()
        
        if done == True:
            close = True
            run = False
        elif run == True:
            path = [start_node]
            # pick a node from the frontier  # choose the node has the least cost
            grid = np.copy(save)
            if not done:
                grid, path, current_node, thresh = generate_step(grid, path, start_node, thresh, goal)
            if thresh == -1: # if it is at goal then finish
                done = True
                ### follow the parents back to the origin
                current_node = path[-1]
                while current_node.parent != None:
                    x, y = current_node.position
                    grid[x,y] = 5
                    current_node = current_node.parent

        
    while not finish:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finish = True
            # wait for user to press any key to start    
            elif event.type == pygame.KEYDOWN:
                finish = True
    pygame.quit() # so that it doesnt "hang" on exit

  else:

    print(f"Astar_Pathfinder. Solving: {address}")
    
    thresh = 0
    done = False
    while not done:
        
      # pick a node from the frontier
      path = [start_node]
      grid = np.copy(save)
      grid, path, current_node, thresh = generate_step(grid, path, start_node, thresh, goal)

      if thresh == -1: # if ai is at goal then finish
        done = True
        # follow the parents back to the origin
        current_node = path[-1]
        while current_node.parent != None:
            x, y = current_node.position
            grid[x,y] = 5
            current_node = current_node.parent


  # export maze to .csv file
  with open(f"mazes_solutions/Astar_{args.maze_file}", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(grid)

  print(f"--- finished {time.time()-start_t0:.3f} s ---")
  exit(0)