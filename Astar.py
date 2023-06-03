import pygame, time, csv, argparse
import numpy as np
from time import sleep

class Node:
    def __init__(self, parent, cost, position):
        self.parent = parent
        self.cost = cost
        self.position = position

def remove_node_frontier(node, frontier):
    # apart from the target node, we retain all other nodes.
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
    # check the pos is in the maze.
    (max_x, max_y) = grid_dim 
    (x, y) = pos
    x_in = (x <= max_x) & (x >= 0)
    y_in = (y <= max_y) & (y >= 0)
    return bool(x_in & y_in)

def compute_successors(current_node, grid, visit):
    
    movements = [(1,0), (-1,0), (0,1), (0,-1)]
    x_0, y_0 = current_node.position
    grid_dim = (len(grid)-1, len(grid[0])-1)
            
    successors = []
    for movement in movements:
        dx, dy = (movement[0], movement[1])
        possible_pos = (x_0+dx, y_0+dy)
        if is_in_map(possible_pos, grid_dim):
            case1 = grid[possible_pos[0], possible_pos[1]] != 0 # not wall
            case2 = possible_pos not in [item.position for item in visit] # dont go to the visited block
            # compute all the valid block's cost
            if bool(case1 & case2):
                cost = compute_cost(possible_pos, (0, len(grid[0])-1))# remember to modify this as goal
                new_node = Node(current_node, cost, possible_pos)
                successors.append(new_node)
    return successors

def pick_node_least_cost(node_list, startp):
    # have only one choice
    if len(node_list) == 1:
        current_node = node_list[0]
        return current_node
    # choose the node has least cost
    current_node = Node(None, np.inf, (startp-1, 0)) # initialize current node
    for node in node_list:
        if node.cost < current_node.cost: # pick up the node with least cost
            current_node = node
    return current_node

def generate_step(grid, frontier, visit, current_node):
    
    # add parent to visit list
    visit.append(current_node)

    # remove parent from frontier
    frontier = remove_node_frontier(current_node, frontier)
    
    # given the node compute the successors
    successors = compute_successors(current_node, grid, visit)
    
    # add successors to the frontier
    for child in successors:
        frontier.append(child)

    # pick one of the successors
    current_node = pick_node_least_cost(successors, len(grid))
    
    # paint the grid with the new node position
    x, y = current_node.position
    grid[x, y] = 4 # paint blue
    
    return grid, frontier, visit, current_node

if __name__ == "__main__":
  start_t0 = time.time()

  # parsing user input
  # example: python Astar_pathfinder.py --display=1 --maze_file=maze_0.csv
  parser = argparse.ArgumentParser()
  parser.add_argument("--display", help="Display generating process 0 for False, 1 for True", default=1, type=int)
  parser.add_argument("--maze_file", help="filename (csv) of the maze you want to load.", default="maze_1.csv", type=str)
  args = parser.parse_args()

  address = "mazes/" + args.maze_file
  grid = np.genfromtxt(address, delimiter=',', dtype=int)

  # define goal and start
  num_rows = len(grid)
  num_columns = len(grid[0])
  goal = (0, num_columns - 1)  # this can change
  start = (num_rows - 1, 0)    # this can change
  grid[-1, 0] = 2
  grid[0, -1] = 3

  # define start node
  start_node = Node(None, 0, start)

  # initialize visit, frontier list
  visit = [] # starts empty
  frontier = [start_node] # starts with the start node


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
    width = height # i want the grid square
    margin = 1 # sets margin between grid locations

    # initialize pygame
    pygame.init()

    # congiguration of the window, can modify to feet your maze.
    WINDOW_SIZE = [800, 800]
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
            
            # pick a node from the frontier
            current_node = pick_node_least_cost(frontier, num_rows)    # choose the node has the least cost
            grid, frontier, visit, current_node = generate_step(grid, frontier, visit, current_node)
            if current_node.position == goal: # if it is at goal then finish
                done = True
                ### follow the parents back to the origin
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

    done = False
    while not done:
        
      # pick a node from the frontier
      current_node = pick_node_least_cost(frontier,num_rows)
      grid, frontier, visit, current_node = generate_step(grid, frontier, visit, current_node)

      if current_node.position == goal: # if it is at goal then finish
        done = True
        # follow the parents back to the origin
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