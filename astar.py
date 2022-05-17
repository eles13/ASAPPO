from heapq import heappop, heappush, heapify
import random

MOVES = tuple([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), ])


class SearchTree:
    
    def __init__(self):
        self._open = []
        self._closed = set()
        self._enc_open_dublicates = 0
        
    def __len__(self):
        return len(self._open) + len(self._closed)
                    
    def open_is_empty(self):
        return len(self._open) == 0
    
    def add_to_open(self, item):
        heappush(self._open, item)
        return    
    
    def get_best_node_from_open(self):
        best = heappop(self._open)
        # while self.was_expanded(best):
        #     if self.open_is_empty():
        #         return None
        #     best = heappop(self._open)
        return best

    def add_to_closed(self, item):
        self._closed.add(item)

    def was_expanded(self, item):
        return item in self._closed

    @property
    def OPEN(self):
        return self._open
    
    @property
    def CLOSED(self):
        return self._closed

    @property
    def number_of_open_dublicates(self):
        return self._enc_open_dublicates

class Node:

    def __init__(self, i, j, g = 0, h = 0, f = None, parent = None, action=None):
        self.i = i
        self.j = j
        self.g = g
        self.h = h
        if f is None:
            self.f = self.g + h
        else:
            self.f = f        
        self.parent = parent  
        self.moveidx = action    
    
    def __eq__(self, other):
        return (self.i == other.i) and (self.j == other.j)
    
    def __hash__(self):
        ij = self.i, self.j
        return hash(ij)

    def __lt__(self, other): 
        return self.f < other.f

def manhattan_distance(i1, j1, i2, j2):
    return abs(i2 - i1) + abs(j2 - j1)
#elif self.hidden_agents[self.grid.positions_xy.index((x + dx, y + dy))] is not None:
def get_neighbors(ag, i, j, t, grid, block_table):
    availabes = []
    for idx, move in enumerate(MOVES):
        dx, dy = move
        # or (i + dx, j + dy) in grid.finishes_xy and not grid.is_active(grid.finishes_xy.index((i + dx, j + dy)))) and \
        if (grid.obstacles[i + dx, j + dy] == 0) and \
            (i + dx, j + dy, t + 1) not in block_table and \
                ((i,j,t+1) in block_table and (i+dx, j+dy, t) in block_table and block_table[(i,j,t+1)] != block_table[(i+dx, j+dy, t)] or \
                     (i,j,t+1) not in block_table or (i+dx, j+dy, t) not in block_table):
            availabes.append((i + dx, j + dy, idx))
        #print(ag, availabes, t)
    return availabes


def local_astar(ag, num_agents, grid, heuristic_func = None, search_tree = None, block_table={}):
    ast = search_tree()
    steps = 0
    start_i, start_j = grid.positions_xy[ag]
    goal_i, goal_j = grid.finishes_xy[ag]
    start = Node(start_i, start_j, g=0, h=heuristic_func(start_i, start_j, goal_i, goal_j))
    
    ast.add_to_open(start)
    
    while not ast.open_is_empty():
        #print(ast.OPEN)
        steps += 1
        curr = ast.get_best_node_from_open()
        if curr is None:
            #print('break on none', ag)
            break
        if (curr.i, curr.j) == (goal_i, goal_j):
            return curr, steps - 1
        
        for (i, j, moveidx) in get_neighbors(ag, curr.i, curr.j, curr.g, grid, block_table):
            # if ast.was_expanded(Node(i, j)):
            #     continue
            succ = Node(i, j, g=curr.g+1, h=heuristic_func(i, j, goal_i, goal_j), parent=curr, action=moveidx)
            ast.add_to_open(succ)    
                
        ast.add_to_closed(curr)
    
    return None, None


def coop_astar(conflicting, num_agents, grid, heuristic_func = None, search_tree = None, max_sim_steps=64, random_boss=False):

    block_table = {(i,j,0): enum for enum, (i,j) in enumerate(grid.positions_xy) if grid.finishes_xy[enum] != (i,j)}
    #print(grid.positions_xy)
    dones = 0
    actions = {}
    for ag in range(len(conflicting)):
        if random_boss:
            boss = random.choice(conflicting)
            while boss in actions: 
                boss = random.choice(conflicting)
            ag = boss
        else:
            ag = conflicting[ag]
        # if ag in [0,7]:
        #     print(ag, block_table)
        if grid.finishes_xy[ag] != grid.positions_xy[ag]:
            finalnode, steps = local_astar(ag, num_agents, grid, heuristic_func, search_tree, block_table)
            # if ag in [0,7]:
            #     print(finalnode, steps)
            if finalnode:
                dones += 1
                while finalnode and finalnode.parent:
                    # if ag in [5,7]:
                    #     print('par', finalnode)
                    if finalnode.parent.parent is None:
                        actions[ag] = finalnode.moveidx
                    block_table[(finalnode.i, finalnode.j, finalnode.g)] = ag
                    finalnode = finalnode.parent
    # print(block_table, actions)
    #print(pg.get_score(), dones)
    return actions
