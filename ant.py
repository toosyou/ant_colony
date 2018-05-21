import numpy as np
from tqdm import tqdm

def read_terrain(filename='P4ds0.txt'):
    terrian = list()
    with open(filename, 'r') as in_terrian:
        size_grid = int(in_terrian.readline())
        for _ in range(size_grid):
            in_terrian.readline() # padding
            line = in_terrian.readline()
            terrian.append([int(x) for x in line.split()])

        in_terrian.readline() # padding
        points = in_terrian.readline()
        start_point = [int(x) for x in points.split()[:2]][::-1]
        dest_point = [int(x) for x in points.split()[2:]][::-1]

    terrian = np.array(terrian)
    return terrian, start_point, dest_point

class Ant:
    def __init__(self, position):
        self.start_point = position
        self.x = position[0]
        self.y = position[1]
        self.gone_map = None
        self.path = [[self.x, self.y]]

    def move(self, pheromones_map, follow_chance=0.8):
        # init gone map
        if self.gone_map is None:
            self.gone_map = np.zeros_like(pheromones_map, dtype=np.bool) # all False
            self.gone_map[self.x, self.y] = True

        if np.random.rand() < follow_chance: # follow
            # find the pheronmones-strongest way
            highest_pheromones = -1
            best_dx = 0
            best_dy = 0
            for dx, dy in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                # check boundary
                if self.x + dx >= pheromones_map.shape[0] or \
                    self.y + dy >= pheromones_map.shape[1] or \
                    self.x + dx < 0 or self.y + dy < 0: continue

                # check gone
                if self.gone_map[self.x + dx, self.y + dy]: continue

                if pheromones_map[self.x + dx, self.y + dy] > highest_pheromones:
                    highest_pheromones = pheromones_map[self.x + dx, self.y + dy]
                    best_dx, best_dy = dx, dy
        else: # random way
            find_a_way = False
            move_option = [[1, 0], [0, 1], [-1, 0], [0, -1]]
            np.random.shuffle(move_option)
            for dx, dy in move_option:

                # check boundary
                if self.x + dx >= pheromones_map.shape[0] or \
                    self.y + dy >= pheromones_map.shape[1] or \
                    self.x + dx < 0 or self.y + dy < 0: continue

                # check gone
                if self.gone_map[self.x + dx, self.y + dy]:continue

                best_dx, best_dy = dx, dy
                find_a_way = True
                break

            if not find_a_way: # DEAD END
                self.x, self.y = self.start_point
                self.gone_map = None
                self.path = [[self.x, self.y]]
                return self.move(pheromones_map, follow_chance)

        self.x, self.y = self.x + best_dx, self.y + best_dy
        self.gone_map[self.x, self.y] = True
        self.path.append([self.x, self.y])
        return self

    def dest_found(self, dest_point):
        return self.x == dest_point[0] and self.y == dest_point[1]

    def cost(self, terrian, alpha=1, beta=10):
        hights = [ terrian[x, y] for x, y in self.path ]
        diff = np.diff(hights)
        return alpha * np.sum(diff*diff) + beta * (len(self.path)-1.)

def find_optimal(terrian, start_point, dest_point,
                    number_ants=100, number_generations = 10,
                    ant_follow_ratio=0.8,
                    local_evaporation_ratio=0.2,
                    global_evaporation_ratio=0.4,
                    cost_alpha=1, cost_beta=10, verbose=False):
    def local_update(pheromones_map, ant):
        maximum_pheromones = -1
        for x, y in ant.path:
            # evaporation
            pheromones_map[x, y] *= (1. - local_evaporation_ratio)
            maximum_pheromones = max(maximum_pheromones, pheromones_map[x, y])
        # ants passing
        for x, y in ant.path:
            pheromones_map[x, y] += maximum_pheromones * local_evaporation_ratio

    def global_update(pheromones_map, best_path, best_cost):
        # evaporation
        pheromones_map *= (1. - global_evaporation_ratio)
        # best path update
        for x, y in best_path:
            pheromones_map[x, y] += 1./best_cost * global_evaporation_ratio

    pheromones_map = np.ones_like(terrian, dtype=np.float)
    best_path = list()
    best_cost = np.inf
    for generation in tqdm(range(number_generations)):
        ants = [Ant(start_point) for _ in range(number_ants)]
        while len(ants):
            for index_ant, ant in enumerate(ants):
                if ant.move(pheromones_map, ant_follow_ratio).dest_found(dest_point): # reach destination
                    local_update(pheromones_map, ant)
                    cost = ant.cost(terrian, cost_alpha, cost_beta)
                    if cost < best_cost: # update best path
                        best_path = ant.path
                        best_cost = cost
                    ants.remove(ant)

        global_update(pheromones_map, best_path, best_cost)
        if verbose: print('generation:', generation, 'best_cost:', best_cost)
    return np.array(best_path), best_cost, pheromones_map

if __name__ == '__main__':
    terrian, start_point, dest_point = read_terrain('P4ds2.txt')
    best_path, best_cost, pheromones_map = find_optimal(terrian, start_point, dest_point,
                                                        number_ants=100, number_generations = 25,
                                                        ant_follow_ratio=0.7,
                                                        local_evaporation_ratio=0.8,
                                                        global_evaporation_ratio=0.1,
                                                        cost_alpha=1, cost_beta=20, verbose=False)
