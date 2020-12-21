import numpy as np
from random import randint
import time


def initial_trees(population_size, n):
    """Create initial trees

    population_size - trees count
    n - function length

    """
    trees = []
    for i in range(population_size):
        tree = randint(0, (2 ** 2 ** (n - 1)) - 1)
        trees.append([tree, 0])

    return trees


def fitness_function(f1, f2, population):
    for p in population:
        alpha_1 = fp[f1 ^ p[0]]
        alpha_2 = fp[f2 ^ p[0]]
        alpha = alpha_1 + alpha_2 + fp[p[0]] 
        p[1] = alpha
   

def get_f(n, func):
    """Return f1, f2."""
    power = 2 ** (n - 1)
    f1 = func >> power
    f2 = func & (2 ** power) - 1

    return f1, f2


def get_neighbors(population, transition_param = 1):
    for p in population:
        for _ in range(10):
            rand = np.random.choice((2 ** (n - 1)) - 1, size = 2, replace = False)
            param_1 = int('0b1' + '0' * rand[0], base = 2)
            param_2 = int('0b1' + '0' * rand[1], base = 2)
            func = p[0]
            neighbor_1 = func ^ param_1
            neighbor_2 = func ^ param_2

            alpha_1 = fp[f1 ^ neighbor_1]
            alpha_2 = fp[f2 ^ neighbor_1]
            alpha_sum_1 = alpha_1 + alpha_2 + fp[neighbor_1]

            alpha_3 = fp[f1 ^ neighbor_2]
            alpha_4 = fp[f2 ^ neighbor_2]
            alpha_sum_2 = alpha_3 + alpha_4 + fp[neighbor_2]
            
            if alpha_sum_1 < alpha_sum_2:
                alpha = alpha_sum_1
                neighbor = neighbor_1
            else:
                alpha = alpha_sum_2
                neighbor = neighbor_2

            if alpha < p[1]:
                p[0] = neighbor
                p[1] = alpha


def tree_jump(population):
    for p in population:
        rand = np.random.choice((2 ** (n - 1)) - 1, size = 2 ** (n - 2), replace = False)
        param = 0
        for i in rand:
            param += int('0b1' + '0' * i, base = 2)

        func = p[0]
        neighbor = func ^ param

        alpha_1 = fp[f1 ^ neighbor]
        alpha_2 = fp[f2 ^ neighbor]
        alpha = alpha_1 + alpha_2 + fp[neighbor]

        if alpha < p[1]:
            p[0] = neighbor
            p[1] = alpha   


if __name__ == '__main__':
    fp = np.memmap(r'F:\Downloads\5v_all', mode='r')
    
    population_size = 10
    n = 6
    func = 2 ** 2 ** 5 + 19781216
    jump_param = 2 ** (n - 2)
    transition_param = n - 1

    population = initial_trees(population_size, n)
    f1, f2 = get_f(n, func)
    fitness_function(f1, f2, population)
    print(population)

    for _ in range(100):
        get_neighbors(population)
        tree_jump(population)
    print(population)




    
    
