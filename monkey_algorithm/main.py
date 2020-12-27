import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from typing import Tuple

LIBRARY_5 = np.memmap(r'F:\Downloads\5v_all', mode='r')
calculated_functions: dict = {}


def create_population(population_size: int, len_max: int, len_min: int, n: int, f1: int, f2: int) -> list:
    """Create population."""
    monkeys = []
    for i in range(population_size):
        if i % 2 == 0:
            monkey = randint(0, (2 ** len_min) - 1)
        else:
            monkey = randint(0, (2 ** len_max) - 1)

        new_f1 = f1 ^ monkey
        new_f2 = f2 ^ monkey
        if n == 6:
            alpha = LIBRARY_5[new_f1] + LIBRARY_5[new_f2] + LIBRARY_5[monkey]
        elif n == 7:
            alpha_new_f1 = six_monkey(new_f1)
            alpha_new_f2 = six_monkey(new_f2)
            alpha_monkey = six_monkey(monkey) 
            alpha = alpha_new_f1 + alpha_new_f2 + alpha_monkey
        monkeys.append([monkey, alpha])
        calculated_functions[monkey] = alpha

    if n == 7:
        print('init population:', monkeys)
    return monkeys


def get_f(n: int, func: int) -> Tuple[int, int]:
    """Return f(0), f(1)."""
    power = 1 << (n - 1)
    f1 = func >> power
    f2 = func & (1 << power) - 1

    return f1, f2


def search_parametr_5(population: list, f1: int, f2: int, jumps: int, n: int, *args) -> None:
    """Calculate difficulty for parameter of five variables."""
    power = (1 << (n - 1)) - 1
    counter = 0
    half_func = [((1 << 32) - 1) >> 16] 

    for p in population:
        counter = 0
        while counter != len(args):
            if counter == 0:
                jump = 32
            else:
                jump = jumps
                rand = [randint(0, power) for _ in range(args[counter])]

            for i in range(jump):
                parameter = 1

                if counter == 0:
                    parameter << i
                else:
                    for k in range(len(rand)):
                        parameter ^= (1 << rand[k])

                func = p[0]
                neighbor = func ^ parameter

                alpha_0 = LIBRARY_5[f1 ^ neighbor]
                alpha_1 = LIBRARY_5[f2 ^ neighbor]
                alpha = alpha_0 + alpha_1 + LIBRARY_5[neighbor]
                
                if alpha < p[1]:
                    p[0] = neighbor
                    p[1] = alpha
                    break

                if alpha >= p[1] and i == (jump - 1):
                    counter += 1

        for hf in half_func:        
            func = p[0]
            neighbor = func ^ hf

            alpha_0 = LIBRARY_5[f1 ^ neighbor]
            alpha_1 = LIBRARY_5[f2 ^ neighbor]
            alpha = alpha_0 + alpha_1 + LIBRARY_5[neighbor]

            if alpha < p[1] and neighbor not in calculated_functions:
                p[0] = neighbor
                p[1] = alpha


def six_monkey(func, population_size: int=2, jumps: int=10, iterations: int=4) -> int:
    """Calculate difficulty for function of six variables."""
    N = 6
    f1, f2 = get_f(N, func)
    len_max = len(bin(f1)[2:])
    len_min = len(bin(f2)[2:])
    if len_max < len_min:
        len_max, len_min = len_min, len_max
    
    population = create_population(population_size, len_max, len_min, n=N, f1=f1, f2=f2)
    for j in range(iterations):
        search_parametr_5(population, f1, f2, jumps, N, 1, 4, 8)
    answer = sorted(population, key=lambda x: x[1])[0]

    return answer[1]


def seven_monkey(func: int, population_size: int=1, jumps: int=1, iterations: int=1, ax_index: int=0) -> list:
    """Calculate difficulty for function of seven variables."""
    N = 7
    f1, f2 = get_f(N, func)
    len_max = len(bin(f1)[2:])
    len_min = len(bin(f2)[2:])
    if len_max < len_min:
        len_max, len_min = len_min, len_max
    population = create_population(population_size, len_max, len_min, n=N, f1=f1, f2=f2)

    for i in range(iterations):
        search_parametr_6(population, f1, f2, jumps, N, 1, 1, 16, 25, 17)
        draw_chart(
                axises[ax_index],
                i,
                sum([x[1] for x in population]) // population_size
            )

    return population


def search_parametr_6(population: list, f1: int, f2: int, jumps: int, n: int, *args) -> None:
    """Calculate difficulty for parameter of six variables."""
    power = (1 << (n - 1)) - 1
    counter = 0
    half_func = [((1 << 64) - 1) >> 32]

    for p in population:
        counter = 0
        while counter != len(args):
            if counter == 0:
                jump = 64
            else:
                jump = jumps
                rand = [randint(0, power) for _ in range(args[counter])]

            for i in range(jump):
                parameter = 1
                
                if counter == 0:
                    parameter << i
                else:
                    for k in range(len(rand)):
                        parameter ^= (1 << rand[k])
                
                func = p[0]
                neighbor = func ^ parameter

                new_f1 = f1 ^ neighbor
                if new_f1 in calculated_functions:
                    alpha_0 = calculated_functions[new_f1]
                else:
                    alpha_0 = six_monkey(new_f1)
                    calculated_functions[new_f1] = alpha_0

                new_f2 = f2 ^ neighbor
                if new_f2 in calculated_functions:
                    alpha_1 = calculated_functions[new_f2]
                else:
                    alpha_1 = six_monkey(new_f2)
                    calculated_functions[new_f2] = alpha_1

                if neighbor in calculated_functions:
                    alpha_neighbor = calculated_functions[neighbor]
                else:
                    alpha_neighbor = six_monkey(neighbor)
                    calculated_functions[neighbor] = alpha_neighbor

                alpha = alpha_0 + alpha_1 + alpha_neighbor

                if alpha < p[1]:
                    p[0] = neighbor
                    p[1] = alpha
                    break

                if alpha >= p[1] and i == (jump - 1):
                    counter += 1

        for hf in half_func:         
            func = p[0]
            neighbor = func ^ hf

            alpha_0 = six_monkey(f1 ^ neighbor)
            alpha_1 = six_monkey(f2 ^ neighbor)
            alpha = alpha_0 + alpha_1 + six_monkey(neighbor)

            if alpha < p[1] and neighbor not in calculated_functions:
                p[0] = neighbor
                p[1] = alpha


def init_chart(func: int):
    """Init chart."""
    fig = plt.figure()
    fig.suptitle(
        f'Boolean function: ({bin(func)[2:]})', 
        fontweight='bold', 
        fontsize=16
    )
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    grid_spec = gridspec.GridSpec(
        ncols=2,
        nrows=2,
        figure=fig,
        wspace=0.4,
        hspace=0.46,
        left=0.06,
        right=0.94
    )
    fig_ax1 = fig.add_subplot(grid_spec[0, 0])
    fig_ax2 = fig.add_subplot(grid_spec[0, 1])
    fig_ax3 = fig.add_subplot(grid_spec[1, 0])
    fig_ax4 = fig.add_subplot(grid_spec[1, 1])

    return (
        fig_ax1,
        fig_ax2,
        fig_ax3,
        fig_ax4,
    )


def draw_chart(fig, iteration: int, min_difficulty: int) -> None:
    """Draw chart."""
    fig.bar(iteration, min_difficulty, color='black')


if __name__ == '__main__':
    start_time = time.time()
    func = int(
        '0b1000111110011101101101001001111011110111101010101110011000011001100000001011',
        base = 2
    )

    if func <= 4_294_967_295:
        print(LIBRARY_5[func])
    elif func <= (1 << 64) - 1:
        print(six_monkey(func))
    else:
        population_size = 10
        jumps = 20
        iterations = 10
        
        axises = init_chart(func)
        for item in axises:
            item.set_xlabel('Iterations', fontsize=18, fontweight='bold')
            item.set_ylabel('Difficulty', fontsize=18, fontweight='bold')

        for i in range(4):
            population_size += i*5
            algo_params = {
                'population_size': population_size,
                'jumps': jumps,
                'iterations': iterations
            }

            population = seven_monkey(func, population_size, jumps, iterations, i)
            result = sorted(population, key=lambda x: x[1])[0]

            algo_params[f'p: {result[0]}'] = f'min: {result[1]}'
            axises[i].set_title(algo_params, fontweight='bold', fontsize=14)
        plt.show()
    print("----- %s seconds -----" % (time.time() - start_time))
