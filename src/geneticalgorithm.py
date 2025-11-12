#This is a genetic algorithm implementation for solving a 
#variation of the one dimensional Bin Packing Problem.

import random
import numpy as np

def create_population(pop_size: int, n_items: int, n_bins: int) -> list[int]:
    """
    **Creates a population of p randomly generated chromosomes**
    
    **Arguments**
        - pop_size
            - The size of the population to be created
        - n_items
            - The number of items to be packed
        - n_bins
            - The number of bins available for packing
    
    **Returns**
            - A list representing the population of chromosomes
    """
    population = []
    for _ in range(pop_size):
        chromosome = np.random.randint(1, n_bins + 1, size=n_items)
        population.append(chromosome.tolist())

    return population

print(create_population(5, 5, 5))

def fitness(chromosome: list[int], item_weights: list[int], num_bins: int) -> int:
    """
    **Calculates the fitness of a chromosome**
    
    **Arguments**
        - chromosome
            - The chromosome to be evaluated
        - item_weights
            - A list of the weights of the items to be packed
        - num_bins
            - The number of bins available for packing
    
    **Returns**
        - The fitness value of the chromosome

    """
    bin_weights = np.zeros(num_bins)
    for i, bin_id in enumerate(chromosome):
        bin_weights[bin_id - 1] += item_weights[i]
    d = max(bin_weights) - min(bin_weights)
    return 100 / (1 + d)
    