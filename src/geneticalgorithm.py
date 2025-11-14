#This is a genetic algorithm implementation for solving a 
#variation of the one dimensional Bin Packing Problem.

import random
import numpy as np
import matplotlib.pyplot as plt

class BinPackingGA():
    """
    **Genetic Algorithm for the Bin Packing Problem (BPP)**

    **Arguments**
        - n_items
            - The number of items to be packed
        - item_weights
            - A list containing the weights of each item
        - n_bins
            - The number of bins available for packing
        - pop_size
            - The size of the population
        - mutation_rate
            - The mutation rate for the genetic algorithm
        - tournament_size
            - The size of the tournament for selection

    **Attributes**
        - crossover_rate
            - The crossover rate for the genetic algorithm
        - elitism_count
            - The number of elite chromosomes to carry over each generation
        - max_evaluations
            - The maximum number of fitness evaluations
        - evaluation_count
            - The number of fitness evaluations performed
        - best_fitness_history  
            - History of best fitness values
        - avg_fitness_history
            - History of average fitness values
    """
    def __init__(
                    self, 
                    n_items:int, # Number of items to be packed
                    item_weights: list[int], # Weights of the items
                    n_bins: int, # Number of bins available for packing
                    pop_size: int, # Population size
                    mutation_rate: float, # Mutation rate
                    tournament_size: int, # Tournament size for selection
                    use_local_search: bool = False, # Extra research parameter, not used in main experiments
                    local_search_rate: float = 0.1 # Extra research parameter, not used in main experiments
                ):
        # Initialise GA Parameters for BPP
        self.item_weights = item_weights
        self.n_bins = n_bins
        self.n_items = n_items
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        # Fixed GA Parameters
        self.crossover_rate = 0.8 # Crossover rate, fixed from problem specification
        self.elitism_count = 1 # Number of elite chromosomes to carry over each generation
        self.max_evaluations = 10000 # Maximum number of fitness evaluations

        # Tracking Variables
        self.evaluation_count = 0 # Number of fitness evaluations performed
        self.best_fitness_history = [] # History of best fitness values
        self.avg_fitness_history = [] # History of average fitness values

        #Create Initial Random Population
        self.population = None

        #Extra Research Parameter
        self.use_local_search = use_local_search
        self.local_search_rate = local_search_rate

    def hill_climb(self, chromosome: list[int], max_iterations: int = 50) -> list[int]:
        """
        **Performs hill climbing local search on a chromosome**

        **Arguments**
            - chromosome
                - The chromosome to be improved
            - max_iterations
                - The maximum number of iterations for hill climbing
        **Returns**
            - The improved chromosome after hill climbing

        """

        #Get the current chromosome and it's fitness
        current_chromosome = chromosome.copy()
        current_fitness = self.fitness(current_chromosome)

        improvements = 0

        for iteration in range(max_iterations):
            #Get a neighbour by moving a random item to a different random bin
            neighbor = current_chromosome.copy()
            item_to_move = random.randint(0, self.n_items - 1)
            current_bin = neighbor[item_to_move]

            #Choose a different bin thats not the current one
            possible_bins = []
            for k in range(1, self.n_bins + 1):
                if k != current_bin:
                    possible_bins.append(k)
            
            # Move the item to the new bin
            new_bin = random.choice(possible_bins)
            neighbor[item_to_move] = new_bin

            #Evaluate the neighbor's fitness
            neighbor_fitness = self.fitness(neighbor)
            self.evaluation_count += 1

            #If the neighbor is better, move to it
            if neighbor_fitness > current_fitness:
                current_chromosome = neighbor
                current_fitness = neighbor_fitness
                improvements += 1

            if self.evaluation_count >= self.max_evaluations:
                break

        return current_chromosome

    def create_population(self, pop_size: int, n_items: int, n_bins: int) -> list[int]:
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
        # Create initial random population, list of random gene assignments to bins
        population = []
        for _ in range(pop_size):
            chromosome = np.random.randint(1, n_bins + 1, size=n_items)
            population.append(chromosome.tolist())

        return population

    def calculate_bin_weights(self, chromosome: list[int]) -> list[float]:
        """
        **Calculates the weights of each bin for a given chromosome**

        **Arguments**
            - chromosome
                - The chromosome representing the packing solution
        **Returns**
            - A list of weights for each bin
        """
        # Calculate the total weight in each bin based on the chromosome
        bin_weights = [0.0] * self.n_bins
        
        # Sum the weights of items assigned to each bin
        for item, bin_num in enumerate(chromosome):
            bin_weights[bin_num - 1] += self.item_weights[item]

        return bin_weights


    def fitness(self, chromosome: list[int]) -> float:
        """
        **Calculates the fitness of a chromosome**
        
        **Arguments**
            - chromosome 
                - The chromosome to be evaluated
        
        **Returns**
            - The fitness value of the chromosome

        """
        # Calculate the fitness of the chromosome based on the difference between max and min bin weights
        bin_weights = self.calculate_bin_weights(chromosome)

        # Filter out empty bins to reduce impact of empty bins on fitness
        # The CA specification only says that heaviest and lightest bins should be considered,
        # Assuming that it means heaviest and lightest bins with items should be taken into account.
        non_empty_bins = [weight for weight in bin_weights if weight > 0]

        if len(non_empty_bins) == 0:
            return 0.0
        
        d = max(non_empty_bins) - min(non_empty_bins)

        return 100 / (1 + d) # CA Fitness Function
    
    def tournament_selection(self, population_set: list[int], fitnesses: list[float]) -> list[int]:
        """
        **Selects a chromosome from the population using tournament selection**
        **Arguments**
            - fitnesses
                - A list of fitness values for the current population
        **Returns**
            - The selected chromosome from the tournamen
        """

        # Randomly sample chromosomes for the tournament
        tournament_sample = random.sample(range(len(population_set)), self.tournament_size)
        tournament_fitness = [] # Fitness values of the sampled chromosomes
        for solution in tournament_sample:
            tournament_fitness.append(fitnesses[solution])
        winner_index = tournament_sample[np.argmax(tournament_fitness)] # Index of the chromosome with the highest fitness in the tournament
        return population_set[winner_index] # Return the winning chromosome

    def uniform_crossover(self, parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        """
        **Performs uniform crossover between two parent chromosomes**
        **Arguments**
            - parent1
                - The first parent chromosome
            - parent2
                - The second parent chromosome
        **Returns**
            - Two child chromosomes resulting from the crossover
        """
        if random.random() > self.crossover_rate:
            # No crossover, return parents as children, represents the chance of no crossover occurring
            return parent1.copy(), parent2.copy()
        
        child1 = []
        child2 = []

        # Perform uniform crossover
        for i in range(self.n_items):
            if random.random() < 0.5: #Coin flip specified by the CA
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])


        return child1, child2
    
    def mutate(self, chromosome: list[int]) -> list[int]:
        """
        **Mutates a chromosome based on the mutation rate**
        **Arguments**
            - chromosome
                - The chromosome to be mutated
        **Returns**
            - The mutated chromosome
        """
        mutated = chromosome.copy()
        for i in range(self.n_items):
            if random.random() < self.mutation_rate: #Mutation occurs based on mutation rate specifed from params
                mutated[i] = random.randint(1, self.n_bins) #Assign a new random bin to the item

        return mutated
    
    def run(self, seed: int = None) -> tuple[list[int], float, list[float], list[float]]:
        """
        **Runs the genetic algorithm to solve the Bin Packing Problem**

        **Arguments**
            - seed
                - An optional seed for random number generation for reproducibility
        **Returns**
            - best_chromosome
                - The best chromosome found
            - best_fitness
                - The fitness of the best chromosome
            - best_fitness_history
                - History of best fitness values over generation
            - avg_fitness_history
                - History of average fitness values over generation
        """
        if seed is not None: # Set seed for reproducibility comment out from completely random results
            random.seed(seed)
            np.random.seed(seed)

        self.population = self.create_population(self.pop_size, self.n_items, self.n_bins)
        
        self.evaluation_count = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

        # Evaluate initial population
        initial_fitnesses = []
        for solution in self.population:
            initial_fitnesses.append(self.fitness(solution))

        # Set initial evaluation count, will be incremented during the run, just counting initial evaluations here
        self.evaluation_count = len(self.population)

        # Track the best chromosome and fitness found so far
        best_index = np.argmax(initial_fitnesses)
        best_chromosome = self.population[best_index]
        best_fitness = initial_fitnesses[best_index]

        # Record initial best and average fitness
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(np.mean(initial_fitnesses))

        while self.evaluation_count < self.max_evaluations:
            
            # Initialize new population and fitness lists for the next generation
            new_population = []
            new_fitnesses = []

            # Carry over just the best chromosome
            elitism_indices = np.argsort(initial_fitnesses)[-self.elitism_count:]

            for i in elitism_indices:
                new_population.append(self.population[i].copy())
                new_fitnesses.append(initial_fitnesses[i])

            while len(new_population) < self.pop_size:
                # Generate new offspring until the new population is filled

                # Tournament Selection
                parent1 = self.tournament_selection(self.population, initial_fitnesses)
                parent2 = self.tournament_selection(self.population, initial_fitnesses)

                # Uniform Crossover
                child1, child2 = self.uniform_crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                if self.use_local_search and random.random() < self.local_search_rate:
                    # Apply hill climbing local search to child1
                    child1 = self.hill_climb(child1)
                    child2 = self.hill_climb(child2)

                if len(new_population) < self.pop_size:
                    # Add first child to new population
                    new_population.append(child1)
                    new_fitnesses.append(self.fitness(child1))
                    self.evaluation_count += 1 

                if len(new_population) < self.pop_size:
                    # Add second child to new population
                    new_population.append(child2)
                    new_fitnesses.append(self.fitness(child2)) 
                    self.evaluation_count += 1 
            
            # Update population and fitnesses for the next generation
            self.population = new_population
            initial_fitnesses = new_fitnesses
            
            # Update best chromosome and fitness found so far
            current_best_index = np.argmax(initial_fitnesses)
            current_best_fitness = initial_fitnesses[current_best_index]

            # Check if we have a new overall best
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = self.population[current_best_index].copy()

            # Record best and average fitness for this generation
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(initial_fitnesses))

            # Check if maximum evaluations reached, if so break out of loop
            if self.evaluation_count >= self.max_evaluations:
                break
            
        return best_chromosome.copy(), best_fitness, self.best_fitness_history, self.avg_fitness_history

def bpp_1() -> tuple[int, list[float], int]:
    """
    **Generates the first Bin Packing Problem instance**

    **Returns**
        - num_items
            - The number of items in the problem instance
        - item_weights
            - A list containing the weights of each item
        - num_bins
            - The number of bins available for packing
    """
    num_items = 500
    num_bins = 10

    # Generate item weights from 1 to 500, according to the problem definition, using indexes 1 to 500
    item_weights = []
    for i in range(1, num_items + 1):
        item_weights.append(i)

    return num_items, item_weights, num_bins


def bpp_2() -> tuple[int, list[float], int]:
    """
    **Generates the second Bin Packing Problem instance**

    **Returns**
        - num_items
            - The number of items in the problem instance
        - item_weights
            - A list containing the weights of each item
        - num_bins
            - The number of bins available for packing
    """
    num_items = 500
    num_bins = 50

    # Generate item weights for 500 items using the formula (i^2)/2 for i from 1 to 500, from the problem definition
    item_weights = []
    for i in range(1, num_items + 1):
        item_weights.append((i*i) / 2)
    
    return num_items, item_weights, num_bins

def run_bpps(local_search: bool) -> dict:
    """
    **Runs the genetic algorithm on predefined Bin Packing Problem instances with various parameter settings**

    **Returns**
        - A dictionary containing the results of each run
    """
    # Different Bin Packing Problem instances
    problem_types = {
        "BPP_1": bpp_1(),
        "BPP_2": bpp_2()
    }

    # Different parameter settings to test
    parameter_settings = [
        {"pop_size": 100, "mutation_rate": 0.01, "tournament_size": 3, "use_local_search": local_search, "local_search_rate": 0.1},
        {"pop_size": 100, "mutation_rate": 0.05, "tournament_size": 3, "use_local_search": local_search, "local_search_rate": 0.1},
        {"pop_size": 100, "mutation_rate": 0.01, "tournament_size": 7, "use_local_search": local_search, "local_search_rate": 0.1},
        {"pop_size": 100, "mutation_rate": 0.05, "tournament_size": 7, "use_local_search": local_search, "local_search_rate": 0.1},
    ]
    results = {}

    for problem_name, (n_items, item_weights, n_bins) in problem_types.items(): #iterate through problem types
        results[problem_name] = {}
        for setting_index, params in enumerate(parameter_settings): #iterate through parameter settings
            param_key = f"Population Size: {params['pop_size']} Mutation Rate: {params['mutation_rate']} Tournament Size: {params['tournament_size']}"
            results[problem_name][param_key] = [] #Initialize list to hold results for this setting

            print(f"Running {problem_name} with {param_key}")

            for trial in range(5):
                seed = trial + 42 #Different seed for each trial, comment to be completely random, but for reproducibility to paper results
                ga = BinPackingGA( #GA Instance, parameters are different for BPP Run 1 and 2
                    n_items=n_items, 
                    item_weights=item_weights,
                    n_bins=n_bins,
                    pop_size=params['pop_size'],
                    mutation_rate=params['mutation_rate'],
                    tournament_size=params['tournament_size'],
                    use_local_search=params.get('use_local_search', False),
                    local_search_rate=params.get('local_search_rate', 0.1)
                )

                # Run the GA
                best_chromosome, best_fitness, best_history, avg_history = ga.run(seed=seed)
                
                # Collect results for this trial
                trial_results = {
                    "trial": trial + 1,
                    "best_fitness": best_fitness,
                    "mean_fitness": avg_history[-1],
                    "best_fitnesses": best_history,
                    "avg_fitnesses": avg_history

                }

                results[problem_name][param_key].append(trial_results)

                print(f"Trial: {trial + 1}, Best Fitness: {best_fitness}, Mean Fitness: {avg_history[-1]}")
            
    return results

#Iterates through results to compute summary statistics
def get_analytics(results) -> dict:
    """
    **Processes the results of the GA runs to compute summary statistics**

    **Arguments**
        - results
            - The raw results from the GA runs
    **Returns**
        - A dictionary containing summary statistics for each problem and parameter setting
    """
    summary = {} #Initialize summary dictionary
    for problem, param_key in results.items():
        summary[problem] = {} #Initialize problem entry
        for setting, trials in results[problem].items(): #Iterate through parameter settings
            best_fitnesses = [trial['best_fitness'] for trial in trials]

            #Compute summary statistics from all five trials in each 
            summary[problem][setting] = {
                "best_fitness_mean": np.mean(best_fitnesses),
                "best_fitness_std": np.std(best_fitnesses),
                "best_fitness_max": max(best_fitnesses),
            }
    return summary

def convergence_comparison(problem_name: str, all_results: dict) -> None:
    """
    **Plots convergence graphs for different parameter settings for each given Bin Packing Problem**
    
    **Arguments**
        - problem_name
            - The name of the problem to plot results for
        - all_results
            - The complete results dictionary from the GA runs
    """
    plt.figure(figsize=(10, 6))
    for setting, trials in all_results[problem_name].items():
        all_histories = [trial['best_fitnesses'] for trial in trials]
        min_length = min(len(history) for history in all_histories)
        trimmed_histories = [history[:min_length] for history in all_histories]
        avg_best_fitness = np.mean(trimmed_histories, axis=0)
        evaluations = range(len(avg_best_fitness))
        plt.plot(evaluations, avg_best_fitness, label=setting)
    plt.title(f"Convergence Graph for {problem_name}")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.show()



#Function to pretty print the analytics
def print_table(analytics) -> None:
    """
    **Prints the summary statistics in a formatted table**

    **Arguments**
        - analytics
            - The summary statistics to be printed
    """
    for problem, settings in analytics.items():
        print(f"\n==================== {problem} ====================")
        print(f"{'Mutation':<10} {'TS':<4} {'Mean Fitness':<15} {'Std Dev':<15} {'Best Fitness':<15}")
        print("-" * 70)

        for setting, stats in settings.items():
            # Extract mutation rate and tournament size from key
            parts = setting.split()
            mutation = parts[5]               # value after "Mutation"
            ts = parts[8]                     # value after "Size:"
            
            #Print formatted statistics like the table in the conference paper
            print(f"{mutation:<10} {ts:<4} "
                  f"{float(stats['best_fitness_mean']):<15.6f} "
                  f"{float(stats['best_fitness_std']):<15.6f} "
                  f"{float(stats['best_fitness_max']):<15.6f}")
            

if __name__ == "__main__":
    results_standard = run_bpps(local_search=False)
    analytics_standard = get_analytics(results_standard)

    print("\n=== Running GA with Local Search ===")
    results_local_search = run_bpps(local_search=True)
    analytics_local_search = get_analytics(results_local_search)
    
    # Compare results
    print("\n=== STANDARD GA RESULTS ===")
    print_table(analytics_standard)
    
    print("\n=== GA WITH LOCAL SEARCH RESULTS ===")
    print_table(analytics_local_search)
    
    # Plot convergence for both
    for problem in results_standard.keys():
        print(f"\n=== Convergence for {problem} (Standard GA) ===")
        convergence_comparison(problem, results_standard)
        print(f"\n=== Convergence for {problem} (GA with Local Search) ===")
        convergence_comparison(problem, results_local_search)