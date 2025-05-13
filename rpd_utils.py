import random
import math
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd

# --- Game Constants ---
COOPERATE = 0
DEFECT = 1

# Payoff matrix: (my_payoff, opponent_payoff)
# R (Reward), S (Sucker), T (Temptation), P (Punishment)
# Standard payoffs: T > R > P > S and 2R > T + S
PAYOFF_MATRIX = {
    (COOPERATE, COOPERATE): (3, 3),  # Both cooperate (Reward)
    (COOPERATE, DEFECT):   (0, 5),  # I cooperate, opponent defects (Sucker)
    (DEFECT, COOPERATE):   (5, 0),  # I defect, opponent cooperates (Temptation)
    (DEFECT, DEFECT):     (1, 1)   # Both defect (Punishment)
}

GENOME_LENGTH = 19  # 16 for history states + 3 for initial moves

# --- Individual Representation ---
class Individual:
    def __init__(self, genome=None, id_num=0):
        if genome is None:
            self.genome = np.random.rand(GENOME_LENGTH) # initialize with random genome if not provided
        else:
            self.genome = np.array(genome)
        self.fitness = 0.0
        self.avg_score_per_round = 0.0
        self.total_score = 0.0
        self.games_played = 0
        self.species_id = None
        self.id = id_num

    def __repr__(self):
        return f"Ind(ID:{self.id}, Fit:{self.fitness:.2f}, Spc:{self.species_id}, Genome_snippet:{self.genome[:3]}...)"

# --- Game Mechanics ---
def get_history_index(my_moves, opp_moves):
    """
    Converts the last two moves of self and opponent into an index for the genome.
    Moves: COOPERATE (0), DEFECT (1)
    History: (my_t-1, my_t-2, opp_t-1, opp_t-2)
    Example: (C,C,C,C) -> (0,0,0,0) -> index 0
             (D,D,D,D) -> (1,1,1,1) -> index 15
    """
    m1, m2 = my_moves[-1], my_moves[-2]
    o1, o2 = opp_moves[-1], opp_moves[-2]
    # Binary to integer: (m2)(m1)(o2)(o1)
    binary_str = f"{m2}{m1}{o2}{o1}"
    index = int(binary_str, 2)
    return index

def get_move(individual: Individual, my_moves, opp_moves):
    """
    Determines the individual's next move based on its genome and game history.
    my_moves and opp_moves are lists of past moves, most recent last.
    
    The first 4 genes in the genome now represent initial moves for these scenarios:
    [0]: First move (no history)
    [1]: My second move when opponent's first move was COOPERATE
    [2]: My second move when opponent's first move was DEFECT
    """
    if len(my_moves) == 0:
        # First move - use gene 0
        prob_cooperate = individual.genome[0]
        return COOPERATE if random.random() < prob_cooperate else DEFECT
    elif len(my_moves) == 1:
        # Second move - use gene 1 or 2 based on opponent's first move
        opp_first_move = opp_moves[0]
        gene_index = 1 if opp_first_move == COOPERATE else 2
        prob_cooperate = individual.genome[gene_index]
        return COOPERATE if random.random() < prob_cooperate else DEFECT
    else:
        # Use the strategy genome (offset by 4 for the initial moves)
        history_index = get_history_index(my_moves, opp_moves)
        prob_cooperate = individual.genome[history_index + 3]  # +3 offset for initial moves
        return COOPERATE if random.random() < prob_cooperate else DEFECT

def play_game(ind1: Individual, ind2: Individual, num_rounds: int):
    """Plays a game of RPD between two individuals."""
    p1_moves = []
    p2_moves = []
    p1_total_score = 0
    p2_total_score = 0

    for _ in range(num_rounds):
        move1 = get_move(ind1, p1_moves, p2_moves)
        move2 = get_move(ind2, p2_moves, p1_moves)

        payoff1, payoff2 = PAYOFF_MATRIX[(move1, move2)]

        p1_total_score += payoff1
        p2_total_score += payoff2

        p1_moves.append(move1)
        p2_moves.append(move2)

    return p1_total_score, p2_total_score

# --- GA Core Functions ---
def initialize_population(pop_size: int):
    """Creates an initial population of random individuals."""
    return [Individual(id_num=i) for i in range(pop_size)]

def calculate_fitnesses(population, num_opponents_to_play: int, num_rounds_per_game: int):
    """
    Calculates fitness for each individual in the population.
    Each individual plays against a random subset of other individuals.
    Fitness is the average score per round achieved by the individual.
    """
    pop_size = len(population)

    for i, ind in enumerate(population):
        ind.fitness = 0  # Reset fitness (will be adjusted by speciation later)
        ind.avg_score_per_round = 0
        ind.total_score = 0
        ind.games_played = 0

        if pop_size <= num_opponents_to_play: # Play against all others if pool is small
            opponent_indices = [j for j in range(pop_size) if j != i]
        else:
            opponent_indices = random.sample([j for j in range(pop_size) if j != i], num_opponents_to_play)
        

        if not opponent_indices and pop_size >= 1:
            ind.avg_score_per_round = 0 # No one to play against
            ind.fitness = 0
            continue


        for opp_idx in opponent_indices:
            opponent = population[opp_idx]
            
            # ind plays as player 1
            score_ind, score_opp = play_game(ind, opponent, num_rounds_per_game)
            ind.total_score += score_ind
            opponent.total_score += score_opp
            ind.games_played += 1
            opponent.games_played += 1
    
    for i, ind in enumerate(population):
        if ind.games_played > 0:
            ind.avg_score_per_round = ind.total_score / (ind.games_played * num_rounds_per_game)
        else: # Only one individual in population or no opponents selected
            ind.avg_score_per_round = 0.0
            
        ind.fitness = ind.avg_score_per_round # Raw fitness before speciation adjustment


# --- Genetic Operators ---
def tournament_selection(population, k=3):
    """Selects an individual using tournament selection."""
    if not population: return None # Handle empty population case
    tournament = random.sample(population, k)
    return max(tournament, key=lambda ind: ind.fitness) # Fitness used here is adjusted fitness


def blend_crossover(parent1: Individual, parent2: Individual, alpha: float):
    """Perform blend crossover"""
    child_genome = np.zeros(GENOME_LENGTH)

    u = random.uniform(0, 1)
    gamma = (1- 2*alpha) * u - alpha

    for i in range(GENOME_LENGTH):
        new_val = parent1.genome[i] * (1 - gamma) + parent2.genome[i] * gamma
        child_genome[i] = np.clip(new_val, 0.0, 1.0)
    
    return child_genome


def gaussian_mutation(genome, mutation_rate: float, mutation_strength=0.1):
    """Applies Gaussian mutation to each gene in the genome."""
    mutated_genome = np.copy(genome)
    for i in range(GENOME_LENGTH):
        if random.random() < mutation_rate:
            mutation = random.gauss(0, mutation_strength)
            mutated_genome[i] += mutation
            mutated_genome[i] = np.clip(mutated_genome[i], 0.0, 1.0) # Keep within [0,1]
    return mutated_genome

# --- Speciation ---
def genomic_distance(ind1: Individual, ind2: Individual, c1=1.0, c2=1.0, c3=0.4):
    """
    Calculates the genomic distance between two individuals.
    This is a simple Euclidean distance metric.
    """
    # Simple Euclidean distance
    return np.linalg.norm(ind1.genome - ind2.genome)

def speciate_population(population, compatibility_threshold: float):
    """
    Assigns individuals to species based on genomic distance.
    Fitness is then shared among members of the same species.
    Returns species representatives and updates individual species_id and fitness.
    """
    if not population:
        return [], {}

    species_representatives = []
    species_members = {} # key: species_id (rep_id), value: list of members

    for ind in population:
        ind.species_id = None
        found_species = False
        # Try to assign to an existing species
        for rep_idx, rep in enumerate(species_representatives):
            if genomic_distance(ind, rep) < compatibility_threshold:
                ind.species_id = rep.id # Use rep's unique ID as species ID
                if rep.id not in species_members:
                    species_members[rep.id] = []
                species_members[rep.id].append(ind)
                found_species = True
                break
        
        # If not assigned, this individual becomes a new representative
        if not found_species:
            # Use the individual's own ID as its new species ID
            # (Need to ensure individual IDs are unique across generations or use a species counter)
            # For simplicity, let's make species_id based on the representative's id.
            ind.species_id = ind.id 
            species_representatives.append(deepcopy(ind)) # Store a copy as representative
            species_members[ind.id] = [ind]

    # Adjust fitness using sharing (Sharing an individual's raw fitness among its species members)
    # ind.fitness will store the adjusted fitness
    for spec_id, members in species_members.items():
        if not members: continue
        species_size = len(members)
        for member in members:
            # Fitness was already set to avg_score_per_round
            # Now adjust it by species size
            if species_size > 0:
                member.fitness = member.avg_score_per_round / species_size
            else: # Should not happen if members is not empty
                member.fitness = 0 
                
    return species_representatives, species_members


# --- Population Management ---
def create_next_generation(population, elite_size: int, tournament_k: int,
                           mutation_rate: float, mutation_strength: float,
                           next_ind_id_start: int, crossover_alpha=None):
    """
    Creates the next generation of individuals.
    Assumes population individuals have their (potentially shared) fitness calculated.
    """
    pop_size = len(population)
    new_population = []
    
    current_ind_id = next_ind_id_start

    # Elitism: Carry over the best individuals
    # Sort by raw score before fitness sharing for elitism, or by shared fitness?
    # Standard elitism usually uses the fitness that selection uses.
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    elites = [deepcopy(population[i]) for i in range(min(elite_size, pop_size))]
    for elite_ind in elites: # Reset ID for new population tracking if desired, or keep old
        elite_ind.id = current_ind_id 
        current_ind_id +=1
    new_population.extend(elites)

    # Fill the rest of the population using selection, crossover, and mutation
    while len(new_population) < pop_size:
        parent1 = tournament_selection(population, k=tournament_k)
        parent2 = tournament_selection(population, k=tournament_k)

        child_genome = blend_crossover(parent1, parent2, crossover_alpha)

        mutated_child_genome = gaussian_mutation(child_genome, mutation_rate, mutation_strength)

        child = Individual(genome=mutated_child_genome, id_num=current_ind_id)
        current_ind_id+=1
        if len(new_population) < pop_size:
            new_population.append(child)
            
    return new_population[:pop_size], current_ind_id # Ensure correct pop size and return next available ID


# --- Helper for Diversity ---
def calculate_population_diversity(population):
    if len(population) < 2:
        return 0.0
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dist = genomic_distance(population[i], population[j])
            distances.append(dist)
    return np.mean(distances) if distances else 0.0

def run_evolution(population_size, num_generations, num_opponents_to_play,
                   num_rounds_per_game, elite_size, tournament_k,
                   mutation_rate, mutation_strength,crossover_alpha,
                   compatibility_threshold,
                   verbose=False, population=None):
    """
    Main loop for running the evolutionary algorithm.
    """
    population_history = []
    avg_raw_score_history = []
    best_raw_score_history = []
    num_species_history = []
    population_diversity_history = []
    avg_fitness_history = []

    if population is None:
        population = initialize_population(population_size)

    for generation in range(num_generations):

        # Calculate fitnesses
        calculate_fitnesses(population, num_opponents_to_play, num_rounds_per_game)
        
        # Speciate population
        species_representatives, species_members = speciate_population(population, compatibility_threshold)
        
        # Create next generation
        next_ind_id_start = max(ind.id for ind in population) + 1
        population, next_ind_id_start = create_next_generation(
            population, elite_size, tournament_k, mutation_rate,
            mutation_strength, next_ind_id_start, crossover_alpha
        )

        population.sort(key=lambda ind: ind.avg_score_per_round, reverse=True)

        avg_raw_score = np.mean([ind.avg_score_per_round for ind in population])
        best_raw_score = max(ind.avg_score_per_round for ind in population)
        num_species = len(species_representatives)
        diversity = calculate_population_diversity(population)
        avg_fitness = np.mean([ind.fitness for ind in population])

        population_history.append(population)
        avg_raw_score_history.append(avg_raw_score)
        best_raw_score_history.append(best_raw_score)
        num_species_history.append(num_species)
        population_diversity_history.append(diversity)
        avg_fitness_history.append(avg_fitness)

            
        if verbose and (generation % verbose == 0 or generation == num_generations - 1):
            print(f"Generation {generation + 1}/{num_generations}")
            print(f"Avg Raw Score: {avg_raw_score:.2f}, Best Raw Score: {best_raw_score:.2f}")
            print(f"Avg Fitness: {avg_fitness:.2f}, Num Species: {num_species}")
            print(f"Population Diversity: {diversity:.2f}")
    return population_history, avg_raw_score_history, best_raw_score_history, avg_fitness_history, num_species_history, population_diversity_history

def _run_evolution_island(population, population_size, num_generations, num_opponents_to_play,
                   num_rounds_per_game, elite_size, tournament_k,
                   mutation_rate, mutation_strength,crossover_alpha,
                   compatibility_threshold,
                   verbose=False):
    return run_evolution(population_size, num_generations, num_opponents_to_play,
                   num_rounds_per_game, elite_size, tournament_k,
                   mutation_rate, mutation_strength,crossover_alpha,
                   compatibility_threshold, verbose=verbose, population=population)

def run_island_model(population_size, num_generations, num_opponents_to_play,
                   num_rounds_per_game, elite_size, tournament_k,
                   mutation_rate, mutation_strength,crossover_alpha,
                   compatibility_threshold, migration_frequency, island_count, num_migrants,
                   verbose=False, processes=-1):
    

    
    partial_run_evolution = partial(_run_evolution_island, population_size=population_size,
                                    num_generations=migration_frequency,
                                    num_opponents_to_play=num_opponents_to_play,
                                    num_rounds_per_game=num_rounds_per_game,
                                    elite_size=elite_size,
                                    tournament_k=tournament_k,
                                    mutation_rate=mutation_rate,
                                    mutation_strength=mutation_strength,
                                    crossover_alpha=crossover_alpha,
                                    compatibility_threshold=compatibility_threshold,
                                    verbose=False)
    
    global_population_list = [[]] * island_count

    if processes == -1:
        processes = cpu_count()
    
    max_processes = min(processes, population_size)

    with Pool(processes=max_processes) as pool:
        population_list = [None] * island_count
        island_results = pool.map(partial_run_evolution, population_list)
        populations, _, _, _, _, _ = zip(*island_results)
        populations = list(populations)
        last_populations = [pop[-1] for pop in populations]
        for i in range(island_count):
            global_population_list[i].append(populations[i])
    
    if verbose:
        print("Migration 1...")

    for i in range(island_count):
        best_agents = populations[i][:num_migrants]
        populations[(i + 1) % island_count] += best_agents
        populations[i] = populations[i][num_migrants:]

    for i in range(migration_frequency, num_generations, migration_frequency):
        with Pool(max_processes) as pool:
            island_results = pool.map(partial_run_evolution, last_populations)
            populations, _, _, _, _, _ = zip(*island_results)
            populations = list(populations)
            last_populations = [pop[-1] for pop in populations]
            for j in range(island_count):
                global_population_list[j].append(populations[j])
        
        if verbose:
            migration_number = i // migration_frequency + 1
            print(f'Migration {migration_number}...')
        
        # Migrate the best genome from each island to the next
        for j in range(island_count):
            best_agents = last_populations[j][:num_migrants]
            last_populations[(j + 1) % island_count] += best_agents
            last_populations[j] = last_populations[j][num_migrants:]
    
    return global_population_list