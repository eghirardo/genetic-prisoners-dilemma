import random
import math
import numpy as np
from copy import deepcopy

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

GENOME_LENGTH = 20  # 16 for history states + 4 for initial moves

# --- Individual Representation ---
class Individual:
    def __init__(self, genome=None, id_num=0):
        if genome is None:
            self.genome = np.random.rand(GENOME_LENGTH)
        else:
            self.genome = np.array(genome)
        self.fitness = 0.0
        self.avg_score_per_round = 0.0 # More direct measure from RPD
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
    # Binary to integer: (m2)(m1)(o2)(o1) - using common RPD notation order for history bits
    # Let's define an order, e.g., My_t-1, My_t-2, Opp_t-1, Opp_t-2
    # index = m1 * 8 + m2 * 4 + o1 * 2 + o2 * 1
    # Or, more standard way: (my_prev1, opp_prev1, my_prev2, opp_prev2) could also be a convention
    # The paper "Evolving C-Max Strategies" uses (OwnPrev, OppPrev, OwnPrev-1, OppPrev-1)
    # Let's use: (My_Prev1, My_Prev2, Opp_Prev1, Opp_Prev2)
    # (0,0,0,0) -> index 0
    # (1,1,1,1) -> index 15
    # My_t-1 = my_moves[-1], My_t-2 = my_moves[-2]
    # Opp_t-1 = opp_moves[-1], Opp_t-2 = opp_moves[-2]
    index = (my_moves[-1] << 3) | (my_moves[-2] << 2) | \
            (opp_moves[-1] << 1) | opp_moves[-2]
    return index

def get_move(individual: Individual, my_moves, opp_moves):
    """
    Determines the individual's next move based on its genome and game history.
    my_moves and opp_moves are lists of past moves, most recent last.
    
    The first 4 genes in the genome now represent initial moves for these scenarios:
    [0]: First move (no history)
    [1]: My second move when opponent's first move was COOPERATE
    [2]: My second move when opponent's first move was DEFECT
    [3]: Reserved for future use (could be used for 3rd move with limited history)
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
        prob_cooperate = individual.genome[history_index + 4]  # +4 offset for initial moves
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
    if pop_size <= 1:
        for ind in population:
            ind.fitness = 0
            ind.avg_score_per_round = 0
        return

    for i, ind in enumerate(population):
        ind.fitness = 0  # Reset fitness (will be adjusted by speciation later)
        ind.avg_score_per_round = 0
        total_score_for_ind = 0
        games_played = 0

        if pop_size <= num_opponents_to_play: # Play against all others if pool is small
            opponent_indices = [j for j in range(pop_size) if j != i]
        else:
            opponent_indices = random.sample([j for j in range(pop_size) if j != i], num_opponents_to_play)
        
        if not opponent_indices and pop_size > 1: # Should not happen if num_opponents_to_play > 0
             # This case can occur if num_opponents_to_play is 0 or population is tiny.
             # Assign a baseline fitness or handle as an error.
             # For now, if no opponents, fitness remains 0.
            pass
        elif not opponent_indices and pop_size == 1:
            ind.avg_score_per_round = 0 # No one to play against
            ind.fitness = 0
            continue


        for opp_idx in opponent_indices:
            opponent = population[opp_idx]
            
            # ind plays as player 1
            score_ind, _ = play_game(ind, opponent, num_rounds_per_game)
            total_score_for_ind += score_ind
            games_played += 1
            
            # ind plays as player 2 (optional, but good for symmetry if strategies are asymmetric)
            # _, score_ind_as_p2 = play_game(opponent, ind, num_rounds_per_game)
            # total_score_for_ind += score_ind_as_p2
            # games_played += 1


        if games_played > 0:
            ind.avg_score_per_round = total_score_for_ind / (games_played * num_rounds_per_game)
        else: # Only one individual in population or no opponents selected
            ind.avg_score_per_round = 0.0 
        
        ind.fitness = ind.avg_score_per_round # Raw fitness before speciation adjustment


# --- Genetic Operators ---
def tournament_selection(population, k=3):
    """Selects an individual using tournament selection."""
    if not population: return None # Handle empty population case
    tournament = random.sample(population, k)
    return max(tournament, key=lambda ind: ind.fitness) # Fitness used here is adjusted fitness

def uniform_crossover(parent1: Individual, parent2: Individual, crossover_rate: float):
    """Performs uniform crossover."""
    if random.random() > crossover_rate:
        return deepcopy(parent1.genome), deepcopy(parent2.genome) # No crossover

    child1_genome = np.copy(parent1.genome)
    child2_genome = np.copy(parent2.genome)
    for i in range(GENOME_LENGTH):
        if random.random() < 0.5:
            child1_genome[i] = parent2.genome[i]
            child2_genome[i] = parent1.genome[i]
    return child1_genome, child2_genome

def blx_alpha_crossover(parent1: Individual, parent2: Individual, crossover_rate: float, alpha=0.5):
    """Performs BLX-alpha crossover."""
    if random.random() > crossover_rate:
        return deepcopy(parent1.genome), deepcopy(parent2.genome)

    child1_genome = np.zeros(GENOME_LENGTH)
    child2_genome = np.zeros(GENOME_LENGTH) # Not always generated by BLX-alpha, but we can make two
    
    for i in range(GENOME_LENGTH):
        p1_gene = parent1.genome[i]
        p2_gene = parent2.genome[i]
        
        d = abs(p1_gene - p2_gene)
        min_val = min(p1_gene, p2_gene) - alpha * d
        max_val = max(p1_gene, p2_gene) + alpha * d
        
        # Child 1
        rand_val1 = random.uniform(min_val, max_val)
        child1_genome[i] = np.clip(rand_val1, 0.0, 1.0) # Probabilities must be [0,1]
        
        # Child 2 (optional, can be different random value in range)
        rand_val2 = random.uniform(min_val, max_val)
        child2_genome[i] = np.clip(rand_val2, 0.0, 1.0)
        
    return child1_genome, child2_genome


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
    Calculates genomic distance based on NEAT's formula (simplified for continuous genomes).
    Here, we'll use Euclidean distance for simplicity as genes are homologous.
    The NEAT coefficients (c1, c2, c3 for excess, disjoint, avg weight diffs) are
    less directly applicable. We can just use Euclidean distance or weighted Euclidean.
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
                           crossover_fn, crossover_rate: float,
                           mutation_fn, mutation_rate: float, mutation_strength: float,
                           next_ind_id_start: int, crossover_alpha=None): # crossover_alpha for BLX
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
        
        if parent1 is None or parent2 is None: # Population too small or all same fitness
            # Fallback: if selection fails, just pick random individuals or break
            if not population: break 
            parent1 = random.choice(population) if population else None
            parent2 = random.choice(population) if population else parent1
            if not parent1: break


        if crossover_alpha is not None: # For BLX-alpha
             child1_genome, child2_genome = crossover_fn(parent1, parent2, crossover_rate, alpha=crossover_alpha)
        else: # For uniform crossover
             child1_genome, child2_genome = crossover_fn(parent1, parent2, crossover_rate)

        mutated_child1_genome = mutation_fn(child1_genome, mutation_rate, mutation_strength)
        mutated_child2_genome = mutation_fn(child2_genome, mutation_rate, mutation_strength)

        child1 = Individual(genome=mutated_child1_genome, id_num=current_ind_id)
        current_ind_id+=1
        if len(new_population) < pop_size:
            new_population.append(child1)
        
        if len(new_population) < pop_size:
            child2 = Individual(genome=mutated_child2_genome, id_num=current_ind_id)
            current_ind_id+=1
            new_population.append(child2)
            
    return new_population[:pop_size], current_ind_id # Ensure correct pop size and return next available ID


# --- Helper for Diversity ---
def calculate_population_diversity(population):
    if len(population) < 2:
        return 0.0
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dist = rpd.genomic_distance(population[i], population[j])
            distances.append(dist)
    return np.mean(distances) if distances else 0.0

def run_evolution(population_size, num_generations, num_opponents_to_play,
                   num_rounds_per_game, elite_size, tournament_k,
                   crossover_rate, mutation_rate, mutation_strength,
                   compatibility_threshold, crossover_fn, mutation_fn,
                   crossover_alpha=None, population=None, verbose=False):
    """
    Main loop for running the evolutionary algorithm.
    """
    if population == None:
        population = initialize_population(population_size)

    for generation in range(num_generations):

        # Calculate fitnesses
        calculate_fitnesses(population, num_opponents_to_play, num_rounds_per_game)
        
        # Speciate population
        species_representatives, species_members = speciate_population(population, compatibility_threshold)
        
        # Create next generation
        next_ind_id_start = max(ind.id for ind in population) + 1
        population, next_ind_id_start = create_next_generation(
            population, elite_size, tournament_k,
            crossover_fn, crossover_rate,
            mutation_fn, mutation_rate, mutation_strength,
            next_ind_id_start, crossover_alpha=crossover_alpha
        )

        if verbose and generation % verbose ==0:
            print(f"Generation {generation + 1}/{num_generations}")
        # Optional: Print diversity of the population
        diversity = calculate_population_diversity(population)
        print(f"Population Diversity: {diversity:.4f}")

    return population