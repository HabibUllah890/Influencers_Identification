
import random
import time
import numpy as np
import pandas as pd
import traceback

try:
    from .fitness import calculate_fitness
except ImportError:
    try:
        from fitness import calculate_fitness
    except ImportError:
        try:
            from algorithms.fitness import calculate_fitness
        except ImportError:
            print("ERROR: Could not import calculate_fitness. Ensure fitness.py is accessible.")
            def calculate_fitness(metrics, target_centroid, target_features): return 0.0


def run_ga(data_df, target_centroid, target_features,
           population_size,
           generations,
           mutation_rate,
           crossover_rate,
           elitism_count,
           group_size,
           N_results):

    print(f"--- Running GA (Final Pop Ranking Method) ---")
    print(f"Params: Pop={population_size}, Gens={generations}, MutRate={mutation_rate:.3f}, "
          f"XRate={crossover_rate:.3f}, Elites={elitism_count}, GroupSize={group_size}, N_Return={N_results}")
    start_time = time.time()

    if not isinstance(data_df, pd.DataFrame) or data_df.empty: print("Error: GA input data_df empty/invalid."); return pd.DataFrame()
    if not data_df.index.is_unique: print("Error: GA input data_df must have unique index."); return pd.DataFrame()
    if not isinstance(target_centroid, np.ndarray) or not isinstance(target_features, list): print("Error: GA target types invalid."); return pd.DataFrame()
    if target_centroid.ndim != 1 or len(target_centroid) != len(target_features): print(f"Error: GA target/features mismatch."); return pd.DataFrame()
    if not all(f in data_df.columns for f in target_features): print(f"Error: GA data_df missing features."); return pd.DataFrame()
    if not isinstance(population_size, int) or population_size <= 0: print(f"Error: Invalid GA population_size: {population_size}"); return pd.DataFrame()
    if not isinstance(generations, int) or generations <= 0: print(f"Error: Invalid GA generations: {generations}"); return pd.DataFrame()
    if not isinstance(mutation_rate, (int, float)) or not (0.0 <= mutation_rate <= 1.0): print(f"Error: Invalid GA mutation_rate: {mutation_rate}"); return pd.DataFrame()
    if not isinstance(crossover_rate, (int, float)) or not (0.0 <= crossover_rate <= 1.0): print(f"Error: Invalid GA crossover_rate: {crossover_rate}"); return pd.DataFrame()
    if not isinstance(elitism_count, int) or not (0 <= elitism_count < population_size): print(f"Error: GA elitism_count invalid for pop size."); return pd.DataFrame()
    if not isinstance(group_size, int) or group_size <= 0: print(f"Error: Invalid GA group_size: {group_size}"); return pd.DataFrame()
    if not isinstance(N_results, int) or N_results <= 0: print(f"Error: Invalid GA N_results: {N_results}"); return pd.DataFrame()


    influencer_indices = list(data_df.index)
    if len(influencer_indices) < group_size: print(f"Error: Not enough influencers for group size {group_size}."); return pd.DataFrame()


    def calculate_group_fitness(individual_indices):
        valid_indices = [idx for idx in individual_indices if idx in data_df.index]
        if not valid_indices: return 0.0
        try:
            group_metrics = data_df.loc[valid_indices, target_features]
            scores = group_metrics.apply(lambda row: calculate_fitness(row, target_centroid, target_features), axis=1)
            valid_scores = scores.dropna(); return valid_scores.mean() if not valid_scores.empty else 0.0
        except Exception as e: print(f"Error calculating group fitness: {e}"); return 0.0

    def initialize_population():
        population = [];
        if not influencer_indices: return []
        for _ in range(population_size): population.append(random.sample(influencer_indices, group_size))
        return population

    def selection(pop, fitnesses, num_to_select):
        if not pop or not fitnesses or len(pop) != len(fitnesses) or num_to_select <= 0: return []
        try:
            paired = list(zip(pop, fitnesses)); paired.sort(key=lambda x: x[1], reverse=True)
            return [ind for ind, fit in paired[:num_to_select]]
        except Exception as e: print(f"Error during selection: {e}"); return []

    def crossover(parent1, parent2):
        combined = list(set(parent1) | set(parent2)); num_combined = len(combined)
        if num_combined < group_size:
            num_missing = group_size - num_combined
            available = [idx for idx in influencer_indices if idx not in combined]
            num_to_sample = min(num_missing, len(available))
            if num_to_sample > 0: combined.extend(random.sample(available, num_to_sample))
        random.shuffle(combined); return combined[:min(len(combined), group_size)]

    def mutate(individual_indices):
        if random.random() > mutation_rate or len(individual_indices) == 0: return list(individual_indices)
        mutated = list(individual_indices); pos = random.randrange(len(mutated)); orig = mutated[pos]
        attempts = 0; max_attempts = len(influencer_indices); current_set = set(mutated)
        while attempts < max_attempts:
            new = random.choice(influencer_indices)
            if new != orig and new not in current_set: mutated[pos] = new; return mutated
            attempts += 1
        return list(individual_indices)

    #  GA Main Loop
    population = initialize_population()
    if not population: print("Error: GA Population initialization failed."); return pd.DataFrame()
    best_group_fitness_overall = -float('inf')
    for gen in range(generations): # Uses parameter
        fitnesses = [calculate_group_fitness(ind) for ind in population]
        valid_fitnesses = [f for f in fitnesses if isinstance(f, (int, float)) and pd.notna(f)]
        if not valid_fitnesses: print(f"Warning: No valid fitness in gen {gen+1}. Stopping."); break
        current_best_group_fitness = max(valid_fitnesses)
        if current_best_group_fitness > best_group_fitness_overall: best_group_fitness_overall = current_best_group_fitness
        if gen % 10 == 0 or gen == generations - 1: print(f"Gen {gen + 1}/{generations}: Best avg group fitness = {current_best_group_fitness:.5f}")
        parents = selection(population, fitnesses, population_size)
        if not parents: print("Warning: Selection failed. Stopping."); break
        next_population = []; elites = selection(population, fitnesses, elitism_count); next_population.extend(elites)
        num_to_generate = population_size - len(next_population); parent_pool = parents
        for _ in range(num_to_generate):
            if len(parent_pool) >= 2: p1, p2 = random.sample(parent_pool, 2)
            elif len(parent_pool) == 1: p1 = p2 = parent_pool[0]
            else: p1 = p2 = random.choice(population)
            if random.random() < crossover_rate: child = crossover(p1, p2)
            else: child = list(random.choice([p1, p2]))
            child = mutate(child)
            if len(child) == group_size: next_population.append(child)
        while len(next_population) < population_size: next_population.append(random.choice(parents) if parents else random.choice(population))
        population = next_population

    print("--- GA Finished Evolution ---")
    print(f"Final best avg group fitness found: {best_group_fitness_overall:.5f}")
    unique_indices = set(); [unique_indices.update(group) for group in population]
    if not unique_indices: print("Error: No unique individuals in final pop."); return pd.DataFrame()
    print(f"Found {len(unique_indices)} unique individuals in final population.")
    try:
        valid_unique_indices = [idx for idx in unique_indices if idx in data_df.index]
        if not valid_unique_indices: print("Error: No valid unique indices."); return pd.DataFrame()
        all_unique_df = data_df.loc[list(valid_unique_indices)].copy()
    except Exception as e: print(f"Error selecting unique data: {e}"); traceback.print_exc(); return pd.DataFrame()
    print("Calculating individual fitness for unique results...")
    individual_fitness_col = 'individual_fitness'
    try:
        all_unique_df[individual_fitness_col] = all_unique_df.apply(lambda row: calculate_fitness(row[target_features], target_centroid, target_features), axis=1)
        if all_unique_df[individual_fitness_col].isnull().any(): print("Warning: NaNs in calculated fitness.")
    except Exception as e: print(f"Error calculating individual fitness: {e}"); traceback.print_exc(); return all_unique_df
    print("Ranking unique individuals...")
    rank_col = 'GA Rank';
    try: all_unique_df[rank_col] = all_unique_df[individual_fitness_col].rank(method='first', ascending=False, na_option='bottom').astype('Int64')
    except Exception as e: print(f"Error ranking individuals: {e}"); traceback.print_exc(); return all_unique_df
    try: final_ranked_df = all_unique_df.sort_values(rank_col).head(N_results) # Uses parameter
    except Exception as e: print(f"Error sorting/selecting top results: {e}"); return all_unique_df
    end_time = time.time()
    print(f"--- GA Post-processing Finished: Returning Top {len(final_ranked_df)}. Total Time = {end_time - start_time:.2f}s ---")
    return final_ranked_df

