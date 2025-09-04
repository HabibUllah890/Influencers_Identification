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

def run_hc(data_df, target_centroid, target_features,
           max_iterations,
           stopping_streak,
           optimize_group_size,
           N_results,
           sample_size=None):


    print(f"--- Running Hill Climbing (Optimizing Group Sum) ---")
    print(f"Params: MaxIter={max_iterations}, StopStreak={stopping_streak}, "
          f"OptimizeGroupSize={optimize_group_size}, N_Return={N_results}, SampleSize={sample_size}")
    start_time = time.time()

    if not isinstance(data_df, pd.DataFrame) or data_df.empty: print("Error: HC input data_df empty/invalid."); return pd.DataFrame()
    if not data_df.index.is_unique: print("Error: HC input data_df must have unique index."); return pd.DataFrame()
    if not isinstance(target_centroid, np.ndarray) or not isinstance(target_features, list): print("Error: HC target types invalid."); return pd.DataFrame()
    if target_centroid.ndim != 1 or len(target_centroid) != len(target_features): print(f"Error: HC target/features mismatch."); return pd.DataFrame()
    if not all(f in data_df.columns for f in target_features): print(f"Error: HC data_df missing features."); return pd.DataFrame()
    # Validate parameters passed from UI
    if not isinstance(max_iterations, int) or max_iterations <= 0: print(f"Error: Invalid HC max_iterations: {max_iterations}"); return pd.DataFrame()
    if not isinstance(stopping_streak, int) or stopping_streak <= 0: print(f"Error: Invalid HC stopping_streak: {stopping_streak}"); return pd.DataFrame()
    if not isinstance(optimize_group_size, int) or optimize_group_size <= 0: print(f"Error: Invalid HC optimize_group_size: {optimize_group_size}"); return pd.DataFrame()
    if not isinstance(N_results, int) or N_results <= 0: print(f"Error: Invalid HC N_results: {N_results}"); return pd.DataFrame()
    if sample_size is not None and (not isinstance(sample_size, int) or sample_size <= 0): print(f"Error: Invalid HC sample_size: {sample_size}"); return pd.DataFrame()

    available_indices = list(data_df.index)
    if sample_size and sample_size > 0 and sample_size < len(available_indices):
        print(f"HC running on a sample of size {sample_size}")
        available_indices = random.sample(available_indices, sample_size)
    elif sample_size and sample_size >= len(available_indices):
         print("Warning: HC sample_size >= data size, using all data.")

    if len(available_indices) < optimize_group_size: # Use the parameter here
        print(f"Error: Not enough available influencers ({len(available_indices)}) for optimize_group_size ({optimize_group_size}).")
        return pd.DataFrame()

    print("Precomputing individual fitness scores...")
    precompute_start = time.time()
    fitness_scores = {}
    valid_indices_for_calc = []
    try:
        relevant_data = data_df.loc[available_indices, target_features]
        for idx in available_indices:
            try:
                metrics = relevant_data.loc[idx]
                fitness_scores[idx] = calculate_fitness(metrics, target_centroid, target_features)
                valid_indices_for_calc.append(idx)
            except Exception as calc_err:
                fitness_scores[idx] = np.nan

        valid_fitness_scores = {idx: score for idx, score in fitness_scores.items() if pd.notna(score)}
        if not valid_fitness_scores: print("Error: Failed to calculate any valid fitness scores."); return pd.DataFrame()
        available_indices = list(valid_fitness_scores.keys())
        print(f"Precomputation finished for {len(available_indices)} influencers in {time.time() - precompute_start:.3f}s")
    except Exception as e: print(f"Error during fitness precomputation: {e}"); traceback.print_exc(); return pd.DataFrame()

    if len(available_indices) < optimize_group_size: # Use parameter
        print(f"Error: Not enough influencers with valid fitness ({len(available_indices)}) for optimize_group_size ({optimize_group_size}).")
        return pd.DataFrame()

    try:
        sorted_influencers = sorted(valid_fitness_scores.items(), key=lambda item: item[1], reverse=True)
        current_solution_indices = [idx for idx, score in sorted_influencers[:optimize_group_size]] # Use parameter
        current_score = sum(valid_fitness_scores[idx] for idx in current_solution_indices)
    except Exception as e: print(f"Error during HC initialization: {e}"); traceback.print_exc(); return pd.DataFrame()

    best_solution_indices = list(current_solution_indices)
    best_score = current_score
    print(f"Initial HC group score (sum of fitness): {best_score:.6f}")

    no_improve_streak = 0
    iteration = 0

    loop_start_time = time.time()
    for iteration in range(max_iterations): # Use parameter
        neighbor_indices = list(current_solution_indices)
        if not neighbor_indices: break

        replace_pos = random.randrange(len(neighbor_indices))
        index_to_replace = neighbor_indices[replace_pos]

        attempts = 0; max_attempts = len(available_indices) * 2
        new_candidate_index = None; current_solution_set = set(neighbor_indices)
        while attempts < max_attempts:
            potential_candidate = random.choice(available_indices)
            if potential_candidate not in current_solution_set:
                new_candidate_index = potential_candidate; break
            attempts += 1

        if new_candidate_index is None:
            no_improve_streak += 1
            if no_improve_streak >= stopping_streak: print(f"Stopping HC at iter {iteration + 1}: No improvement streak (swap failed)."); break # Use parameter
            continue

        neighbor_indices[replace_pos] = new_candidate_index
        try: neighbor_score = sum(valid_fitness_scores[idx] for idx in neighbor_indices)
        except KeyError as e: print(f"Warning: KeyError calculating neighbor score (index {e} missing?). Skipping."); continue

        if neighbor_score > best_score:
            best_solution_indices = list(neighbor_indices)
            best_score = neighbor_score
            current_solution_indices = list(neighbor_indices)
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        if no_improve_streak >= stopping_streak: # Use parameter
            print(f"Stopping HC at iter {iteration + 1}: No improvement streak limit ({stopping_streak}) reached.")
            break

    print(f"HC Loop finished after {iteration + 1} iterations (Loop time: {time.time() - loop_start_time:.2f}s)")
    print(f"--- Hill Climbing Finished. Best group score (sum): {best_score:.6f} ---")

    if not best_solution_indices: print("Error: HC ended with no best solution."); return pd.DataFrame()

    try:
        result_df = data_df.loc[best_solution_indices].copy()
        fitness_col_name = 'individual_fitness' # Consistent naming
        result_df[fitness_col_name] = result_df.index.map(valid_fitness_scores)
        rank_col_name = 'HC Rank'
        result_df[rank_col_name] = result_df[fitness_col_name].rank(method='first', ascending=False, na_option='bottom').astype('Int64')
        final_ranked_df = result_df.sort_values(rank_col_name).head(N_results) # Use parameter

    except Exception as e: print(f"Error preparing HC output DataFrame: {e}"); traceback.print_exc(); return pd.DataFrame()

    end_time = time.time()
    print(f"--- HC Returning Top {len(final_ranked_df)} influencers. Total Time = {end_time - start_time:.2f}s ---")

    return final_ranked_df

