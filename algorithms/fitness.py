import pandas as pd
import numpy as np
import traceback

def calculate_fitness(influencer_metrics, centroid_vector, features_list):


    if centroid_vector is None or features_list is None or not isinstance(features_list, list):
        return 0.0
    if not isinstance(influencer_metrics, (pd.Series, dict)):
         return 0.0
    if not features_list:
        return 0.0

    if len(centroid_vector) != len(features_list):
        return 0.0

    try:
        influencer_features = []
        missing_or_invalid_feature = False
        for feature in features_list:
            value = influencer_metrics.get(feature)
            if value is None or pd.isna(value):
                missing_or_invalid_feature = True; break
            try:
                influencer_features.append(float(value))
            except (ValueError, TypeError):
                missing_or_invalid_feature = True; break

        if missing_or_invalid_feature:
             return 0.0
        if len(influencer_features) != len(centroid_vector):
            print(f"Fitness Logic Error: Feature extraction resulted in mismatched length ({len(influencer_features)} vs {len(centroid_vector)}).")
            return 0.0

        # Calculate Distance and Fitness
        influencer_vector = np.array(influencer_features)

        if influencer_vector.shape != centroid_vector.shape:
             print(f"Fitness Logic Error: Shape mismatch {influencer_vector.shape} vs {centroid_vector.shape}.")
             return 0.0

        distance = np.linalg.norm(influencer_vector - centroid_vector)

        epsilon = 1e-9
        distance = max(0.0, distance)
        fitness_score = 1.0 / (1.0 + distance + epsilon)

        return float(fitness_score) if pd.notna(fitness_score) and np.isfinite(fitness_score) else 0.0

    except Exception as e:
        print(f"ERROR calculating fitness: {e}")
        traceback.print_exc()
        return 0.0

