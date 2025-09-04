
import pytest
import numpy as np
import pandas as pd

try:
    from algorithms.fitness import calculate_fitness
except ImportError:
    try:
        from fitness import calculate_fitness
    except ImportError:
        print("ERROR: Could not import calculate_fitness in test file. Adjust path.")


TEST_FEATURES = ['feature_a_norm', 'feature_b_norm', 'feature_c_norm']
TARGET_CENTROID = np.array([0.8, 0.2, 0.5])
EPSILON = 1e-9



def test_perfect_match():
    """ Test fitness when influencer matches centroid exactly (distance 0). """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': 0.2, 'feature_c_norm': 0.5})
    expected_fitness = 1.0 / (1.0 + 0.0 + EPSILON)
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == pytest.approx(expected_fitness)

def test_known_distance():
    """ Test fitness for a known, non-zero distance. """
    influencer = pd.Series({'feature_a_norm': 0.7, 'feature_b_norm': 0.3, 'feature_c_norm': 0.6})

    distance = np.sqrt(0.03)
    expected_fitness = 1.0 / (1.0 + distance + EPSILON)
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == pytest.approx(expected_fitness)

def test_different_distance():
    """ Test fitness for another known, non-zero distance. """
    influencer = pd.Series({'feature_a_norm': 0.0, 'feature_b_norm': 0.0, 'feature_c_norm': 0.0})

    distance = np.sqrt(0.93)
    expected_fitness = 1.0 / (1.0 + distance + EPSILON)
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == pytest.approx(expected_fitness)

def test_missing_feature():
    """ Test fitness returns 0.0 when a required feature is missing. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_c_norm': 0.5}) # Missing feature_b_norm
    expected_fitness = 0.0
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == expected_fitness

def test_nan_feature():
    """ Test fitness returns 0.0 when a required feature is NaN. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': np.nan, 'feature_c_norm': 0.5})
    expected_fitness = 0.0
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == expected_fitness

def test_non_numeric_feature():
    """ Test fitness returns 0.0 when a required feature has a non-numeric value. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': 'invalid_string', 'feature_c_norm': 0.5})
    expected_fitness = 0.0
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == expected_fitness

def test_mismatched_centroid_feature_length():
    """ Test fitness returns 0.0 when centroid and feature list lengths differ. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': 0.2, 'feature_c_norm': 0.5})
    wrong_centroid = np.array([0.8, 0.2]) # Only 2 elements, features list has 3
    expected_fitness = 0.0
    # This check happens early in the function
    assert calculate_fitness(influencer, wrong_centroid, TEST_FEATURES) == expected_fitness

def test_mismatched_feature_centroid_length():
    """ Test fitness returns 0.0 when feature list and centroid lengths differ. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': 0.2, 'feature_c_norm': 0.5})
    wrong_features = ['feature_a_norm', 'feature_b_norm'] # Only 2 features
    expected_fitness = 0.0
    # This check happens early in the function
    assert calculate_fitness(influencer, TARGET_CENTROID, wrong_features) == expected_fitness


def test_empty_features_list():
    """ Test fitness returns 0.0 with an empty features list. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': 0.2, 'feature_c_norm': 0.5})
    empty_features = []
    empty_centroid = np.array([])
    expected_fitness = 0.0
    # The initial length check should catch this
    assert calculate_fitness(influencer, empty_centroid, empty_features) == expected_fitness

def test_none_centroid():
    """ Test fitness returns 0.0 with None centroid. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': 0.2, 'feature_c_norm': 0.5})
    expected_fitness = 0.0
    assert calculate_fitness(influencer, None, TEST_FEATURES) == expected_fitness

def test_none_features_list():
    """ Test fitness returns 0.0 with None features list. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': 0.2, 'feature_c_norm': 0.5})
    expected_fitness = 0.0
    assert calculate_fitness(influencer, TARGET_CENTROID, None) == expected_fitness

def test_invalid_influencer_metrics_type():
    """ Test fitness returns 0.0 if influencer_metrics is not Series or dict. """
    influencer = [0.8, 0.2, 0.5] # Pass a list instead
    expected_fitness = 0.0
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == expected_fitness


def test_dict_input():
    """ Test fitness works correctly with a dictionary input for influencer_metrics. """
    influencer = {'feature_a_norm': 0.7, 'feature_b_norm': 0.3, 'feature_c_norm': 0.6}
    distance = np.sqrt(0.03)
    expected_fitness = 1.0 / (1.0 + distance + EPSILON)
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == pytest.approx(expected_fitness)

def test_extra_features_in_influencer():
    """ Test fitness ignores extra features in the influencer data if using dict/Series. """
    influencer = pd.Series({
        'feature_a_norm': 0.8,
        'feature_b_norm': 0.2,
        'feature_c_norm': 0.5,
        'extra_feature': 999 # Should be ignored by the loop over TEST_FEATURES
    })
    expected_fitness = 1.0 / (1.0 + 0.0 + EPSILON)
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == pytest.approx(expected_fitness)

def test_zero_distance_edge_case():
    """ Explicitly test the epsilon addition for zero distance. """
    influencer = pd.Series({'feature_a_norm': 0.8, 'feature_b_norm': 0.2, 'feature_c_norm': 0.5})
    expected_fitness = 1.0 / (1.0 + EPSILON) # distance is 0
    assert calculate_fitness(influencer, TARGET_CENTROID, TEST_FEATURES) == pytest.approx(expected_fitness)

