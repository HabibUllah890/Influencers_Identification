
import pandas as pd
import numpy as np
import os
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import warnings
import traceback

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def clean_numeric(value):
    """Cleans string representations of numbers (handles K, M, B)."""
    if pd.isna(value): return np.nan
    num_str = str(value).strip().replace(',', '')
    num_str = num_str.replace(' ', '')
    multiplier = 1
    if 'K' in num_str.upper(): multiplier = 1e3; num_str = num_str.upper().replace('K', '')
    elif 'M' in num_str.upper(): multiplier = 1e6; num_str = num_str.upper().replace('M', '')
    elif 'B' in num_str.upper(): multiplier = 1e9; num_str = num_str.upper().replace('B', '')
    try:
        # Extract the first valid float number found
        match = re.match(r"[-+]?\d*\.?\d+", num_str)
        if match: return float(match.group(0)) * multiplier
        else: return np.nan # Return NaN if no number pattern found
    except:
        return np.nan # Return NaN on any conversion error

# --- Data Loading and Preprocessing Function ---
def load_and_preprocess_data(filepath='../data/tiktok_top_1000.csv'): # Default relative path
    """Loads data, cleans, calculates metrics, normalizes, and returns DataFrame."""
    try:
        # Try interpreting filepath relative to this script's location
        if not os.path.isabs(filepath):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             filepath = os.path.join(script_dir, filepath)

        # Fallback checks if initial path doesn't exist
        if not os.path.exists(filepath):
             alt_path_cwd = 'tiktok_top_1000.csv'
             alt_path_proj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'tiktok_top_1000.csv')
             if os.path.exists(alt_path_cwd): filepath = alt_path_cwd
             elif os.path.exists(alt_path_proj_root): filepath = alt_path_proj_root
             else: raise FileNotFoundError(f"Cannot find data file at specified or fallback paths: {filepath}")

        print(f"Attempting to load data from: {filepath}")
        try: df = pd.read_csv(filepath)
        except UnicodeDecodeError: df = pd.read_csv(filepath, encoding='latin1')
        print(f"Successfully loaded data. Shape: {df.shape}")
    except FileNotFoundError: print(f"ERROR: Data file not found at {filepath}"); return pd.DataFrame()
    except Exception as e: print(f"ERROR loading data: {e}"); return pd.DataFrame()

    # --- Column Renaming ---
    column_map = {
        'account': 'Username','Tiktok name': 'Username','Tiktoker name': 'Username','Channel name': 'Username','Username': 'Username',
        'title': 'Name','Name': 'Name','Influencer name': 'Name',
        'Subscribers count': 'Followers','Followers count': 'Followers','Followers': 'Followers',
        'Likes avg.': 'Likes','Likes count': 'Likes','Likes': 'Likes',
        'Comments avg.': 'Comments','Comments count': 'Comments','Comments': 'Comments',
        'Shares avg.': 'Shares','Shares count': 'Shares','Shares': 'Shares',
        'Views avg.': 'ViewsAvgPerPost','Views avg': 'ViewsAvgPerPost','Avg. views': 'ViewsAvgPerPost','Views': 'ViewsAvgPerPost',
    }
    rename_dict = {}; df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for p_name, s_name in column_map.items():
        p_name_lower = p_name.lower().strip()
        if p_name_lower in df_cols_lower and s_name not in rename_dict.values():
            original_case_col = df_cols_lower[p_name_lower]
            rename_dict[original_case_col] = s_name
    df.rename(columns=rename_dict, inplace=True)

    # --- Data Cleaning ---
    numeric_cols = ['Followers', 'Likes' , 'ViewsAvgPerPost', 'Comments', 'Shares']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object': df[col] = df[col].apply(clean_numeric)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                median_val = df[col].median(); fill_val = median_val if pd.notna(median_val) else 0
                df[col] = df[col].fillna(fill_val)
        else: print(f"Warning: Expected numeric column '{col}' not found.")

    essential_cols = ['Username', 'Followers', 'Likes', 'Comments', 'Shares', 'ViewsAvgPerPost']
    missing_essential = [col for col in essential_cols if col not in df.columns or df[col].isnull().all()]
    if missing_essential: print(f"ERROR: Missing essential columns for calculations: {missing_essential}"); return pd.DataFrame()


    print("Calculating derived metrics (Using ER per View = (L+C+S) / Views)...")
    engagement_components = [ 'Likes' , 'Comments', 'Shares']
    df['TotalEngagement'] = df[engagement_components].sum(axis=1)
    # Engagement Rate per View
    df['EngagementRate'] = df.apply(lambda row: (row['TotalEngagement'] / row['ViewsAvgPerPost']) if row['ViewsAvgPerPost'] > 0 else 0, axis=1)
    # Other Ratios
    df['CommentsPerLike'] = df.apply(lambda row: (row['Comments'] / row['Likes']) if row['Likes'] > 0 else 0, axis=1)
    df['SharesPerLike'] = df.apply(lambda row: (row['Shares'] / row['Likes']) if row['Likes'] > 0 else 0, axis=1)
    print("Derived metrics calculated.")


    # --- Normalization ---
    metrics_to_normalize = list(
        {'Followers', 'Likes', 'Comments', 'Shares', 'ViewsAvgPerPost', 'TotalEngagement', 'EngagementRate',
         'CommentsPerLike', 'SharesPerLike'})
    metrics_to_normalize = sorted([col for col in metrics_to_normalize if col in df.columns])
    print(f"Attempting to normalize: {metrics_to_normalize}")

    for col in metrics_to_normalize:
         df[col] = df[col].replace([np.inf, -np.inf], np.nan)
         if df[col].isnull().any():
              median_val = df[col].median(); fill_val = median_val if pd.notna(median_val) else 0
              df[col] = df[col].fillna(fill_val)
         min_val = df[col].min(); max_val = df[col].max()
         normalized_col_name = f'{col}_normalized'
         if pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val) > 1e-9:
             df[normalized_col_name] = (df[col] - min_val) / (max_val - min_val)
         else: df[normalized_col_name] = 0.5 # Assign default for constant columns
    print(f"Normalization complete.")

    # --- Deduplication & Final Checks ---
    if 'Rank' in df.columns: df.sort_values('Rank', inplace=True)
    initial_rows = len(df)
    if 'Username' in df.columns:
        df.drop_duplicates(subset=['Username'], keep='first', inplace=True)
        if initial_rows > len(df): print(f"Removed {initial_rows - len(df)} duplicate rows based on 'Username'.")
        # Reset index AFTER dropping duplicates and potential sort
        df.reset_index(drop=True, inplace=True)
    else: print("Warning: Cannot drop duplicates as 'Username' column not found.")

    if 'Username' not in df.columns: print("ERROR: 'Username' column missing."); return pd.DataFrame()
    if 'Name' not in df.columns: df['Name'] = df['Username'] # Use Username if Name missing
    print(f"Data loading and preprocessing finished. Final DataFrame shape: {df.shape}")
    return df


# --- Clustering Function ---
def perform_clustering_analysis(df_processed, features_to_cluster=None, k_range=range(2, 8), default_k=3):
    """Performs K-Means clustering and returns results."""
    print("\n--- Starting Clustering Analysis Function ---")
    if df_processed.empty:
        print("ERROR: Input DataFrame for clustering is empty.")
        return pd.DataFrame(), None, [], pd.DataFrame(), 0, pd.DataFrame(), pd.DataFrame()

    if features_to_cluster is None:
        features_to_cluster = [
            'Followers_normalized',
            'Likes_normalized',
            'Comments_normalized',
            'ViewsAvgPerPost_normalized',
            'Shares_normalized'
        ]
        print(f"Using default features for clustering: {features_to_cluster}")

    features_available = []
    for f in features_to_cluster:
        if f in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[f]):
            features_available.append(f)
        else: print(f"Warning: Feature '{f}' not found or not numeric. Skipping for clustering.")
    if len(features_available) < 2:
        print(f"\nError: Need at least 2 valid numeric normalized features. Found: {features_available}")
        return df_processed, None, [], pd.DataFrame(), 0, pd.DataFrame(), pd.DataFrame()
    features_to_cluster = features_available

    print(f"Clustering based on features: {features_to_cluster}")
    X = df_processed[features_to_cluster].copy()

    # Final check/impute for invalid values
    if X.isnull().values.any() or np.isinf(X.values).any():
        print("Warning: Imputing NaN/Inf found in clustering features with 0.")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        for col in X.columns: # Ensure numeric
             if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # --- Find Optimal K ---
    print(f"Finding Optimal K (Testing K={min(k_range)} to {max(k_range)})...")
    sse = {}; silhouette_scores = {};
    best_k_silhouette = default_k; max_silhouette = -1.1
    for k in k_range:
        if k >= X.shape[0]: continue
        try:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42, verbose=0)
            X_float = X.astype(float)
            labels = kmeans.fit_predict(X_float)
            sse[k] = kmeans.inertia_
            unique_labels = len(set(labels))
            if unique_labels > 1 and unique_labels < X.shape[0]:
                score = silhouette_score(X_float, labels)
                silhouette_scores[k] = score
                if score > max_silhouette: max_silhouette = score; best_k_silhouette = k
            else: silhouette_scores[k] = -1
        except Exception as e: print(f"Warning: Could not calculate K={k}. Error: {e}"); sse[k] = np.nan; silhouette_scores[k] = np.nan

    sse_df = pd.DataFrame(list(sse.items()), columns=['K', 'SSE']).sort_values('K').reset_index(drop=True)
    sil_df = pd.DataFrame(list(silhouette_scores.items()), columns=['K', 'Silhouette Score']).sort_values('K').reset_index(drop=True)

    print("\n--- Elbow Method Results ---"); print(sse_df.to_markdown(index=False, floatfmt=".2f"))
    print("\n--- Silhouette Scores Results ---"); print(sil_df.to_markdown(index=False, floatfmt=".4f"))

    chosen_k = default_k
    valid_sil = sil_df.dropna(subset=['Silhouette Score'])
    if not valid_sil.empty and valid_sil['Silhouette Score'].max() > -1:
         chosen_k = int(valid_sil.loc[valid_sil['Silhouette Score'].idxmax()]['K'])
    print(f"\nSelected K = {chosen_k} (Max Silhouette Score method)")
    print(f"Proceeding with K = {chosen_k}")

    # --- Apply K-Means ---
    print(f"Applying K-Means with K={chosen_k}...")
    all_centroids = None
    df_with_clusters = df_processed.copy()
    cluster_analysis_df = pd.DataFrame()
    try:
        kmeans_final = KMeans(n_clusters=chosen_k, init='k-means++', n_init=10, max_iter=300, random_state=42,
                              verbose=0)
        df_with_clusters['Cluster'] = kmeans_final.fit_predict(X.astype(float))
        all_centroids = kmeans_final.cluster_centers_
        print(f"Assigned influencers to {chosen_k} clusters.")

        # --- Analyze Clusters ---
        print("Analyzing clusters...")
        analysis_metrics = ['Followers', 'Likes', 'Comments', 'Shares', 'ViewsAvgPerPost', 'EngagementRate',
                            'TotalEngagement']
        analysis_metrics = [m for m in analysis_metrics if m in df_with_clusters.columns]
        cluster_analysis_df = df_with_clusters.groupby('Cluster')[analysis_metrics].mean()
        cluster_size = df_with_clusters['Cluster'].value_counts().sort_index()
        cluster_analysis_df['Size'] = cluster_size
        cluster_analysis_df.reset_index(inplace=True)  # Make Cluster a column

        # --- Add Additional Benchmark Metrics ---
        print("Computing Benchmark Metrics...")
        X_cluster = X.values  # Original data for evaluation

        # Silhouette Score (already computed)
        silhouette_avg = silhouette_score(X_cluster, df_with_clusters['Cluster'])
        print(f"Silhouette Score: {silhouette_avg:.4f}")


    except Exception as e:
        print(f"ERROR during final K-Means or Analysis: {e}")
        traceback.print_exc()
        return df_processed, None, features_to_cluster, pd.DataFrame(), chosen_k, sil_df, sse_df

    print("--- Clustering Analysis Function Finished ---")
    return df_with_clusters, all_centroids, features_to_cluster, cluster_analysis_df, chosen_k, sil_df, sse_df

if __name__ == "__main__":
    print("Running data loader and clustering as a standalone script...")
    df_processed = load_and_preprocess_data() # Use default path

    if not df_processed.empty:
        # --- 1. Basic EDA (Original Scale) ---
        print("\n--- Descriptive Statistics (Original Scale) ---")
        stats_cols = ['Followers', 'Likes', 'Comments', 'Shares', 'ViewsAvgPerPost', 'EngagementRate', 'TotalEngagement']
        stats_cols = [col for col in stats_cols if col in df_processed.columns]
        if stats_cols:
            try: pd.set_option('display.float_format', '{:,.2f}'.format); print(df_processed[stats_cols].describe().to_markdown()); pd.reset_option('display.float_format')
            except Exception as e: print(f"Could not generate descriptive stats: {e}")

        print("\n--- Correlation Matrix (Original Scale) ---")
        corr_cols = ['Followers', 'Likes', 'Comments', 'Shares', 'ViewsAvgPerPost', 'EngagementRate', 'TotalEngagement']
        corr_cols = [col for col in corr_cols if col in df_processed.columns]
        if len(corr_cols) > 1:
            try: correlation_matrix = df_processed[corr_cols].corr(); print("Correlation coefficients:"); print(correlation_matrix.round(3).to_markdown())
            except Exception as e: print(f"Could not calculate correlation matrix: {e}")


        print("\n--- Descriptive Statistics (NORMALIZED - For Artificial Centroid) ---")

        norm_stats_cols = ['Followers_normalized', 'EngagementRate_normalized']

        norm_stats_cols = [col for col in norm_stats_cols if col in df_processed.columns]
        if len(norm_stats_cols) >= 2:
             try:
                 print(df_processed[norm_stats_cols].describe().to_markdown())
             except Exception as e:
                 print(f"Could not generate normalized descriptive stats: {e}")
        else:
             print(f"Could not generate normalized stats. Required columns not found: {norm_stats_cols}")

        cluster_features_to_run = [

        'Followers_normalized',
        'Likes_normalized',
        'Comments_normalized',
        'ViewsAvgPerPost_normalized',
        'Shares_normalized'

         ]

        results = perform_clustering_analysis(df_processed, features_to_cluster=cluster_features_to_run)
        (df_final, centroids, features_used, analysis, k, sil, sse) = results

        if centroids is not None:
            print("\n--- Clustering Results Summary ---")
            print(f"K Chosen: {k}")
            print(f"Features Used: {features_used}")
            print("\nCluster Analysis (Avg Metrics):"); print(analysis.to_markdown(index=False))
            print("\nCentroids (Normalized):"); print(pd.DataFrame(centroids, columns=features_used).to_markdown(index=True, floatfmt=".3f"))

        else: print("Clustering failed.")
    else: print("Data loading failed.")
    print("\n--- Script Finished ---")

