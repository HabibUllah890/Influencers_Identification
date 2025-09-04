
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx, no_update
import pandas as pd
import numpy as np

import time
import traceback
import os

try:
    from utils.data_loader import load_and_preprocess_data, perform_clustering_analysis
    from algorithms.genetic_algorithm import run_ga
    from algorithms.hill_climbing import run_hc
except ImportError as e:
    print(f"--- ERROR: Failed to import necessary modules: {e} ---")
    print("--- Please ensure utils/data_loader.py, algorithms/genetic_algorithm.py, and algorithms/hill_climbing.py exist and are importable. ---")
    def load_and_preprocess_data(filepath): print(f"Dummy load called for {filepath}"); return pd.DataFrame({'Username':['user1','user2'], 'Name':['Name 1', 'Name 2'], 'Followers':[1000,2000], 'Likes':[100,200], 'Comments':[10,20], 'ViewsAvgPerPost':[500,1000], 'Shares':[5,10], 'EngagementRate':[0.1, 0.05], 'Followers_normalized':[0.1,0.2], 'Likes_normalized':[0.1,0.2], 'Comments_normalized':[0.1,0.2], 'ViewsAvgPerPost_normalized':[0.1,0.2], 'Shares_normalized':[0.1,0.2]})
    def perform_clustering_analysis(df, features_to_cluster): print("Dummy cluster called"); analysis_df=pd.DataFrame({'Cluster':[0,1], 'Size':[1,1]}); centroids=np.array([[0.1,0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2,0.2]]); sil_df=pd.DataFrame({'K':[2, 3, 4],'Silhouette Score':[0.8, 0.7, 0.6]}); sse_df=pd.DataFrame({'K':[2,3,4],'SSE':[10,8,7]}); return df, centroids, features_to_cluster, analysis_df, 2, sil_df, sse_df # Return dummy sil_df too
    def run_ga(**kwargs): print("Dummy run_ga called"); time.sleep(0.2); N=kwargs.get('N_results',5); return pd.DataFrame({'Username': [f'GA_User_{i}' for i in range(N)], 'Name': [f'GA Influencer Name {i}' for i in range(N)], 'GA Rank': list(range(1, N+1)), 'individual_fitness': np.random.rand(N)*0.1 + 0.85, 'Followers': np.random.randint(1000, 100000, N), 'Likes': np.random.randint(100, 10000, N), 'Comments': np.random.randint(10, 1000, N), 'ViewsAvgPerPost': np.random.randint(1000, 50000, N), 'Shares': np.random.randint(5, 500, N), 'EngagementRate': np.random.rand(N)*0.1, 'Error': [None]*N})
    def run_hc(**kwargs): print("Dummy run_hc called"); time.sleep(0.1); N=kwargs.get('N_results',5); return pd.DataFrame({'Username': [f'HC_User_{i}' for i in range(N)], 'Name': [f'HC Influencer Name {i}' for i in range(N)], 'HC Rank': list(range(1, N+1)), 'individual_fitness': np.random.rand(N)*0.1 + 0.80, 'Followers': np.random.randint(1000, 100000, N), 'Likes': np.random.randint(100, 10000, N), 'Comments': np.random.randint(10, 1000, N), 'ViewsAvgPerPost': np.random.randint(1000, 50000, N), 'Shares': np.random.randint(5, 500, N), 'EngagementRate': np.random.rand(N)*0.1, 'Error': [None]*N })

print("Loading and preprocessing initial data...")
DATA_FILEPATH = '/Final_Year_Project/data/tiktok_top_1000.csv'

df_processed_global = pd.DataFrame()

if not os.path.exists(DATA_FILEPATH):
    print(f"--- FATAL ERROR: Data file not found at {DATA_FILEPATH} ---")
    print("--- Please ensure the file exists and the path is correct. ---")
    print("--- Application will load structure, but cannot process data without the file. ---")
else:
    try:
        print(f"Attempting to load data from: {DATA_FILEPATH}")
        df_processed_global = load_and_preprocess_data(DATA_FILEPATH)
        if df_processed_global is None: # Function might return None on error
             print("--- ERROR: load_and_preprocess_data returned None. Check function for errors. ---")
             df_processed_global = pd.DataFrame() # Assign empty DF
        elif not isinstance(df_processed_global, pd.DataFrame):
             print(f"--- ERROR: Expected DataFrame from load_and_preprocess_data, got {type(df_processed_global)}. ---")
             df_processed_global = pd.DataFrame() # Assign empty DF
        elif df_processed_global.empty:
            print("--- WARNING: Data loaded but resulted in an empty DataFrame. Check preprocessing logic in data_loader.py. ---")
        else:
            print(f"Initial data loaded successfully. Shape: {df_processed_global.shape}")
            # Ensure required columns for display exist after loading
            required_cols = ['Name', 'Username', 'Followers', 'Likes', 'Comments', 'ViewsAvgPerPost', 'Shares', 'EngagementRate'] # Add any other cols needed
            missing_display_cols = [col for col in required_cols if col not in df_processed_global.columns]
            if missing_display_cols:
                print(f"--- WARNING: Loaded data is missing expected columns for display: {missing_display_cols} ---")

    except Exception as e:
        print(f"--- FATAL ERROR: Failed to load or preprocess data from {DATA_FILEPATH} ---")
        print(f"--- Error: {e} ---")
        traceback.print_exc()
        print("--- Application may not function correctly. ---")
        df_processed_global = pd.DataFrame() # Ensure it's an empty DF


def generate_cluster_descriptions(centroids_array, feature_names, analysis_df):

    descriptions = {}
    # Basic input validation
    if not isinstance(centroids_array, np.ndarray) or centroids_array.ndim != 2 or \
       not isinstance(feature_names, list) or len(feature_names) == 0 or \
       not isinstance(analysis_df, pd.DataFrame):
        print("Warning: Invalid input for generating descriptions (types or missing dataframe).")
        num_clusters_fallback = len(centroids_array) if isinstance(centroids_array, np.ndarray) else 0
        return {i: f"Cluster {i} (Size: ?)" for i in range(num_clusters_fallback)}

    # Check required columns in analysis_df
    if 'Size' not in analysis_df.columns or 'Cluster' not in analysis_df.columns:
         print("Warning: analysis_df missing required columns ('Size', 'Cluster') for descriptions.")
         # Fallback to basic description without size
         num_clusters = centroids_array.shape[0]
         num_features = len(feature_names)
         # Adjust features if dimensions mismatch
         if centroids_array.shape[1] != num_features:
             print(f"Warning: Centroid dimensions ({centroids_array.shape[1]}) mismatch number of feature names ({num_features}). Adjusting.")
             num_features = min(centroids_array.shape[1], num_features)
             feature_names = feature_names[:num_features]
         # Create descriptions without size
         for i in range(num_clusters):
             feature_value_strings = [f"{feature_names[j].replace('_normalized', '')}: {centroids_array[i, j]:.3f}" for j in range(num_features)]
             descriptions[i] = f"Cluster {i} | {'; '.join(feature_value_strings)}"
         return descriptions

    num_clusters = centroids_array.shape[0]
    if num_clusters == 0: return {}

    num_features = len(feature_names)
    if centroids_array.shape[1] != num_features:
        print(f"Warning: Centroid dimensions ({centroids_array.shape[1]}) mismatch number of feature names ({num_features}). Truncating feature names list.")
        num_features = min(centroids_array.shape[1], num_features)
        feature_names = feature_names[:num_features] # Use only the names we have values for

    try:
        if 'Cluster' in analysis_df.columns: analysis_df_indexed = analysis_df.set_index('Cluster')
        else: print("Warning: 'Cluster' column missing, cannot set index."); analysis_df_indexed = None
    except KeyError: print("Warning: 'Cluster' column not found for indexing."); analysis_df_indexed = None

    for i in range(num_clusters):
        size = '?';
        if analysis_df_indexed is not None and i in analysis_df_indexed.index:
             try: size_val = analysis_df_indexed.loc[i, 'Size']; size = int(size_val) if pd.notna(size_val) else '?'
             except (ValueError, TypeError, KeyError) as e: print(f"Warning: Could not get size for cluster {i} from index: {e}")
        elif i < len(analysis_df) and 'Size' in analysis_df.columns: # Fallback
             try: size_val = analysis_df.iloc[i]['Size']; size = int(size_val) if pd.notna(size_val) else '?'
             except (ValueError, TypeError, IndexError) as e: print(f"Warning: Could not get size for cluster {i} using iloc: {e}")

        feature_value_strings = []
        for j in range(num_features):
            feature_name = feature_names[j]
            display_name = feature_name.replace('_normalized', '')
            value = centroids_array[i, j] if j < centroids_array.shape[1] else np.nan
            feature_value_strings.append(f"{display_name}: {value:.3f}")

        details = "; ".join(feature_value_strings)
        descriptions[i] = f"Cluster {i} (Size: {size}) | {details}"

    return descriptions


# --- Initialize Dash App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


app.layout = html.Div(id='app-container', children=[
    html.H1("Influencer Identification Tool"),
    html.P("Run clustering, select profile, configure & run GA/HC to identify influencers."),
    html.Hr(),
    dcc.Store(id='clustering-data-store', storage_type='memory'),
    dcc.Store(id='selected-target-store', storage_type='memory'),

    # Step 1: Clustering (with Silhouette Table Div)
    html.Div(className='config-section', children=[
         html.H3("Step 1: Run Clustering Analysis"),
         html.Button('Run Clustering & Show Profiles', id='run-clustering-button', n_clicks=0),
         dcc.Loading(id="loading-clustering", type="circle", children=[
             html.Div(id='cluster-summary-display', className='status-info', children="Clustering not yet run."),
             html.Div(id='silhouette-table-div', style={'marginTop': '15px'}) # Div for Silhouette table
         ]),
    ]),

    # Step 2: Cluster Selection
    html.Div(id='cluster-selection-area', className='config-section', style={'display': 'none', 'marginTop': '20px'}, children=[
        html.H3("Step 2: Select Target Cluster Profile"),
        dcc.RadioItems(id='cluster-selector', options=[], value=None, labelStyle={'display': 'block', 'margin': '5px'}),
        html.Div(id='selection-status', className='status-info', style={'marginTop': '10px', 'minHeight': '20px'})
    ]),

    # Step 3: Algorithm Config & Run (Fully Configurable)
    html.Div(id='run-algorithms-area', className='config-section', style={'display': 'none', 'marginTop': '20px'}, children=[
        html.H3("Step 3: Configure & Run Identification Algorithms"),
        html.Div(className='config-params-grid', style={'display':'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap':'15px 20px', 'alignItems': 'end'}, children=[
             # GA Params
             html.Div([ html.Label("GA Gens:", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='ga-generations', type='number', value=100, min=10, step=10, style={'width':'90%'}) ]),
             html.Div([ html.Label("GA Pop Size:", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='ga-pop-size', type='number', value=50, min=10, step=10, style={'width':'90%'}) ]),
             html.Div([ html.Label("GA Mut % (0-1):", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='ga-mut-rate', type='number', value=0.1, min=0.0, max=1.0, step=0.05, style={'width':'90%'}) ]),
             html.Div([ html.Label("GA Crossover % (0-1):", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='ga-crossover-rate', type='number', value=0.8, min=0.0, max=1.0, step=0.1, style={'width':'90%'}) ]),
             html.Div([ html.Label("GA Elitism Count:", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='ga-elitism-count', type='number', value=2, min=0, step=1, style={'width':'90%'}) ]),
             html.Div([ html.Label("GA Group Size (Internal):", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='ga-group-size', type='number', value=50, min=10, step=10, style={'width':'90%'}) ]),
             # HC Params
             html.Div([ html.Label("HC Max Iter:", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='hc-max-iter', type='number', value=500, min=50, step=100, style={'width':'90%'}) ]),
             html.Div([ html.Label("HC Stopping Streak:", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='hc-stopping-streak', type='number', value=100, min=10, step=10, style={'width':'90%'}) ]),
             html.Div([ html.Label("HC Optimize Group Size:", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='hc-optimize-group-size', type='number', value=50, min=10, step=10, style={'width':'90%'}) ]),
             # Comparison & Display Params
             html.Div([ html.Label("Compare Top N:", style={'fontWeight':'bold', 'display':'block'}), dcc.Input(id='compare-n-input', type='number', value=15, min=5, max=1000, step=5, style={'width':'90%'}) ]),
             html.Div([ html.Label("Results Per Page:", style={'fontWeight':'bold', 'display':'block'}), dcc.Dropdown(id='results-per-page',
                            options=[ {'label': f'{n} per page', 'value': n} for n in [15, 25, 50, 100]], value=25, clearable=False, style={'width': '95%'} )]),
        ]),
        html.Button('Run Identification (GA & HC)', id='run-ga-hc-button', n_clicks=0, style={'marginTop': '20px', 'padding':'10px 15px', 'cursor':'pointer'}),
        dcc.Loading(id="loading-algorithms", type="circle", children=[
             html.Div(id='run-status-message', className='status-info', style={'minHeight': '20px', 'marginTop': '10px'})
         ]),
    ]),

    # Step 4: Results
    html.Div(id='results-area', className='results-section', style={'display': 'none', 'marginTop': '20px'}, children=[
        html.H3("Identification Results"),
        html.P(id='results-description'),
        html.Div(id='performance-comparison-div', style={'marginBottom': '20px', 'padding': '10px', 'border': '1px solid #ddd', 'backgroundColor': '#f9f9f9'}),



        html.P(id='results-count-text'), # Dynamic text
        html.Div(className='results-columns-container', style={'display':'flex', 'flexWrap':'wrap', 'gap':'20px', 'marginBottom': '30px'}, children=[
           html.Div(className='ranking-column', style={'flex':'1', 'minWidth': '450px'}, children=[
                html.H4("Genetic Algorithm Results"),
                dash_table.DataTable(
                    id='ga-ranking-table', columns=[], data=[],
                    page_current=0, page_size=25, page_action='native', # Pagination enabled
                    sort_action='native', filter_action='native', export_format='csv',
                    style_table={'overflowX': 'auto', 'width': '100%'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230, 230, 230)'},
                    # ** UPDATED STYLING **
                    style_cell={'textAlign': 'left', 'padding':'3px 5px', # Reduced padding
                                'minWidth': '50px', 'width': 'auto', 'maxWidth':'250px',
                                'whiteSpace': 'normal', 'height': 'auto', 'fontSize': '12px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'Name'}, 'minWidth': '150px', 'width': '25%', 'fontWeight':'bold'},
                        {'if': {'column_id': 'GA Rank'}, 'width': '40px', 'textAlign': 'center'},
                    ]
                )
           ]),
           html.Div(className='ranking-column', style={'flex':'1', 'minWidth': '450px'}, children=[
                html.H4("Hill Climbing Results"),
                dash_table.DataTable(
                    id='hc-ranking-table', columns=[], data=[],
                    page_current=0, page_size=25, page_action='native', # Pagination enabled
                    sort_action='native', filter_action='native', export_format='csv',
                    style_table={'overflowX': 'auto', 'width': '100%'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230, 230, 230)'},
                    # ** UPDATED STYLING **
                    style_cell={'textAlign': 'left', 'padding':'3px 5px', # Reduced padding
                                'minWidth': '50px', 'width': 'auto', 'maxWidth':'250px',
                                'whiteSpace': 'normal', 'height': 'auto', 'fontSize': '12px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'Name'}, 'minWidth': '150px', 'width': '25%', 'fontWeight':'bold'},
                        {'if': {'column_id': 'HC Rank'}, 'width': '40px', 'textAlign': 'center'},
                        # Removed specific widths for other columns
                    ]
                )
           ])
        ]),
        # Common Influencers Table
        html.Div(id='common-influencers-div', style={'marginTop': '20px'}, children=[
            html.Div(id='common-influencers-title-div'),
            dash_table.DataTable(
                id='common-influencers-table', columns=[], data=[], page_size=15, page_action='native', page_current=0,
                style_table={'overflowX': 'auto', 'width': '100%'},
                style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230, 230, 230)'},
                # ** UPDATED STYLING **
                style_cell={'textAlign': 'left', 'padding':'3px 5px', # Reduced padding
                            'minWidth': '60px', 'width': 'auto', 'maxWidth':'250px',
                            'whiteSpace': 'normal', 'height': 'auto', 'fontSize': '12px'},
                style_cell_conditional=[
                        {'if': {'column_id': 'Name'}, 'minWidth': '150px', 'width': '25%', 'fontWeight':'bold'},
                        {'if': {'column_id': 'Username'}, 'width': '100px'},
                        {'if': {'column_id': 'GA Rank'}, 'width': '50px', 'textAlign': 'center'},
                        {'if': {'column_id': 'HC Rank'}, 'width': '50px', 'textAlign': 'center'},
                        # Removed specific widths for fitness columns
                ]
            )
        ])
    ]),
])


# --- CALLBACK 1: Run Clustering & Populate Options (MODIFIED - Added Silhouette Table Output) ---
@app.callback(
    # Existing Outputs
    Output('clustering-data-store', 'data'),
    Output('cluster-summary-display', 'children'),
    Output('cluster-selector', 'options'),
    Output('cluster-selector', 'value'),
    Output('cluster-selection-area', 'style'),
    Output('run-algorithms-area', 'style'),
    Output('results-area', 'style', allow_duplicate=True),
    Output('selected-target-store', 'data', allow_duplicate=True),
    # ** NEW Output for Silhouette Table Div **
    Output('silhouette-table-div', 'children'),
    # Inputs
    Input('run-clustering-button', 'n_clicks'),
    prevent_initial_call=True
)
def trigger_clustering(n_clicks):
    global df_processed_global
    start_time = time.time()
    silhouette_table_content = []

    default_outputs = (None, "Click button to run clustering.", [], None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None, [])

    if n_clicks == 0:
        return (no_update,) * 8 + ([],) # Return empty list for silhouette table on initial load

    if not isinstance(df_processed_global, pd.DataFrame) or df_processed_global.empty:
        status = "Error: Global data empty."; print(status)
        return None, status, [], None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None, [html.P(status, style={'color':'red'})] # Show error in table div

    features_for_clustering_run = ['Followers_normalized', 'Likes_normalized', 'Comments_normalized', 'ViewsAvgPerPost_normalized', 'Shares_normalized']
    print(f"Callback 1: Running clustering with features: {features_for_clustering_run}")
    actual_features_to_cluster = [f for f in features_for_clustering_run if f in df_processed_global.columns]
    if len(actual_features_to_cluster) < 2:
        summary_text = f"Error: Need >= 2 valid clustering features."; print(summary_text)
        return None, summary_text, [], None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None, [html.P(summary_text, style={'color':'red'})]

    print("Performing clustering analysis...")
    try: results = perform_clustering_analysis(df_processed_global, features_to_cluster=actual_features_to_cluster)
    except Exception as e:
        summary_text = f"Error during clustering: {e}"; print(summary_text); traceback.print_exc()
        return None, summary_text, [], None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None, [html.P(summary_text, style={'color':'red'})]

    if results is None:
        summary_text = "Error: Clustering returned None."; print(summary_text)
        return None, summary_text, [], None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None, [html.P(summary_text, style={'color':'red'})]

    try:
        # Ensure sil_df and sse_df are unpacked (even if sse_df isn't used yet)
        (df_c, all_centroids, features_used, analysis_df, k_used, sil_df, sse_df) = results
        # Basic validation...
        if not isinstance(sil_df, pd.DataFrame) or 'K' not in sil_df.columns or 'Silhouette Score' not in sil_df.columns:
             print("Warning: Silhouette DataFrame (sil_df) missing or has incorrect columns.")
             sil_df = pd.DataFrame() # Use empty df if invalid
        # ... other validation ...
    except (TypeError, ValueError) as e:
        summary_text = f"Error unpacking/validating clustering results: {e}"; print(summary_text); traceback.print_exc()
        return None, summary_text, [], None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None, [html.P(summary_text, style={'color':'red'})]

    print("Generating cluster descriptions...")
    cluster_descriptions = generate_cluster_descriptions(all_centroids, features_used, analysis_df)
    stored_data = {'centroids': all_centroids.tolist(), 'features': features_used, 'descriptions': cluster_descriptions, 'k': k_used, 'analysis_json': analysis_df.to_json(orient='split')}
    cluster_options = [{'label': desc, 'value': i} for i, desc in cluster_descriptions.items()] if cluster_descriptions else []

    max_sil_score = 'N/A'
    if not sil_df.empty:
        try:
            numeric_scores = pd.to_numeric(sil_df['Silhouette Score'], errors='coerce')
            if not numeric_scores.isnull().all(): max_sil_score = f"{numeric_scores.max():.4f}"
            # ** Format Silhouette Table **
            sil_df_display = sil_df[['K', 'Silhouette Score']].copy()
            sil_df_display['Silhouette Score'] = pd.to_numeric(sil_df_display['Silhouette Score'], errors='coerce').round(4)
            sil_df_display.rename(columns={'K': 'Num Clusters (K)', 'Silhouette Score': 'Avg Silhouette Score'}, inplace=True)
            silhouette_table_content = [
                html.H5("Silhouette Score Evaluation:", style={'marginTop':'10px', 'marginBottom':'5px'}),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in sil_df_display.columns],
                    data=sil_df_display.to_dict('records'),
                    style_table={'maxWidth': '300px'}, # Keep table compact
                    style_cell={'textAlign': 'center', 'padding': '4px'},
                    style_header={'fontWeight': 'bold'}
                )
            ]
        except Exception as e:
            print(f"Warning: Could not process silhouette scores or create table: {e}")
            silhouette_table_content = [html.P("Could not display Silhouette scores.", style={'color':'orange'})]
    else:
         silhouette_table_content = [html.P("Silhouette score data not available.", style={'color':'grey'})]


    clustering_time = time.time() - start_time
    summary_text = f"Clustering Complete ({clustering_time:.2f}s): Found {k_used} clusters using {len(features_used)} features. Max Silhouette: {max_sil_score}."
    print("Clustering results ready.")
    select_style = {'display': 'block', 'marginTop': '20px'}; run_style = {'display': 'block', 'marginTop': '20px'}; results_style = {'display': 'none'}

    return stored_data, summary_text, cluster_options, None, select_style, run_style, results_style, None, silhouette_table_content


@app.callback(
    Output('selected-target-store', 'data'), Output('selection-status', 'children'),
    Input('cluster-selector', 'value'), State('clustering-data-store', 'data'),
    prevent_initial_call=True
)
def store_selection(selected_cluster_index, stored_cluster_data):
    if selected_cluster_index is None: return no_update, "Please select a cluster profile above."
    if stored_cluster_data is None: print("Warning: Clustering data missing."); return None, "Error: Clustering data missing."
    try:
        selected_idx_int = int(selected_cluster_index); required_keys = ['centroids', 'features', 'descriptions']
        if not isinstance(stored_cluster_data, dict) or not all(k in stored_cluster_data for k in required_keys): print("Error: Invalid stored cluster data."); return None, "Error: Invalid clustering data."
        all_centroids_list = stored_cluster_data['centroids']; features_used = stored_cluster_data['features']; descriptions = stored_cluster_data.get('descriptions', {})
        if not isinstance(all_centroids_list, list) or not (0 <= selected_idx_int < len(all_centroids_list)): print(f"Selection Error: Index {selected_idx_int} out of range."); return None, "Selection Error: Invalid index."
        selected_description = descriptions.get(selected_idx_int, descriptions.get(str(selected_idx_int), f"Cluster {selected_idx_int}"))
        selected_centroid_list = all_centroids_list[selected_idx_int]
        if not isinstance(selected_centroid_list, list) or not all(isinstance(x, (int, float)) for x in selected_centroid_list): print(f"Error: Invalid centroid data."); return None, "Error: Corrupted centroid data."
        target_data = {'centroid': selected_centroid_list, 'features': features_used, 'description': selected_description}
        status_text = f"Target set to: {selected_description}"; print(f"Target set for Cluster {selected_idx_int}")
        return target_data, status_text
    except (KeyError, ValueError, TypeError, IndexError) as e: print(f"Error processing selection: {e}"); traceback.print_exc(); return None, f"Error processing selection: {e}"
    except Exception as e: print(f"Unexpected error storing selection: {e}"); traceback.print_exc(); return None, f"Unexpected error."


# --- CALLBACK 3: Run GA / HC Algos & Comparisons
@app.callback(
    # Outputs for individual tables data/cols (4)
    Output('ga-ranking-table', 'data'), Output('hc-ranking-table', 'data'),
    Output('ga-ranking-table', 'columns'), Output('hc-ranking-table', 'columns'),
    # Outputs for table page size (2)
    Output('ga-ranking-table', 'page_size'), Output('hc-ranking-table', 'page_size'),
    # Output for performance comparison div (1)
    Output('performance-comparison-div', 'children'),
    # Outputs for common influencers table data/cols (2)
    Output('common-influencers-table', 'data'), Output('common-influencers-table', 'columns'),
    # Output for common influencers title (1)
    Output('common-influencers-title-div', 'children'),
    # Output for results count text (1)
    Output('results-count-text', 'children'),
    # ** REMOVED Outputs for Graphs **
    # Other outputs (4)
    Output('run-status-message', 'children'), Output('run-status-message', 'className'),
    Output('results-area', 'style', allow_duplicate=True), Output('results-description', 'children'),
    # Inputs
    Input('run-ga-hc-button', 'n_clicks'),
    # States (All configurable params + new page size)
    State('selected-target-store', 'data'),
    State('ga-generations', 'value'), State('ga-pop-size', 'value'), State('ga-mut-rate', 'value'),
    State('ga-crossover-rate', 'value'), State('ga-elitism-count', 'value'), State('ga-group-size', 'value'),
    State('hc-max-iter', 'value'), State('hc-stopping-streak', 'value'), State('hc-optimize-group-size', 'value'),
    State('compare-n-input', 'value'),
    State('results-per-page', 'value'),
    prevent_initial_call=True
)
def run_algorithms(n_clicks, selected_target_data,
                  # GA Params from UI
                  ga_gen, ga_pop, ga_mut, ga_cross, ga_elite, ga_group,
                  # HC Params from UI
                  hc_iter, hc_streak, hc_group,
                  # Comparison Param from UI
                  compare_n,
                  # Display Param from UI
                  results_per_page):
    start_time = time.time()
    # Initialize outputs
    performance_content = []; common_title = []; results_count_text = ""
    status_msg = "Processing..."; status_class = "status-info"
    results_style = {'display': 'none'}; results_desc = ""
    num_outputs = 15 # Reduced by 2 (no graphs)
    default_page_size = 25
    # Default output structure (adjust for removed outputs)
    default_outputs = (
        [],[],[],[], default_page_size, default_page_size, [], [],[], [], "",
        # Removed scatter_fig, hist_fig
        status_msg, status_class, results_style, results_desc
    )

    triggered_id = ctx.triggered_id
    if not triggered_id or triggered_id != 'run-ga-hc-button': return (no_update,) * num_outputs

    # --- Validate ALL User Inputs ---
    try:
        default_n_compare=15; min_n_compare=5; max_n_compare=1000
        N_compare = int(compare_n) if compare_n is not None else default_n_compare
        if not (min_n_compare <= N_compare <= max_n_compare): N_compare = default_n_compare; print(f"Warn: Compare N invalid. Using {N_compare}.")
        default_results_per_page = 25; valid_page_options=[15, 25, 50, 100]
        page_size = int(results_per_page) if results_per_page is not None else default_results_per_page
        if page_size not in valid_page_options: page_size = default_results_per_page; print(f"Warn: Invalid Results Per Page. Using {page_size}.")
        # GA Params validation...
        default_ga_pop=50; default_ga_gen=100; default_ga_mut=0.1; default_ga_cross=0.8; default_ga_elite=2; default_ga_group=50
        ga_pop_int = int(ga_pop) if ga_pop is not None and int(ga_pop) > 0 else default_ga_pop
        ga_gen_int = int(ga_gen) if ga_gen is not None and int(ga_gen) > 0 else default_ga_gen
        ga_mut_float = float(ga_mut) if ga_mut is not None and 0.0 <= float(ga_mut) <= 1.0 else default_ga_mut
        ga_cross_float = float(ga_cross) if ga_cross is not None and 0.0 <= float(ga_cross) <= 1.0 else default_ga_cross
        ga_elite_int = int(ga_elite) if ga_elite is not None and 0 <= int(ga_elite) < ga_pop_int else default_ga_elite
        ga_group_int = int(ga_group) if ga_group is not None and int(ga_group) > 0 else default_ga_group
        # HC Params validation...
        default_hc_iter=500; default_hc_streak=100; default_hc_group=50
        hc_iter_int = int(hc_iter) if hc_iter is not None and int(hc_iter) > 0 else default_hc_iter
        hc_streak_int = int(hc_streak) if hc_streak is not None and int(hc_streak) > 0 else default_hc_streak
        hc_group_int = int(hc_group) if hc_group is not None and int(hc_group) > 0 else default_hc_group
    except (ValueError, TypeError, AssertionError) as val_e:
        status_msg = f"Error: Invalid parameter input ({val_e})."; status_class = "status-error"; results_style = {'display': 'none'}
        print(status_msg); traceback.print_exc()
        performance_content = html.P(status_msg, style={'color': 'red'}); common_title = html.H4("Common Influencers (Error)")
        results_count_text = ""
        # Return defaults, update relevant outputs (adjust for removed graph outputs)
        return ([],[],[],[], page_size, page_size, performance_content, [],[], common_title, results_count_text, status_msg, status_class, results_style, "")

    # Set internal N for algos
    internal_N_results = 1000
    ga_params = {'population_size': ga_pop_int, 'generations': ga_gen_int, 'mutation_rate': ga_mut_float, 'crossover_rate': ga_cross_float, 'elitism_count': ga_elite_int, 'group_size': ga_group_int, 'N_results': internal_N_results}
    hc_params = {'max_iterations': hc_iter_int, 'stopping_streak': hc_streak_int, 'optimize_group_size': hc_group_int, 'N_results': internal_N_results}

    if selected_target_data is None: status_msg = "Error: Select target cluster."; status_class="status-warning"; results_style={'display':'none'}; performance_content = html.P(status_msg); common_title = html.H4("Common (Error)"); results_count_text = ""; print(status_msg); return ([],[],[],[], page_size, page_size, performance_content, [],[], common_title, results_count_text, status_msg, status_class, results_style, "")
    if not isinstance(df_processed_global, pd.DataFrame) or df_processed_global.empty: status_msg = "Error: Global data empty."; status_class="status-warning"; results_style={'display':'none'}; performance_content = html.P(status_msg); common_title = html.H4("Common (Error)"); results_count_text = ""; print(status_msg); return ([],[],[],[], page_size, page_size, performance_content, [],[], common_title, results_count_text, status_msg, status_class, results_style, "")
    df_run = df_processed_global.copy()
    try: target_centroid_list = selected_target_data['centroid']; target_features = selected_target_data['features']; target_centroid = np.array(target_centroid_list); # Further validation...
    except Exception as e: status_msg = f"Error processing target data: {e}"; status_class="status-warning"; results_style={'display':'none'}; print(f"Error target: {e}"); traceback.print_exc(); performance_content = html.P(status_msg); common_title = html.H4("Common (Error)"); results_count_text = ""; return ([],[],[],[], page_size, page_size, performance_content, [],[], common_title, results_count_text, status_msg, status_class, results_style, "")
    target_desc = selected_target_data.get('description', 'Selected Profile'); results_desc = f"Target: {target_desc}"
    print(f"Run Algos Callback: Targeting profile defined by {len(target_features)} features.")
    missing_req_cols = [f for f in target_features if f not in df_run.columns]
    if missing_req_cols: status_msg = f"Error: Data missing features: {missing_req_cols}"; status_class="status-warning"; results_style={'display':'none'}; performance_content = html.P(status_msg); common_title = html.H4("Common (Error)"); results_count_text = ""; print(status_msg); return ([],[],[],[], page_size, page_size, performance_content, [],[], common_title, results_count_text, status_msg, status_class, results_style, "")

    # --- Run Algorithms ---
    ga_ranked_df = pd.DataFrame(); hc_ranked_df = pd.DataFrame()
    ga_error = None; hc_error = None
    ga_exec_time = -1; hc_exec_time = -1
    fitness_col_name = 'individual_fitness'
    try: # GA Execution
        print("--- Starting GA ---"); ga_start_run_time = time.time()
        ga_result = run_ga(data_df=df_run, target_centroid=target_centroid, target_features=target_features, **ga_params)
        ga_exec_time = time.time() - ga_start_run_time
        if isinstance(ga_result, pd.DataFrame): ga_ranked_df = ga_result
        else: raise TypeError(f"Unexpected return type from run_ga: {type(ga_result)}")
        print(f"--- GA Finished: Time = {ga_exec_time:.2f}s ---")
        if not ga_ranked_df.empty and not all(col in ga_ranked_df.columns for col in ['Name', 'GA Rank', fitness_col_name]): print(f"Warn: GA result missing display cols.")
    except Exception as e: ga_error = f"GA Error: {e}"; print(ga_error); traceback.print_exc(); ga_ranked_df = pd.DataFrame([{'Error': ga_error}])
    try: # HC Execution
        print("--- Starting HC ---"); hc_start_run_time = time.time()
        hc_result = run_hc(data_df=df_run, target_centroid=target_centroid, target_features=target_features, **hc_params)
        hc_exec_time = time.time() - hc_start_run_time
        if isinstance(hc_result, pd.DataFrame): hc_ranked_df = hc_result
        else: raise TypeError(f"Unexpected return type from run_hc: {type(hc_result)}")
        print(f"--- HC Finished: Time = {hc_exec_time:.2f}s ---")
        if not hc_ranked_df.empty and not all(col in hc_ranked_df.columns for col in ['Name', 'HC Rank', fitness_col_name]): print(f"Warn: HC result missing display cols.")
    except Exception as e: hc_error = f"HC Error: {e}"; print(hc_error); traceback.print_exc(); hc_ranked_df = pd.DataFrame([{'Error': hc_error}])

    def format_individual_table(df_result, target_features, rank_col_name, fitness_col_name, error_msg=None):
        """Formats GA or HC table showing Name, Rank, Fitness, and original feature values."""
        base_cols = ['Name', rank_col_name, fitness_col_name]; original_features = [f.replace('_normalized', '') for f in target_features]; other_cols = ['Followers', 'Likes', 'Comments', 'ViewsAvgPerPost', 'Shares', 'EngagementRate']
        cols_to_display_ordered = []; all_available_cols = list(df_result.columns) if isinstance(df_result, pd.DataFrame) else []
        default_data = [{"Info": f"No results/cols for {rank_col_name}."}]; default_cols = [{"name": "Info", "id": "Info"}]; error_data = [{"Error": error_msg or f"Unknown error"}]; error_cols = [{"name": "Error", "id": "Error"}]
        if error_msg: return error_data, error_cols
        if not isinstance(df_result, pd.DataFrame) or df_result.empty: return default_data, default_cols
        if 'Error' in df_result.columns and not df_result['Error'].isnull().all(): return df_result[['Error']].dropna().to_dict('records'), [{"name": "Error", "id": "Error"}]
        for col in base_cols:
            if col in all_available_cols: cols_to_display_ordered.append(col)
        for orig_feat in original_features:
             if orig_feat in all_available_cols and orig_feat not in cols_to_display_ordered: cols_to_display_ordered.append(orig_feat)
        for other_col in other_cols:
             if other_col in all_available_cols and other_col not in cols_to_display_ordered: cols_to_display_ordered.append(other_col)
        if 'Name' not in cols_to_display_ordered and 'Username' in all_available_cols: cols_to_display_ordered.insert(0, 'Username')
        if not any(c in cols_to_display_ordered for c in ['Name', 'Username']) or rank_col_name not in cols_to_display_ordered:
             print(f"Warn: Essential cols missing for {rank_col_name} table."); fallback_cols = [c for c in all_available_cols if '_normalized' not in c and c not in ['index', 'Error']];
             if not fallback_cols: return default_data, default_cols; cols = [{"name": i, "id": i} for i in fallback_cols];
             try: data = df_result[fallback_cols].round(4).to_dict('records')
             except: data = df_result[fallback_cols].astype(str).to_dict('records'); return data, cols
        try:
            df_display = df_result[cols_to_display_ordered].copy(); numeric_cols = df_display.select_dtypes(include=np.number).columns; cols_to_round = [c for c in numeric_cols if c not in [rank_col_name]]
            if cols_to_round:
                 for col in ['Followers', 'Likes', 'Comments', 'ViewsAvgPerPost', 'Shares']:
                      if col in df_display.columns and col in cols_to_round:
                           try: df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
                           except: pass
                 other_numeric_cols = [c for c in cols_to_round if c not in ['Followers', 'Likes', 'Comments', 'ViewsAvgPerPost', 'Shares']]
                 if other_numeric_cols: df_display[other_numeric_cols] = df_display[other_numeric_cols].round(4)
            data = df_display.to_dict('records'); cols = [{"name": i, "id": i} for i in cols_to_display_ordered]
        except Exception as format_e: print(f"Error formatting table {rank_col_name}: {format_e}"); traceback.print_exc(); return [{"Error": f"Format error: {format_e}"}], [{"name": "Error", "id": "Error"}]
        return data, cols

    # --- Format Individual GA/HC Tables ---
    ga_data, ga_cols = format_individual_table(ga_ranked_df, target_features, 'GA Rank', fitness_col_name, ga_error)
    hc_data, hc_cols = format_individual_table(hc_ranked_df, target_features, 'HC Rank', fitness_col_name, hc_error)


    best_fitness_ga_val, avg_fitness_ga_val = np.nan, np.nan; best_fitness_hc_val, avg_fitness_hc_val = np.nan, np.nan
    if ga_error is None and isinstance(ga_ranked_df, pd.DataFrame) and not ga_ranked_df.empty and fitness_col_name in ga_ranked_df.columns:
        try: numeric_fitness_ga = pd.to_numeric(ga_ranked_df[fitness_col_name], errors='coerce'); best_fitness_ga_val = numeric_fitness_ga.iloc[0]; avg_fitness_ga_val = numeric_fitness_ga.head(N_compare).mean()
        except Exception as e: print(f"Error calc GA fitness metrics: {e}")
    if hc_error is None and isinstance(hc_ranked_df, pd.DataFrame) and not hc_ranked_df.empty and fitness_col_name in hc_ranked_df.columns:
         try: numeric_fitness_hc = pd.to_numeric(hc_ranked_df[fitness_col_name], errors='coerce'); best_fitness_hc_val = numeric_fitness_hc.iloc[0]; avg_fitness_hc_val = numeric_fitness_hc.head(N_compare).mean()
         except Exception as e: print(f"Error calc HC fitness metrics: {e}")
    best_fitness_ga_str=f"{best_fitness_ga_val:.4f}" if pd.notna(best_fitness_ga_val) else ('Error' if ga_error else 'N/A'); avg_fitness_ga_str=f"{avg_fitness_ga_val:.4f}" if pd.notna(avg_fitness_ga_val) else ('Error' if ga_error else 'N/A'); best_fitness_hc_str=f"{best_fitness_hc_val:.4f}" if pd.notna(best_fitness_hc_val) else ('Error' if hc_error else 'N/A'); avg_fitness_hc_str=f"{avg_fitness_hc_val:.4f}" if pd.notna(avg_fitness_hc_val) else ('Error' if hc_error else 'N/A'); ga_time_str=f"{ga_exec_time:.2f}" if ga_exec_time>=0 else 'Error'; hc_time_str=f"{hc_exec_time:.2f}" if hc_exec_time>=0 else 'Error'; perf_table_header=[html.Thead(html.Tr([html.Th("Metric"),html.Th("GA"),html.Th("HC")]))]; perf_table_body=[html.Tbody([html.Tr([html.Td("Best Fitness Found"),html.Td(best_fitness_ga_str),html.Td(best_fitness_hc_str)]),html.Tr([html.Td(f"Avg Fitness (Top {N_compare})"),html.Td(avg_fitness_ga_str),html.Td(avg_fitness_hc_str)]),html.Tr([html.Td("Execution Time (s)"),html.Td(ga_time_str),html.Td(hc_time_str)])])]; performance_content=[html.H4("Algorithm Performance Comparison"),html.Table(perf_table_header+perf_table_body,className='comparison-metric-table',style={'width':'auto','minWidth':'350px','marginBottom':'10px','borderCollapse':'collapse','border':'1px solid #ccc'})];
    if ga_error: performance_content.append(html.P(f"GA Status: {ga_error}", style={'color':'red','fontSize':'small','margin':'2px 0'}))
    if hc_error: performance_content.append(html.P(f"HC Status: {hc_error}", style={'color':'red','fontSize':'small','margin':'2px 0'}))

    common_data = []; common_cols = [{"name": "Info", "id": "Info"}]; common_usernames = []
    common_title = html.H4(f"Common Top Influencers (Found in Top {N_compare} by both GA & HC)")
    if ga_error is None and hc_error is None and isinstance(ga_ranked_df, pd.DataFrame) and not ga_ranked_df.empty and 'Username' in ga_ranked_df.columns and 'GA Rank' in ga_ranked_df.columns and fitness_col_name in ga_ranked_df.columns and isinstance(hc_ranked_df, pd.DataFrame) and not hc_ranked_df.empty and 'Username' in hc_ranked_df.columns and 'HC Rank' in hc_ranked_df.columns and fitness_col_name in hc_ranked_df.columns and 'Name' in ga_ranked_df.columns:
        try:
            ga_top_users = set(ga_ranked_df.head(N_compare)['Username']); hc_top_users = set(hc_ranked_df.head(N_compare)['Username'])
            common_usernames = list(ga_top_users.intersection(hc_top_users))
            if common_usernames:
                print(f"Found {len(common_usernames)} common influencers in Top {N_compare}.")
                ga_common_df = ga_ranked_df[ga_ranked_df['Username'].isin(common_usernames)][['Username', 'Name', 'GA Rank', fitness_col_name]].rename(columns={fitness_col_name: 'Fitness_GA'})
                hc_common_df = hc_ranked_df[hc_ranked_df['Username'].isin(common_usernames)][['Username', 'HC Rank', fitness_col_name]].rename(columns={fitness_col_name: 'Fitness_HC'})
                common_df = pd.merge(ga_common_df, hc_common_df[['Username', 'HC Rank', 'Fitness_HC']], on='Username', how='inner').sort_values(by='GA Rank')
                common_cols_display = ['Name', 'Username', 'GA Rank', 'Fitness_GA', 'HC Rank', 'Fitness_HC']
                common_df_display = common_df[common_cols_display].copy()
                numeric_common_cols = common_df_display.select_dtypes(include=np.number).columns
                if not numeric_common_cols.empty: common_df_display[numeric_common_cols] = common_df_display[numeric_common_cols].round(4)
                common_data = common_df_display.to_dict('records')
                common_cols = [{"name": i, "id": i} for i in common_cols_display]
            else: print(f"No common influencers found in Top {N_compare}."); common_data = [{"Info": f"No common influencers found in Top {N_compare}."}]
        except Exception as e: print(f"Error during common influencer calc: {e}"); traceback.print_exc(); common_data = [{"Error": f"Failed to calc common: {e}"}]; common_cols = [{"name": "Error", "id": "Error"}]
    elif ga_error or hc_error: common_data = [{"Info": f"Common influencers not calculated due to errors."}]
    else: common_data = [{"Info": f"Common influencers not calculated (check results)."}]


    total_time = time.time() - start_time
    ga_results_count = len(ga_data) if isinstance(ga_data, list) and ga_data and 'Error' not in ga_data[0] else 0
    hc_results_count = len(hc_data) if isinstance(hc_data, list) and hc_data and 'Error' not in hc_data[0] else 0
    # ** UPDATED results count text **
    results_count_text = f"Showing {ga_results_count} GA results and {hc_results_count} HC results (ranked by fitness), {page_size} per page."

    # (Status message logic remains the same, uses N_compare)
    if ga_error or hc_error: status_msg_parts = []; status_class = "status-warning"; status_msg = f"Completed with issues in {total_time:.2f}s."
    elif (isinstance(ga_ranked_df, pd.DataFrame) and ga_ranked_df.empty) or (isinstance(hc_ranked_df, pd.DataFrame) and hc_ranked_df.empty) or \
         (isinstance(ga_ranked_df, pd.DataFrame) and 'Error' in ga_ranked_df.columns and not ga_ranked_df['Error'].isnull().all()) or \
         (isinstance(hc_ranked_df, pd.DataFrame) and 'Error' in hc_ranked_df.columns and not hc_ranked_df['Error'].isnull().all()):
         status_msg = f"Completed in {total_time:.2f}s, but one or both algorithms returned no valid results or errors."; status_class = "status-warning"
    elif best_fitness_ga_str == 'N/A' or best_fitness_hc_str == 'N/A' or avg_fitness_ga_str == 'N/A' or avg_fitness_hc_str == 'N/A':
         status_msg = f"Completed in {total_time:.2f}s, but fitness metrics could not be fully calculated."; status_class = "status-warning"
    else:
        status_msg = f"Identification complete in {total_time:.2f} seconds."
        if common_usernames: status_msg += f" Found {len(common_usernames)} common influencers in Top {N_compare}."
        else: status_msg += f" No common influencers found in Top {N_compare}."
        status_class = "status-info"; print(status_msg)

    results_style = {'display': 'block', 'marginTop': '20px'}

    return (
        ga_data, hc_data, ga_cols, hc_cols,           # Individual Tables Data/Cols (4)
        page_size, page_size,                         # Table Page Sizes (2)
        performance_content,                          # Performance Div Content (1)
        common_data, common_cols,                     # Common Table Data/Cols (2)
        common_title,                                 # Common Table Title (1)
        results_count_text,                           # Results Count Text (1)
        # ** Removed scatter_fig, hist_fig **
        status_msg, status_class,                     # Status Message/Class (2)
        results_style,                                # Results Area Style (1)
        results_desc                                  # Results Description (1)
    ) # Total 15 outputs


# --- Run the App ---
if __name__ == '__main__':
    if not isinstance(df_processed_global, pd.DataFrame) or df_processed_global.empty: print("\n--- WARNING: Global DataFrame empty. ---")
    else: print("\n--- Global DataFrame loaded. Ready to start server. ---\n")
    print(f"--> Access the application at: http://127.0.0.1:8050/")
    app.run(debug=True, port=8050) # Set debug=False for production

