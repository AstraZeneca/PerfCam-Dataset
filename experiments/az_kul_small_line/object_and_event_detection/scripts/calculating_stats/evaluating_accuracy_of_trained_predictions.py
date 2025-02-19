import time
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

save_to_file = True
figures_folder = '../../stats/figures'
generate_csv_stats = True
ground_truth_stats_folder_path = "../../stats/csv/ground_truth"
predicted_stats_folder_path = "../../stats/csv/calculated"

pio.kaleido.scope.mathjax = None

tickvals_global = []
tickvals_labales_global = []

###############################################################################
# 0) Define some helper functions and/or global variables
###############################################################################

legend_font_size = 17
xtick_font_size = 18
xtick_title_font_size = 24

ytick_font_size = 18
ytick_title_font_size = 24

###############################################################################
# 1) Define edges & cameras
###############################################################################
# Dictionary: Edge label --> ground_truth CSV path
edge_truth_files = {
    'Edge 1': '../../stats/csv/ground_truth/products_passing_on_edge1.csv',
    'Edge 2': '../../stats/csv/ground_truth/products_passing_on_edge2.csv',
}

# Dictionary: Edge label --> pattern in "Counter Name"
# Adjust the pattern if your CSV uses slightly different naming
edge_patterns = {
    'Edge 1': 'edge_1',
    'Edge 2': 'edge_2_beginning',
}

# List of (camera name, CSV path)
camera_files = [
    ('Camera 1', '../../trained/predictions/camera-1/edge_counters.csv'),
    ('Camera 2', '../../trained/predictions/camera-2/edge_counters.csv'),
    ('Camera 3', '../../trained/predictions/camera-3/edge_counters.csv'),
    ('Camera 4', '../../trained/predictions/camera-4/edge_counters.csv'),
]

###############################################################################
# 2) Load ground truth data (for both edges) into one DataFrame
###############################################################################
all_ground_truth = []
for edge_name, csv_path in edge_truth_files.items():
    df_gt = pd.read_csv(csv_path, header=None, names=['Timestamp'])
    df_gt['Timestamp'] = pd.to_datetime(df_gt['Timestamp'])
    df_gt['Count'] = range(1, len(df_gt) + 1)  # cumulative count
    df_gt['Edge'] = edge_name
    all_ground_truth.append(df_gt)

ground_truth_df = pd.concat(all_ground_truth, ignore_index=True)
# columns: ['Timestamp', 'Count', 'Edge']

# Load and process edge counter data
max_edge_timestamps = []
for camera_name, csv_path in camera_files:
    df_edge = pd.read_csv(csv_path)
    # Assuming the timestamp column is named 'Timestamp' and is in the format HH:MM:SS.sss
    df_edge['Timestamp'] = pd.to_timedelta(df_edge['Timestamp'])
    max_edge_timestamps.append(df_edge['Timestamp'].max())

# Calculate the maximum timestamp from the edge counters
max_edge_timestamp = max(max_edge_timestamps)

# Assuming ground_truth_df is already created as per your provided code snippet
# Find the maximum timestamp in the ground truth data
max_ground_truth_timestamp = ground_truth_df['Timestamp'].max()

# Extract only the time part from the ground truth timestamp
max_ground_truth_time = max_ground_truth_timestamp.time()

# Convert the max_ground_truth_time to a Timedelta
max_ground_truth_timedelta = pd.to_timedelta(
    f"{max_ground_truth_time.hour:02}:{max_ground_truth_time.minute:02}:{max_ground_truth_time.second:02}.{max_ground_truth_time.microsecond // 1000:03}"
)

# Calculate the overall maximum timestamp
overall_max_timestamp = max(max_edge_timestamp, max_ground_truth_timedelta)
overall_max_timestamp_str = f"{int(overall_max_timestamp.total_seconds() // 3600):02}:{int((overall_max_timestamp.total_seconds() % 3600) // 60):02}:{int(overall_max_timestamp.total_seconds() % 60):02}"

print(f"Maximum timestamp from edge counters: {max_edge_timestamp}")
print(f"Maximum timestamp from ground truth: {max_ground_truth_timestamp}")
print(f"Overall maximum timestamp: {overall_max_timestamp}")

###############################################################################
# 3) Load predictions from all cameras, for all edges, into one DataFrame
###############################################################################
all_predictions = []
camera_column_choice = {}

for camera_name, csv_path in camera_files:
    df_cam = pd.read_csv(csv_path)
    df_cam['Camera'] = camera_name

    # For each edge pattern, extract the relevant rows
    for edge_name, pattern in edge_patterns.items():
        df_edge = df_cam[df_cam['Counter Name'].str.contains(pattern)]
        if df_edge.empty:
            continue  # no data for this edge in this camera CSV

        # Decide which column to pick (Count In vs Count Out) based on last row
        last_in = df_edge['Count In'].iloc[-1]
        last_out = df_edge['Count Out'].iloc[-1]
        if last_in >= last_out:
            col_choice = 'Count In'
        else:
            col_choice = 'Count Out'

        camera_column_choice[(camera_name, edge_name)] = col_choice
        df_edge['Selected Count'] = df_edge[col_choice]
        df_edge['Edge'] = edge_name  # label which edge this data is for

        all_predictions.append(df_edge)

predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_df['Timestamp'] = pd.to_datetime(predictions_df['Timestamp'], infer_datetime_format=True)

colors = px.colors.qualitative.Plotly


# columns: ['Timestamp', 'Counter Name', 'Count In', 'Count Out', 'Camera',
#           'Selected Count', 'Edge', ...]

###############################################################################
# 4) Utility functions for resampling & ground truth
###############################################################################
def resample_and_aggregate(df, time_col='Timestamp', value_col='Selected Count',
                           freq='1T', agg=['mean', 'std']):
    """
    - Sets time_col as DateTimeIndex, sorts by it
    - Resamples at frequency freq
    - Returns aggregated columns (e.g. 'mean' and 'std')
    """
    df = df.copy()
    df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)
    df_resampled = df[value_col].resample(freq).agg(agg)
    return df_resampled


def resample_ground_truth(df, time_col='Timestamp', freq='1T'):
    """
    - Sets time_col as DateTimeIndex, sorts by it
    - Resamples the 'Count' column by freq
    - Since Count is cumulative, we use max() within each time bin
    """
    df = df.copy()
    df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)
    df_resampled = df['Count'].resample(freq).max()
    return df_resampled


###############################################################################
# 5) Plot predictions vs ground truth (aggregated) for a single edge
###############################################################################
def plot_predictions_vs_ground_truth_aggregated(predictions, ground_truth, edge_name, freq='1T'):
    """
    predictions: DataFrame with 'Timestamp', 'Selected Count', 'Camera', 'Edge'
    ground_truth: DataFrame with 'Timestamp', 'Count', 'Edge'
    edge_name: which edge to plot
    freq: resampling frequency (e.g. '1T')
    """
    fig = go.Figure()
    global tickvals_global
    global tickvals_labales_global

    # 1) Filter for this edge and resample ground truth
    edge_gt = ground_truth[ground_truth['Edge'] == edge_name].copy()
    gt_resampled = resample_ground_truth(edge_gt, 'Timestamp', freq=freq)
    fig.add_trace(
        go.Scatter(
            x=gt_resampled.index,
            y=gt_resampled.values,
            mode='lines',
            name=f'Ground Truth {edge_name}',
            line=dict(color='green', dash='dash'),
            hoverinfo='x+y'
        )
    )

    # 2) For each camera, resample predictions
    edge_preds = predictions[predictions['Edge'] == edge_name].copy()
    for idx, camera in enumerate(edge_preds['Camera'].unique()):
        camera_data = edge_preds[edge_preds['Camera'] == camera].copy()
        camera_resampled = (
            resample_and_aggregate(
                camera_data,
                time_col='Timestamp',
                value_col='Selected Count',
                freq=freq,
                agg=['mean', 'std']
            )
            .reset_index()
            .rename(columns={'mean': 'Mean', 'std': 'Std'})
        )

        # Color override for Camera 3, otherwise use the Plotly sequence
        if camera == "Camera 3":
            this_color = "orange"
        else:
            this_color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=camera_resampled['Timestamp'],
                y=camera_resampled['Mean'],
                mode='lines+markers',
                name=f'{camera} (Mean) {edge_name}',
                line=dict(color=this_color),
                hoverinfo='x+y+name',
            )
        )

    # Combine timestamps into a single sorted list
    combined_timestamps = sorted(pd.concat([
        pd.Series(gt_resampled.index),
        pd.Series(edge_preds['Timestamp'].unique())
    ]).unique())

    # Define tick interval
    tick_interval = 10000
    tickvals = list(range(0, len(combined_timestamps), round(len(combined_timestamps) / 15)))

    # Ensure the last timestamp is included
    # Define tick labels
    tick_labels = [combined_timestamps[i].strftime('%-H:%M:%S') for i in tickvals] + [
        pd.Timestamp(overall_max_timestamp_str).strftime('%-H:%M:%S')]
    # tick_labels.append(overall_max_timestamp_str) # In case you want to force add the last timestamp in xticks
    tick_vals = [combined_timestamps[i] for i in tickvals] + [pd.Timestamp(overall_max_timestamp_str)]
    tickvals_global = deepcopy(tick_vals)
    tickvals_labales_global = deepcopy(tick_labels)

    # 3) Update layout
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.94, y=0.05, xanchor='right', yanchor='bottom', font=dict(size=legend_font_size)),
        margin=dict(r=0, l=5, t=5, b=5),
        xaxis=dict(
            title=dict(
                text='Time (H:MM:SS)',
                font=dict(size=xtick_title_font_size)
            ),
            tickfont=dict(size=xtick_font_size),
            tickformat='%-H:%M:%S',
            tickvals=tickvals_global,
            ticktext=tickvals_labales_global,
            tickangle=90,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title=dict(
                text='Cumulative Count',
                font=dict(size=ytick_title_font_size)
            ),
            tickfont=dict(size=ytick_font_size),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            rangemode='nonnegative',
            range=[0, 3600],
        ),
    )

    return fig


###############################################################################
# 6) Plot each edge in a loop
###############################################################################
for edge_name in edge_truth_files.keys():
    fig = plot_predictions_vs_ground_truth_aggregated(
        predictions_df,
        ground_truth_df,
        edge_name=edge_name,
        freq='1T'  # 1-minute bins
    )

    # Save to PDF
    edge_name_to_save = edge_name.lower().replace(' ', '_')

    if save_to_file:
        fig.write_image(f"{figures_folder}/{edge_name_to_save}_cumulative_count_plot.pdf", format="pdf")

        # The following statements are here due to a wierd Plotly bug: https://github.com/plotly/plotly.py/issues/3469
        time.sleep(2)
        fig.write_image(f"{figures_folder}/{edge_name_to_save}_cumulative_count_plot.pdf", format="pdf")
    else:
        fig.show()
        # Optionally, also display interactively

# ###############################################################################
# # 7) (Optional) Combined figure for both edges
# ###############################################################################

pass


###############################################################################
# 8) Accuracy-Shifting Functions
###############################################################################

def get_accuracy_series_for_plot(pred_df, gt_df, freq='1T'):
    """
    - Shift pred_df by best_shift
    - Resample pred_df & gt_df
    - Compute an 'Accuracy' column over time
    - Return a DataFrame with [Timestamp, Accuracy, Camera]
    """
    # Shift predictions
    shifted = pred_df.copy()
    shifted['Timestamp'] = shifted['Timestamp']

    # Resample predictions
    pred_resampled = (
        shifted
        .set_index('Timestamp')
        .sort_index()
        .resample(freq)['Selected Count']
        .max()
        .fillna(method='ffill')
        .rename('Pred')
    )

    # Resample ground truth
    gt_resampled = (
        gt_df
        .set_index('Timestamp')
        .sort_index()
        .resample(freq)['Count']
        .max()
        .fillna(method='ffill')
        .rename('GT')
    )

    # Compute per-minute counts by taking the difference
    pred_counts = pred_resampled.diff().fillna(pred_resampled.iloc[0]).rename('Pred_Count')
    gt_counts = gt_resampled.diff().fillna(gt_resampled.iloc[0]).rename('GT_Count')

    # Merge everything into a df
    merged = pd.concat([pred_resampled, gt_resampled, pred_counts, gt_counts], axis=1)

    # Row-wise accuracy
    def calc_accuracy(row):
        if pd.isna(row['GT']) or row['GT'] == 0:
            return np.nan
        return 1 - abs(row['Pred'] - row['GT']) / row['GT']

    # Function to calculate per-minute weighted accuracy
    def calc_weighted_accuracy(row, sum_of_all_gt_counts):
        if pd.isna(row['GT']) or row['GT'] == 0:
            return np.nan
        return 1 - ((abs(row['Pred'] - row['GT']) / row['GT']) * (row['GT_Count'] / sum_of_all_gt_counts))

    # Calculate per-minute weighted accuracy
    # merged['Accuracy'] = merged.apply(calc_weighted_accuracy, axis=1, args=(merged['GT_Count'].sum(),))

    merged['Accuracy'] = merged.apply(calc_accuracy, axis=1)

    merged.reset_index(inplace=True)
    camera_name = pred_df['Camera'].iloc[0] if len(pred_df) > 0 else 'Unknown'
    merged['Camera'] = camera_name
    return merged


def plot_accuracy_lines(accuracy_dfs, edge_name):
    """
    accuracy_dfs: list of DataFrames with columns [Timestamp, Accuracy, Camera]
    edge_name: label to display in the chart title
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    combined = pd.concat(accuracy_dfs, ignore_index=True)
    cameras = combined['Camera'].unique()

    # Keep track of where we've placed annotation text (in y-data coords),
    # so we can avoid overlapping text.
    used_y_positions = []
    proximity_threshold = 0.03  # Adjust the vertical spacing as needed

    for i, cam in enumerate(cameras):
        sub = combined[combined['Camera'] == cam]

        # 1) Plot the accuracy line
        fig.add_trace(
            go.Scatter(
                x=sub['Timestamp'],
                y=sub['Accuracy'],
                mode='lines+markers',
                name=f'{cam} ({edge_name})',
                connectgaps=True,
                line=dict(color=colors[i % len(colors)])
            )
        )

        # 2) Compute mean accuracy and draw a horizontal line
        # mean_acc = sub['Accuracy'].mean(skipna=True) # Incorrect way to calculate, due to uneven count in each minute
        mean_acc = sub.ffill().iloc[-1]['Pred'] / sub.ffill().iloc[-1]['GT']

        if pd.notna(mean_acc):
            # Plot the dotted line at the exact mean
            fig.add_hline(
                y=mean_acc,
                line_dash='dot',
                line_color=colors[i % len(colors)],
            )

            # 3) Decide where to place the annotation text
            #    Start by placing text exactly at mean_acc (y=mean_acc).
            #    If that is too close to an existing annotation, shift upward.
            adjusted_y = mean_acc
            while any(abs(adjusted_y - pos) < proximity_threshold for pos in used_y_positions):
                adjusted_y += proximity_threshold

            # Record the position so subsequent annotations avoid it.
            used_y_positions.append(adjusted_y)

            # 4) Create the text annotation outside the chart
            #    We place the text at `x=1.02` in paper coordinates (a bit to the right).
            #    Note that the dotted line remains at y=mean_acc, but
            #    the annotation text might be shifted slightly if there’s an overlap.
            # fig.update_layout(width=fig.layout.get('width', 800) + 100)  # Default width = 800, then add 100 pixels

            fig.add_annotation(
                x=1.00,
                xref='paper',
                xanchor='left',
                y=adjusted_y,
                yref='y',
                text=f"{mean_acc * 100:.1f}%",  # no extra text, just the numeric %
                showarrow=False,
                font=dict(size=18, color=colors[i % len(colors)])
            )

    # 5) Adjust layout: increase right margin so the outside text isn't clipped
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.39, y=0.05, xanchor='right', yanchor='bottom', font=dict(size=legend_font_size), ),
        margin=dict(r=66, l=5, t=5, b=5),
        xaxis=dict(
            title=dict(
                text='Time (H:MM:SS)',
                font=dict(size=xtick_title_font_size),
            ),
            tickfont=dict(size=xtick_font_size),
            tickformat='%-H:%M:%S',
            tickvals=tickvals_global,
            ticktext=tickvals_labales_global,
            tickangle=90,  # Rotate x-axis tick     labels to 90 degrees
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(
                text='Accuracy (%)',
                font=dict(size=ytick_title_font_size),
            ),
            tickfont=dict(size=ytick_font_size),
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            rangemode='nonnegative'
        ),
    )

    return fig


def generate_per_minute_csv(df, folder_path, file_name_prefix, id_prefix):
    """
    Generate per-minute CSV files in the format:
    "ID, Start Timestamp, End Timestamp, Occurrences Count"
    for each edge in the DataFrame by calculating the difference between
    consecutive minute intervals.
    """
    for edge_name in df['Edge'].unique():
        # Filter data for the specific edge
        edge_df = df[df['Edge'] == edge_name].copy()

        # Resample the data to get the largest count per minute
        per_minute = (
            edge_df
            .set_index('Timestamp')
            .sort_index()
            .resample('1T')['Count']
            .max()
            .reset_index()
            .ffill()
        )

        # Calculate the per-minute occurrences count
        per_minute['Occurrences Count'] = per_minute['Count'].diff().fillna(0).astype(int)
        per_minute['Occurrences Count'][0] = per_minute['Count'][0]

        # If there are no data points, ensure that the DataFrame is not empty
        if per_minute.empty:
            continue

        # Prepare data for CSV
        per_minute['End Timestamp'] = per_minute['Timestamp'] + pd.Timedelta(minutes=1)
        per_minute['ID'] = [f"{i}" for i in range(len(per_minute))]
        per_minute.rename(columns={'Timestamp': 'Start Timestamp'}, inplace=True)
        per_minute = per_minute[['ID', 'Start Timestamp', 'End Timestamp', 'Occurrences Count']]

        # Replace the date component with 2024-12-16
        new_date = pd.Timestamp("2024-12-16")
        per_minute["Start Timestamp"] = per_minute["Start Timestamp"].apply(
            lambda x: new_date.replace(hour=x.hour, minute=x.minute, second=x.second))
        per_minute["End Timestamp"] = per_minute["End Timestamp"].apply(
            lambda x: new_date.replace(hour=x.hour, minute=x.minute, second=x.second))

        # Save to CSV
        file_path = f"{folder_path}/{file_name_prefix}_{edge_name.lower().replace(' ', '_')}.csv"
        per_minute.to_csv(file_path, index=False, header=True)


###############################################################################
# 9) Compute & Plot Accuracy for Each Edge
###############################################################################

# We'll do it edge by edge.
for edge_name in ground_truth_df['Edge'].unique():
    # Filter ground truth for this edge
    gt_edge = ground_truth_df[ground_truth_df['Edge'] == edge_name].copy()

    # For each camera that has data for this edge, find best shift, plot accuracy
    accuracy_dfs_for_edge = []
    for cam in predictions_df['Camera'].unique():
        pred_edge_cam = predictions_df[
            (predictions_df['Edge'] == edge_name) &
            (predictions_df['Camera'] == cam)
            ].copy()
        if pred_edge_cam.empty:
            continue

        # 1) Get the final time series for that best shift
        acc_df = get_accuracy_series_for_plot(pred_edge_cam, gt_edge, freq='1T')
        accuracy_dfs_for_edge.append(acc_df)

    # 2) Plot accuracy lines for all cameras (this edge)
    if len(accuracy_dfs_for_edge) > 0:
        fig_acc = plot_accuracy_lines(accuracy_dfs_for_edge, edge_name=edge_name)
        edge_name_to_save = edge_name.lower().replace(' ', '_')
        fig_acc.write_image(f"{figures_folder}/accuracy_{edge_name_to_save}.pdf") if save_to_file else fig_acc.show()

###############################################################################
# 10) Generate per-minute CSV for ground truth
###############################################################################

generate_per_minute_csv(
    ground_truth_df,
    ground_truth_stats_folder_path,
    'products_passing_on_per_minute',
    ''
)

# Generate per-minute CSV for predictions
# Adjust the DataFrame to ensure it contains the 'Count' column for this function
predictions_df['Count'] = predictions_df['Selected Count']  # Ensure 'Count' column exists

generate_per_minute_csv(
    predictions_df,
    predicted_stats_folder_path,
    'products_passing_on_per_minute',
    ''
)
