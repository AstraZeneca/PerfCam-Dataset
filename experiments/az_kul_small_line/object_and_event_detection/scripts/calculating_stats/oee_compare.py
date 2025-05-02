import os
from functools import reduce

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

############################################
# 0) Define helper functions and global vars
############################################

xtick_font_size = 24
xtick_title_font_size = 24

ytick_font_size = 24
ytick_title_font_size = 24

legend_font_size = 17
legend_position_x = 0.5
legend_position_y = -0.495

margin_dict = dict(r=50, l=50, t=5, b=5)

xaxis_dict = dict(
    title=dict(
        text="Time (H:MM:SS)",
        font=dict(size=xtick_title_font_size),
    ),
    tickfont=dict(size=xtick_font_size),  # Adjust font size here
    tickvals=list(range(0, 80, 5)),
    ticktext=['0:00:00', '0:05:00', '0:10:00', '0:15:00', '0:20:00', '0:25:00', '0:30:00', '0:35:00',
              '0:40:00', '0:45:00', '0:50:00', '0:55:00', '1:00:00', '1:05:00', '1:10:00'],
    tickangle=90,
    showgrid=True,
    gridwidth=1,
    gridcolor='lightgray',
    showline=True,
    linecolor="lightgray",
    range=[0, 70]
)

yaxis_dict = dict(
    title=dict(
        font=dict(size=ytick_title_font_size),
    ),
    tickfont=dict(size=ytick_font_size),
    showgrid=True,
    gridwidth=1,
    gridcolor='lightgray',
    rangemode='nonnegative',
    showline=True,
    linecolor="lightgray",
    range=[0, 1.407]
)

legend_dict = dict(
    orientation="h",  # Set the legend to horizontal
    x=legend_position_x,  # Center the legend horizontally
    y=legend_position_y,  # Position the legend below the figure
    xanchor="center",  # Align the legend's center with the x position
    yanchor="top",  # Anchor the legend's top to the y position
    font=dict(size=legend_font_size),
    # bgcolor='rgba(255, 255, 255, 0.8)'  # Optional: White background with 50% opacity
)

colors = px.colors.qualitative.Plotly


def prepare_figure(df, name, unit="fraction", xaxis_dictionary=None, yaxis_dictionary=None, margin_dictionary=None,
                   legend_dictionary=None, pred_bar_name=None, gt_bar_name=None):
    name_lower = name.lower()
    pred_column_name = f"{name_lower}_pred"
    gt_column_name = f"{name_lower}_gt"
    yaxis_title = f"{name} ({unit})"
    xaxis_dictionary = xaxis_dictionary or xaxis_dict
    yaxis_dictionary = yaxis_dictionary or yaxis_dict
    margin_dictionary = margin_dictionary or margin_dict
    legend_dictionary = legend_dictionary or legend_dict
    pred_bar_name = pred_bar_name or f"Predicted<br>{name}"
    gt_bar_name = gt_bar_name or f"Ground Truth<br>{name}"

    fig = go.Figure(data=[
        go.Bar(name=pred_bar_name, x=df['minute'], y=df[pred_column_name], opacity=0.8),
        go.Bar(name=gt_bar_name, x=df['minute'], y=df[gt_column_name], opacity=0.8)
    ])

    fig.update_layout(
        yaxis_title=yaxis_title,
        barmode='group',
        yaxis=yaxis_dictionary,
        xaxis=xaxis_dictionary,
        margin=margin_dictionary,
        legend=legend_dictionary,
        plot_bgcolor='white',
    )

    moving_average_pred = df[pred_column_name].rolling(window=5, center=True).mean().dropna()
    moving_average_gt = df[gt_column_name].rolling(window=5, center=True).mean().dropna()

    # Calculate difference rate for error bars
    error_diff_rate = abs(moving_average_pred - moving_average_gt)

    fig.add_trace(
        go.Scatter(
            x=moving_average_pred.index + 1,
            y=moving_average_pred,
            mode='lines+markers',
            name=f'5-point Rolling Mean of <br>Predictions With Error Bars',
            connectgaps=True,
            line=dict(color='#636EFA'),
            error_y=dict(
                type='data',  # Use data-based error bars
                array=error_diff_rate,  # Array of error values
                visible=True,  # Show error bars
                color='blue',  # Set error bar color to grey
                thickness=1,  # Adjust the thickness of the error bar line
                width=1.5,  # Make the caps (tips) narrower by reducing their width
            )
        )
    )

    pio.full_figure_for_development(fig, warn=False)

    return fig


############################################
# 1) Define functions to calculate metrics #
############################################
def calculate_performance(ideal_cycle_time_sec, total_count, run_time_sec, minute=None):
    """
    ideal_cycle_time_sec: Ideal cycle time (seconds) per product
    total_count: # of products in that minute
    run_time_sec: Actual production run time in seconds for that minute
    Returns performance as a fraction (unitless).
    """
    if run_time_sec <= 0 or total_count <= 0:
        return 0.0
    ideal_run_time_sec = ideal_cycle_time_sec * total_count
    return ideal_run_time_sec / run_time_sec


def calculate_quality(total_count, defective_count, minute=None):
    """
    total_count: total products in that minute
    defective_count: # of bad/defective products in that minute
    Returns fraction [0,1].
    """
    if total_count <= 0:
        return 0.0
    good_count = total_count - defective_count
    return good_count / total_count


def calculate_oee(availability, performance, quality):
    """
    availability, performance, quality: each in [0,1]
    OEE = A x P x Q
    """
    return availability * performance * quality


#############################################
# 2) Define helper to sum up defective data #
#############################################
def get_defective_count_per_minute(df_occurrences, multiplier):
    """
    Given DataFrame with columns:
      'Occurrences Count', 'Start Timestamp', 'End Timestamp', etc.
    multiplier: number of defective products per occurrence

    Returns a Series indexed by 'minute' with sum of defective counts.
    """
    df = df_occurrences.copy()

    # Ensure timestamps are in datetime (if you rely on time-based grouping)
    if not pd.api.types.is_datetime64_any_dtype(df['Start Timestamp']):
        df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'], errors='coerce')

    # Use row index as 'minute' or parse from timestamp:
    df['minute'] = df.index + 1
    # Each occurrence = multiplier * defective
    df['defective_count'] = df['Occurrences Count'] * multiplier

    # Sum over each minute
    defective_count_per_minute = df.groupby('minute')['defective_count'].sum()
    return defective_count_per_minute


##############################################################
# 3) Main function to read CSVs, compute OEE per minute, etc #
##############################################################
def main(output_folder="../../stats/figures/oee", edges_to_read=None, good_edge=None):
    """
    :param output_folder: where to save OEE_per_minute.csv and charts (PDF)
    :param edges_to_read: list of edge numbers, e.g. [1, 2, 3]
    :param good_edge: the "edge number" that corresponds to good products (for OEE)

    Example usage:
       main(output_folder="../../stats/figures/oee", edges_to_read=[1,2], good_edge=2)
    """

    if edges_to_read is None:
        # Default: read just edges 1 and 2
        edges_to_read = [1, 2]
    if good_edge not in edges_to_read:
        raise ValueError(f"good_edge={good_edge} not in edges_to_read={edges_to_read}")

    # -----------------------------
    # (A) Read the predicted Availability CSV
    # -----------------------------
    # Format: 'Start Timestamp', 'End Timestamp', 'Availability (%)', 'Stop Time'
    # => We'll call them "predicted" or "calc" availability.
    availability_csv_pred = "../../stats/csv/calculated/availability_per_minute.csv"
    df_avail_pred = pd.read_csv(availability_csv_pred)

    # Convert timestamps
    df_avail_pred['Start Timestamp'] = pd.to_datetime(df_avail_pred['Start Timestamp'], errors='coerce')
    df_avail_pred['End Timestamp'] = pd.to_datetime(df_avail_pred['End Timestamp'], errors='coerce')

    df_avail_pred['minute'] = df_avail_pred.index + 1
    df_avail_pred.rename(columns={
        'Availability (%)': 'availability_pred_percent',
        'Stop Time': 'stop_time_pred_sec'
    }, inplace=True)

    # df_avail_pred['availability_gt'] = 1.0  # assume always available
    # df_avail_pred['stop_time_gt_sec'] = 0.0

    # -----------------------------
    # (B) Read defective occurrences (predicted)
    # -----------------------------
    # Summing from multiple event CSVs (as in your original code).
    triple_double_pickup_csv = "../../stats/csv/ground_truth/events/triple_double_pickup/node_2/occurrences_per_minute.csv"
    triple_pickup_csv = "../../stats/csv/ground_truth/events/triple_pickup/node_2/occurrences_per_minute.csv"
    pickup_csv = "../../stats/csv/ground_truth/events/pickup/node_2/occurrences_per_minute.csv"
    double_handed_pickup_csv = "../../stats/csv/ground_truth/events/double_handed_pickup/node_2/occurrences_per_minute.csv"
    double_pickup_csv = "../../stats/csv/ground_truth/events/double_pickup/node_2/occurrences_per_minute.csv"

    def load_occurrences_if_exists(path, multiplier):
        if os.path.exists(path):
            tmp = pd.read_csv(path)
            return get_defective_count_per_minute(tmp, multiplier)
        else:
            print(f"[WARN] Path not found: {path}")
            return pd.Series(dtype='float64')  # empty

    # Summed defective for predicted:
    dfs_def_pred = []
    dfs_def_pred.append(load_occurrences_if_exists(triple_double_pickup_csv, 3))
    dfs_def_pred.append(load_occurrences_if_exists(triple_pickup_csv, 3))
    dfs_def_pred.append(load_occurrences_if_exists(pickup_csv, 1))
    dfs_def_pred.append(load_occurrences_if_exists(double_handed_pickup_csv, 2))
    dfs_def_pred.append(load_occurrences_if_exists(double_pickup_csv, 2))

    df_defective_pred = pd.concat(dfs_def_pred, axis=0).groupby(level=0).sum()
    df_defective_pred.name = 'defective_pred_count'
    df_defective_pred = df_defective_pred.reset_index()  # columns: minute, defective_pred_count

    # If you also have ground-truth defective from a separate set of event CSVs,
    # you could do the same approach. For simplicity, let's assume the same
    # events are also ground truth (not typical in real scenario, but just example).
    # If you had separate files, you'd load them similarly.
    df_defective_gt = df_defective_pred.copy()
    df_defective_gt.rename(columns={'defective_pred_count': 'defective_gt_count'}, inplace=True)

    # -----------------------------
    # (C) Read predicted product counts for each edge
    # -----------------------------
    edge_dfs_pred = []
    for e in edges_to_read:
        edge_csv = f"../../stats/csv/calculated/products_passing_on_per_minute_edge_{e}.csv"
        if os.path.exists(edge_csv):
            df_edge = pd.read_csv(edge_csv)
            df_edge['minute'] = df_edge.index + 1
            df_edge_agg = df_edge.groupby('minute')['Occurrences Count'].sum().reset_index()
            df_edge_agg.rename(columns={'Occurrences Count': f'edge_{e}_pred_count'}, inplace=True)
            edge_dfs_pred.append(df_edge_agg)
        else:
            print(f"[WARN] Predicted CSV not found for edge {e}: {edge_csv}")
            df_dummy = pd.DataFrame({'minute': [], f'edge_{e}_pred_count': []})
            edge_dfs_pred.append(df_dummy)

    if len(edge_dfs_pred) > 0:
        df_edges_merged_pred = reduce(
            lambda left, right: pd.merge(left, right, on='minute', how='outer'),
            edge_dfs_pred
        )
    else:
        df_edges_merged_pred = pd.DataFrame({'minute': []})

    # -----------------------------
    # (C2) Read ground-truth product counts for each edge
    # -----------------------------
    edge_dfs_gt = []
    for e in edges_to_read:
        edge_csv_gt = f"../../stats/csv/ground_truth/products_passing_on_per_minute_edge_{e}.csv"
        if os.path.exists(edge_csv_gt):
            df_gt_edge = pd.read_csv(edge_csv_gt)
            df_gt_edge['minute'] = df_gt_edge.index + 1
            df_gt_edge_agg = df_gt_edge.groupby('minute')['Occurrences Count'].sum().reset_index()
            df_gt_edge_agg.rename(columns={'Occurrences Count': f'edge_{e}_gt_count'}, inplace=True)
            edge_dfs_gt.append(df_gt_edge_agg)
        else:
            raise Exception(f"[ERROR] Ground truth CSV not found for edge {e}: {edge_csv_gt}")
            # df_dummy = pd.DataFrame({'minute': [], f'edge_{e}_gt_count': []})
            # edge_dfs_gt.append(df_dummy)

    if len(edge_dfs_gt) > 0:
        df_edges_merged_gt = reduce(
            lambda left, right: pd.merge(left, right, on='minute', how='outer'),
            edge_dfs_gt
        )
    else:
        df_edges_merged_gt = pd.DataFrame({'minute': []})

    # -----------------------------
    # (D) Merge predicted + GT availability/defective/edges
    # -----------------------------
    df_merged = df_avail_pred.merge(df_defective_pred, on='minute', how='outer')
    df_merged = df_merged.merge(df_defective_gt, on='minute', how='outer')
    df_merged = df_merged.merge(df_edges_merged_pred, on='minute', how='outer')
    df_merged = df_merged.merge(df_edges_merged_gt, on='minute', how='outer')

    # Fill missing with 0
    cols_fill_zero = [c for c in df_merged.columns if 'count' in c or 'stop_time' in c or 'availability' in c]
    df_merged[cols_fill_zero] = df_merged[cols_fill_zero].fillna(0.0)

    # Sort by minute ascending
    df_merged.sort_values('minute', inplace=True)

    # -----------------------------
    # (E) Compute predicted OEE measures
    # -----------------------------
    # For predicted:
    df_merged['run_time_pred_sec'] = 60.0 - df_merged['stop_time_pred_sec']
    df_merged['availability_pred'] = df_merged['availability_pred_percent'] / 100.0

    # Predicted good count from the chosen "good_edge"
    good_col_pred = f'edge_{good_edge}_pred_count'
    if good_col_pred not in df_merged.columns:
        df_merged[good_col_pred] = 0.0

    df_merged['good_pred_count'] = df_merged[good_col_pred]
    df_merged['total_pred_count'] = df_merged['defective_pred_count'] + df_merged['good_pred_count']

    # Predicted Performance, Quality, OEE
    ideal_cycle_time_sec = 1.1

    df_merged['performance_pred'] = df_merged.apply(
        lambda row: calculate_performance(ideal_cycle_time_sec,
                                          row['total_pred_count'],
                                          60,
                                          row['minute']),
        axis=1
    )
    df_merged['quality_pred'] = df_merged.apply(
        lambda row: calculate_quality(row['total_pred_count'], row['defective_pred_count'], row['minute']),
        axis=1
    )
    df_merged['oee_pred'] = df_merged.apply(
        lambda row: calculate_oee(row['availability_pred'], row['performance_pred'], row['quality_pred']),
        axis=1
    )

    # -----------------------------
    # (F) Compute ground-truth OEE measures
    # -----------------------------
    # We'll assume ground-truth availability = availability extracted with sensors
    df_merged['run_time_gt_sec'] = df_merged['run_time_pred_sec']
    df_merged['availability_gt'] = df_merged['availability_pred']

    good_col_gt = f'edge_{good_edge}_gt_count'
    if good_col_gt not in df_merged.columns:
        df_merged[good_col_gt] = 0.0

    df_merged['good_gt_count'] = df_merged[good_col_gt]
    df_merged['total_gt_count'] = df_merged['defective_gt_count'] + df_merged['good_gt_count']

    df_merged['performance_gt'] = df_merged.apply(
        lambda row: calculate_performance(ideal_cycle_time_sec, row['total_gt_count'], 60),
        axis=1
    )
    df_merged['quality_gt'] = df_merged.apply(
        lambda row: calculate_quality(row['total_gt_count'], row['defective_gt_count']),
        axis=1
    )
    df_merged['oee_gt'] = df_merged.apply(
        lambda row: calculate_oee(row['availability_gt'], row['performance_gt'], row['quality_gt']),
        axis=1
    )

    # -----------------------------
    # (G) Compute error (absolute difference) for each measure
    # -----------------------------
    df_merged['availability_error'] = abs(df_merged['availability_pred'] - df_merged['availability_gt'])
    df_merged['performance_error'] = abs(df_merged['performance_pred'] - df_merged['performance_gt'])
    df_merged['quality_error'] = abs(df_merged['quality_pred'] - df_merged['quality_gt'])
    df_merged['oee_error'] = abs(df_merged['oee_pred'] - df_merged['oee_gt'])

    # -----------------------------
    # (H) Save final CSV
    # -----------------------------
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_csv_path = os.path.join(output_folder + "/../../../csv/calculated", "OEE_per_minute.csv")

    columns_to_save = [
                          'minute',
                          'availability_pred', 'performance_pred', 'quality_pred', 'oee_pred',
                          'availability_gt', 'performance_gt', 'quality_gt', 'oee_gt',
                          'availability_error', 'performance_error', 'quality_error', 'oee_error',
                          'total_pred_count', 'total_gt_count',
                          'defective_pred_count', 'defective_gt_count'
                      ] + [c for c in df_merged.columns if 'edge_' in c and c.endswith('_count')]
    columns_to_save = list(dict.fromkeys(columns_to_save))  # remove duplicates, keep order

    df_out = df_merged[columns_to_save].copy().iloc[:-2]  # Removing last 2 minutes data as it's incomplete
    df_out.to_csv(output_csv_path, index=False)
    print(f"Saved OEE per minute CSV to: {output_csv_path}")

    pio.kaleido.scope.mathjax = None

    # -----------------------------
    # (I) Create bar charts: Pred vs GT for availability, performance, quality, OEE
    # -----------------------------
    fig_avail = prepare_figure(df=df_out, name="Availability")
    fig_perf = prepare_figure(df=df_out, name="Performance",
                              pred_bar_name='<span style="font-size: 14px">Predicted<br>Performance</span>')
    fig_quality = prepare_figure(df=df_out, name="Quality")
    fig_oee = prepare_figure(df=df_out, name="OEE")
    fig_error = go.Figure(data=[
        go.Bar(name='Predicted OEE Error<br>vs. Ground Truth', x=df_out['minute'],
               y=df_out['oee_error'], showlegend=True, opacity=0.8),
    ])

    yaxis_dict["tickvals"] = [0, 0.2, 0.4, 0.6, 0.8, 1]
    yaxis_dict["ticktext"] = ['0%', '20%', '40%', '60%', '80%', '100%']
    yaxis_dict["range"] = [0, 1.01]
    yaxis_dict['tickfont']["size"] = ytick_font_size

    fig_error.update_layout(
        yaxis_title="OEE Absolute Error (%)",
        barmode='group',
        margin=margin_dict,
        yaxis=yaxis_dict,
        xaxis=xaxis_dict,
        legend=legend_dict,
        plot_bgcolor='white',
    )

    moving_average_pred = df_out['oee_error'].rolling(window=5, center=True).mean().dropna()

    fig_error.add_trace(
        go.Scatter(
            x=moving_average_pred.index + 1,
            y=moving_average_pred,
            mode='lines+markers',
            name=f'5-point Rolling Mean<br>of Predictions',
            connectgaps=True,
            line=dict(color='#636EFA'),
        )
    )

    # Compute mean accuracy and draw a horizontal line
    total_gt_count = df_out['total_gt_count'].sum()
    total_pred_count = df_out['total_pred_count'].sum()
    total_pred_defect_count = df_out['defective_pred_count'].sum()
    total_pred_good_count = total_pred_count - total_pred_defect_count
    total_gt_defect_count = df_out['defective_gt_count'].sum()
    total_gt_good_count = total_gt_count - total_gt_defect_count
    total_planned = df_out.last_valid_index() + 1
    total_operating = df_out['availability_gt'].mean() * total_planned
    total_operating_sec = total_operating * 60
    total_planned_sec = total_planned * 60

    total_oee_pred = ((total_operating_sec / total_planned_sec) *
                      ((ideal_cycle_time_sec * total_pred_count) / total_planned_sec) *
                      (total_pred_good_count / total_pred_count))

    total_oee_gt = ((total_operating_sec / total_planned_sec) *
                      ((ideal_cycle_time_sec * total_gt_count) / total_planned_sec) *
                      (total_gt_good_count / total_gt_count))

    # mean_err = df_out['oee_error'].mean(skipna=True) # Incorrect calculating OEE avg, due to uneven count per minute
    mean_err = 1 - (total_oee_pred / total_oee_gt)

    if pd.notna(mean_err):
        # Plot the dotted line at the exact mean
        fig_error.add_hline(

            y=mean_err,
            line_dash='dot',
            line_color="#636EFA",
        )

        fig_error.add_annotation(
            x=1.00,
            xref='paper',
            xanchor='left',
            yref='y',
            y=0.04,
            text=f"{mean_err * 100:.1f}%",  # no extra text, just the numeric %
            showarrow=False,
            font=dict(size=18, color="#636EFA")
        )

    pio.full_figure_for_development(fig_error, warn=False)

    # -----------------------------
    # (J) Save the figures as PDF
    # -----------------------------
    fig_avail.write_image(os.path.join(output_folder, "availability_pred_vs_gt.pdf"), format="pdf")
    fig_perf.write_image(os.path.join(output_folder, "performance_pred_vs_gt.pdf"), format="pdf")
    fig_quality.write_image(os.path.join(output_folder, "quality_pred_vs_gt.pdf"), format="pdf")
    fig_oee.write_image(os.path.join(output_folder, "oee_pred_vs_gt.pdf"), format="pdf")
    fig_error.write_image(os.path.join(output_folder, "oee_error_rate.pdf"), format="pdf")

    print(f"Saved all PDF figures under: {output_folder}")


########################################
# 4) Entry point to run the script
########################################
if __name__ == "__main__":
    # Example usage
    main(
        output_folder="../../stats/figures/oee/compare",
        edges_to_read=[1, 2],
        good_edge=2
    )
