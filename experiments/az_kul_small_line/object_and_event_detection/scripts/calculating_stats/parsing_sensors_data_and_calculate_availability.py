import csv
from copy import deepcopy
from datetime import timedelta

import PyPDF2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

save_to_file = True
calculated_availability_csv = '../../stats/csv/calculated/availability_per_minute.csv'
calculated_stops_csv = '../../stats/csv/calculated/stops.csv'
sensor_data_log_file_path = '../../sensors/nicla_sense_me_on_motor.log.csv'
figures_folder_path = '../../stats/figures'
use_latex = False

if not use_latex:
    pio.kaleido.scope.mathjax = None
    font_family = None
    legend_y = -0.32
    crop_size = 9
else:
    font_family = "Computer Modern"
    legend_y = -0.32
    crop_size = 17

tickfont_size = 26
titlefont_size = 28
titlefont_size_y2 = 26
legend_font_size = 25
availability_legend_font_size = 19

yaxis_tickfont_dict = dict(
    size=tickfont_size,
    family=font_family,
)

xaxis_dict = dict(
    tickfont=yaxis_tickfont_dict,
    title='Time (H:MM:SS)',
    titlefont=dict(
        size=titlefont_size,
        family=font_family,
    ),
    tickformat='%H:%M:%S',  # Format to display only hours, minutes, and seconds
    tickangle=90,  # Rotate x-axis tick     labels to 90 degrees
)

legend_dict = dict(
    orientation="h",  # Set the legend to horizontal
    x=0.5,  # Center the legend horizontally
    y=legend_y,  # Position the legend below the figure
    xanchor="center",  # Align the legend's center with the x position
    yanchor="top",  # Anchor the legend's top to the y position
    font=dict(
        size=legend_font_size,
        family=font_family,
    ),
)

availability_legend_dict = deepcopy(legend_dict)
availability_legend_dict["font"]["size"] = availability_legend_font_size


# Function to filter the lists
def filter_list(data):
    # Keep the first and last items, and every 3rd item in between
    return [data[i] for i in range(len(data)) if i == 0 or i == len(data) - 1 or (i % 3 == 0)]


# A helper function that converts a float to LaTeX-ish scientific notation
def to_latex_sci(num):
    """
    Convert a floating-point number into a string like $1.23\\times10^{-6}$.
    """

    if not use_latex:
        return num

    # Handle zero:
    if num == 0:
        return r"$0$"

    # Remember the sign separately so we can attach it if negative
    try:
        sign_str = "-" if num < 0 else ""
    except:
        return f"${num}$"

    num_abs = abs(num)

    # If the number is extremely small or large, we do the exponent logic
    # Note: log10(0) is not defined, so we handled zero above
    exp = int(np.floor(np.log10(num_abs)))
    base = num_abs / 10 ** exp

    # Choose how many decimal places for the base. Feel free to adjust:
    base_str = f"{base:.2f}"

    # If exponent is 0, just return the base (no exponent part needed).
    if exp == 0:
        value = rf"${sign_str}{base_str}$"
    else:
        # Otherwise, format in LaTeX: e.g. $1.23\times10^{-6}$
        value = rf"${sign_str}{base_str}\times10^{{{exp}}}$"

    return value


def generate_ticks(y_min, y_max, tick_count):
    if tick_count < 2:
        raise ValueError("tick_count must be greater than 2")

    if y_min > y_max:
        raise ValueError("y_min must be less than or equal to y_max")

    # Check if zero is within the range
    include_zero = y_min <= 0 <= y_max

    # Initialize the range
    total_range = y_max - y_min
    tick_spacing = total_range / (tick_count - 1)
    y_min = y_min - tick_spacing
    y_max = y_max + tick_spacing

    if include_zero:
        while True:
            # Calculate positive and negative ticks symmetrically around zero
            positive_ticks = []
            current_tick = 0
            while current_tick < y_max:
                positive_ticks.append(current_tick)
                current_tick += tick_spacing

            negative_ticks = []
            current_tick = -tick_spacing
            while current_tick > y_min:
                negative_ticks.append(current_tick)
                current_tick -= tick_spacing

            # Combine negative and positive ticks and sort them
            ticks = sorted(negative_ticks + positive_ticks)

            # if ticks[0] <= y_min:
            break
            # else:
            #     tick_spacing = total_range / (tick_count - 2)

        ticks = [round(x, 1) for x in ticks]
    else:
        # Simply generate evenly spaced ticks without including zero
        ticks = [y_min + i * tick_spacing for i in range(tick_count)]

    return ticks


# Function to read and parse the .log file
def read_log_file(file_path):
    parsed_data = []
    first_timestamp = None
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            time_stamp = int(parts[0])
            if first_timestamp is None:
                first_timestamp = time_stamp
            time_elapsed_ns = time_stamp - first_timestamp
            time_elapsed_s = time_elapsed_ns / 1e9
            formatted_time = str(timedelta(seconds=time_elapsed_s))
            values = {part.split(':')[0]: float(part.split(':')[1]) for part in parts[1:]}
            values['time'] = formatted_time
            parsed_data.append(values)
    return parsed_data


# Read data from the .log file
parsed_data = read_log_file(sensor_data_log_file_path)

# Create a DataFrame
df = pd.DataFrame(parsed_data)

# Calculate the change in accelerometer readings
df['acc_change'] = df[['acc_X', 'acc_Y', 'acc_Z']].diff().abs().sum(axis=1)

# Use a rolling window to smooth the change signal
rolling_window_size = 150
df['acc_change_smooth'] = df['acc_change'].rolling(window=rolling_window_size).mean()

# Detect running based on high changes
threshold = 100
df['running'] = df['acc_change_smooth'] >= threshold

# Create Plotly plot with secondary y-axis for availability
fig = go.Figure()

# Add accelerometer data trace
fig.add_trace(go.Scatter(
    x=df['time'],
    y=df['acc_change'],
    mode='lines',
    name='Original Change',
    opacity=0.5
))

# Add rolling mean trace
fig.add_trace(go.Scatter(
    x=df['time'],
    y=df['acc_change_smooth'],
    mode='lines',
    name='Rolling Mean of Change',
    line=dict(color='red')
))

# Add availability trace on the secondary y-axis
fig.add_trace(go.Scatter(
    x=df['time'],
    y=df['running'].astype(int),
    mode='lines',
    name='Availability',
    line=dict(color='green'),
    yaxis='y2',
))

tick_interval = 1999  # Example: every 60 indices
tickvals = filter_list(list(range(0, len(df), tick_interval)))

if (len(df) - 1) not in tickvals:
    tickvals.pop()
    tickvals.append(len(df) - 1)  # Ensure the last index is included

tick_labels = [df['time'][i].split('.')[0] for i in tickvals]

y_ticks = list(range(0, 3500, 500))
y_labels = [to_latex_sci(v) for v in y_ticks]

fig.update_xaxes(tickvals=tickvals, ticktext=tick_labels)

# Update layout with secondary y-axis for availability
fig.update_layout(
    xaxis=xaxis_dict,
    yaxis=dict(
        title='Change in Acceleration (m/s²)',
        tickfont=yaxis_tickfont_dict,
        titlefont=dict(
            size=titlefont_size,
            family=font_family,
        ),
        tickformat=".0e",  # Scientific notation with X decimal places
        exponentformat="power",
        showexponent="all",  # Show exponents on all ticks
    ),
    yaxis2=dict(
        title='Availability (1:Running,0:Stopped)',
        overlaying='y',
        side='right',
        tickvals=[0, 1],  # Show only 0 and 1 on the y-axis
        tickfont=yaxis_tickfont_dict,
        titlefont=dict(
            size=titlefont_size_y2,
            family=font_family,
        )
    ),
    legend=availability_legend_dict,
    margin=dict(r=5, l=5, t=10, b=160),  # Add some extra bottom margin to fit the legend
    height=650,  # Set the desired height of the figure (e.g., 800 pixels)
    width=800,
)


def crop_pdf(input_pdf_path, output_pdf_path, crop_pixels):
    # Open the PDF file
    with open(input_pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        writer = PyPDF2.PdfWriter()

        # Iterate through each page
        for page in reader.pages:
            # Get the current page dimensions
            current_media_box = page.mediabox
            lower_left_x = current_media_box.lower_left[0]
            lower_left_y = current_media_box.lower_left[1]
            upper_right_x = current_media_box.upper_right[0]
            upper_right_y = current_media_box.upper_right[1]

            # Set new dimensions (reduce height by crop_pixels)
            page.mediabox.lower_left = (lower_left_x, lower_left_y + crop_pixels)
            page.mediabox.upper_right = (upper_right_x, upper_right_y)

            # Add the modified page to the writer object
            writer.add_page(page)

        # Save the cropped PDF to the output file
        with open(output_pdf_path, 'wb') as output_pdf_file:
            writer.write(output_pdf_file)


# Function to create plots for a group of data
def create_group_plot(df, y_list, title):
    fig = go.Figure()

    y_min = float('inf')
    y_max = float('-inf')

    for y in y_list:
        x_data = df["time"]
        y_data = df[y]
        y_data_list = df[y].tolist()
        y_min = min(y_data_list + [y_min])
        y_max = max(y_data_list + [y_max])

        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=y))

    y_ticks = generate_ticks(y_min, y_max, 6)
    y_labels = [to_latex_sci(v) for v in y_ticks]

    tickformat_val = ".0e" if title in ['Acceleration (m/s²)', 'Magnetic Flux (μT)', 'Angular Velocity (°/s)'] else None
    exponentformat_val = "power" if title in ['Acceleration (m/s²)', 'Magnetic Flux (μT)',
                                              'Angular Velocity (°/s)'] else None
    showexponent_val = "all" if title in ['Acceleration (m/s²)', 'Magnetic Flux (μT)',
                                          'Angular Velocity (°/s)'] else None

    fig.update_layout(
        xaxis=xaxis_dict,
        yaxis=dict(
            title=f"{title}",
            tickfont=yaxis_tickfont_dict,
            titlefont=dict(
                size=titlefont_size,
                family=font_family,
            ),
            tickformat=tickformat_val,  # Scientific notation with X decimal places
            exponentformat=exponentformat_val,
            showexponent=showexponent_val,  # Show exponents on all ticks
        ),
        legend=legend_dict,
        margin=dict(r=5, l=5, t=10, b=155),
        height=650,  # Set the desired height of the figure (e.g., 800 pixels)
    )

    fig.update_xaxes(tickvals=tickvals, ticktext=tick_labels)
    return fig


# Plot sensor data grouped by type
figures = {
    'Accelerometer': create_group_plot(df, ['acc_X', 'acc_Y', 'acc_Z'], 'Acceleration (m/s²)'),
    'Gyroscope': create_group_plot(df, ['gyro_X', 'gyro_Y', 'gyro_Z'], 'Angular Velocity (°/s)'),
    'Magnetometer': create_group_plot(df, ['mag_X', 'mag_Y', 'mag_Z'], 'Magnetic Flux (μT)'),
    'Temperature': create_group_plot(df, ['Temp'], 'Temperature (°C)'),
    'Humidity': create_group_plot(df, ['Humidity'], 'Humidity (%RH)'),
    'Pressure': create_group_plot(df, ['Pressure'], 'Pressure (hPa)'),
    'CO2': create_group_plot(df, ['CO2'], 'CO2 Levels (PPM)'),
    'Availability': fig,
}

# Show plots
for name, figure in figures.items():
    pio.full_figure_for_development(figure, warn=False)
    file_to_save = f"{figures_folder_path}/{name.lower()}.pdf"
    figure.write_image(file_to_save) if save_to_file else figure.show()

    # See this bug, to understand why I'm doing this: https://github.com/plotly/plotly.py/issues/3469
    crop_pdf(file_to_save, file_to_save, crop_size)

# Identify start and end of stops using a shift operation
df['stop'] = ~df['running']
df['stop_start'] = df['stop'] & ~df['stop'].shift(fill_value=False)
df['stop_end'] = df['stop'] & ~df['stop'].shift(-1, fill_value=False)

# Get indices of stop starts and ends
stop_starts = df[df['stop_start']].index
stop_ends = df[df['stop_end']].index

# Prepare data for CSV
stop_data = []
for start, end in zip(stop_starts, stop_ends):
    start_time = df.loc[start, 'time']
    end_time = df.loc[end, 'time']
    stop_data.append({'ID': len(stop_data) + 1, 'Start Timestamp': start_time, 'End Timestamp': end_time})

# Write to CSV
with open(calculated_stops_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['ID', 'Start Timestamp', 'End Timestamp'])
    writer.writeheader()
    writer.writerows(stop_data)

######################


# Define the duration for each interval (1 minute)
interval_duration = timedelta(minutes=1)

# Convert 'time' column to timedelta for easier manipulation
df['time_timedelta'] = pd.to_timedelta(df['time'])

# Initialize variables
availability_data = []
start_time = timedelta()

# Calculate availability for each minute
for minute in range(70):  # For a total of 70 minutes
    end_time = start_time + interval_duration

    # Filter data within the current interval
    interval_data = df[(df['time_timedelta'] >= start_time) & (df['time_timedelta'] < end_time)]

    if not interval_data.empty:
        # Select rows where 'stop' is True
        stop_true_data = interval_data[interval_data['stop'] == True]

        # Calculate availability as the percentage of 'running' time
        stopping_time = stop_true_data['time_timedelta'].diff().sum()

        # Convert the sum to seconds
        total_stopping_seconds = stopping_time.total_seconds()

        total_time = 60
        availability_percentage = ((total_time - total_stopping_seconds) / total_time) * 100

        # Format start and end time as 'HH:MM:SS.sss'
        formatted_start_time = str(start_time)
        formatted_end_time = str(end_time)

        # Prepare data for CSV
        availability_data.append({
            'Start Timestamp': formatted_start_time,
            'End Timestamp': formatted_end_time,
            'Availability (%)': availability_percentage,
            'Stop Time': total_stopping_seconds,
        })

    # Move to the next interval
    start_time = end_time

# Write to CSV
with open(calculated_availability_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Start Timestamp', 'End Timestamp', 'Availability (%)', 'Stop Time'])
    writer.writeheader()
    writer.writerows(availability_data)
