#!/usr/bin/env python3

import csv
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd

# -------------------------------------------------------------------------
# 1) Configuration
# -------------------------------------------------------------------------

CSV_FILE = "../events/ground_truth_events.csv"
CAMERA_FILES = [
    "../camera_footage/1.mp4",
    "../camera_footage/2.mp4",
    "../camera_footage/3.mp4",
    "../camera_footage/4.mp4",
]

VIDEO_OUTPUT_DIR = os.path.expanduser("../events/video_clips")
CSV_OUTPUT_DIR = os.path.expanduser("../stats/csv/ground_truth/events")

GENERATE_VIDEOS = False
GENERATE_CSV = True

# Define your start/end action pairs
ACTION_PAIRS = {
    "cross adjustment starts":        ["cross adjustment ends"],
    "crawled inside room":            ["crawled outside room"],
    "camera rotation starts":         ["camera rotation ends"],
    "double handed adjusting starts": ["double handed adjusting ends"],
    "line jamming starts":            ["line jammed"],
    "line preparing to start":        ["line started"],
    "door closing starts":            ["door opening starts", "door closed"],
    "exiting room":                   ["exited room"],
    "entering room":                  ["entered room"],
    "preparing to focus starts":      ["focused"],
    "door opening starts":            ["door closing starts", "door opened"],
    "double handed pickup starts":    ["double handed pickup ends"],
    "line unjamming":                 ["line unjammed"],
    "pickup starts":                  ["pickup ends"],
    "leaving starts":                 ["left"],
    "triple pickup starts":           ["triple pickup ends"],
    "distracting starts":             ["distracted"],
    "line preparing to stop":         ["line stopped"],
    "adjusting starts":               ["adjusting ends"],
    "returning starts":               ["returned"],
    "double pickup starts":           ["double pickup ends"],
    "triple-double pickup starts":    ["triple-double pickup ends"],
}

# -------------------------------------------------------------------------
# 2) Helper Functions
# -------------------------------------------------------------------------

def parse_timestamp_to_seconds(timestr: str) -> float:
    """
    Parses 'HH:MM:SS.fff' (or .ffffff) into total seconds as a float.
    """
    dt = datetime.strptime(timestr, "%H:%M:%S.%f")
    delta = timedelta(
        hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond
    )
    return delta.total_seconds()


def calculate_duration(start_time_str, end_time_str) -> float:
    """
    Returns the difference in seconds between two timestamps in HH:MM:SS.fff format.
    """
    fmt = "%H:%M:%S.%f"
    start_dt = datetime.strptime(start_time_str, fmt)
    end_dt = datetime.strptime(end_time_str, fmt)
    return (end_dt - start_dt).total_seconds()


def generate_per_minute_breakdown_pandas(occurrences_list, total_minutes=70):
    """
    Given a list of dicts: [{"ID", "Start Timestamp", "End Timestamp"}, ...],
    produce EXACTLY 'total_minutes' rows (e.g. 70 for 1:10), each with:
      Start Timestamp, End Timestamp, Occurrences Count
    where Occurrences Count = how many intervals overlap that minute.
    """

    # Convert to a DataFrame
    df = pd.DataFrame(occurrences_list)  # columns: ID, Start Timestamp, End Timestamp

    if df.empty:
        # If no occurrences, just produce zero-occurrence rows
        # But we still produce 70 rows with 0
        rows = []
        zero_dt = datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
        for minute_idx in range(total_minutes):
            minute_start = zero_dt + timedelta(minutes=minute_idx)
            minute_end = minute_start + timedelta(seconds=59.999999)
            rows.append({
                "Start Timestamp": minute_start.strftime("%H:%M:%S.%f")[:-3],
                "End Timestamp": minute_end.strftime("%H:%M:%S.%f")[:-3],
                "Occurrences Count": 0
            })
        return pd.DataFrame(rows)

    # Helper to parse times
    def time_to_offset_seconds(tstr):
        dt = datetime.strptime(tstr, "%H:%M:%S.%f")
        return (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

    # Add numeric columns for overlap checking
    df["start_s"] = df["Start Timestamp"].apply(time_to_offset_seconds)
    df["end_s"] = df["End Timestamp"].apply(time_to_offset_seconds)

    # We'll build a list of rows for minute 0..(total_minutes - 1)
    results = []
    zero_dt = datetime.strptime("00:00:00.000", "%H:%M:%S.%f")

    for minute_idx in range(total_minutes):
        minute_start_s = minute_idx * 60.0
        minute_end_s = minute_start_s + 60.0 - 0.001  # ~59.999

        # For each interval in df, check overlap
        overlap_count = 0
        for _, row in df.iterrows():
            overlap_start = max(minute_start_s, row["start_s"])
            overlap_end = min(minute_end_s, row["end_s"])
            if overlap_end > overlap_start:
                overlap_count += 1

        # Format HH:MM:SS.mmm for the minute boundaries
        minute_start_dt = zero_dt + timedelta(seconds=minute_start_s)
        minute_end_dt = zero_dt + timedelta(seconds=minute_end_s)

        results.append({
            "Start Timestamp": minute_start_dt.strftime("%H:%M:%S.%f")[:-3],
            "End Timestamp": minute_end_dt.strftime("%H:%M:%S.%f")[:-3],
            "Occurrences Count": overlap_count
        })

    return pd.DataFrame(results)


# -------------------------------------------------------------------------
# 3) Main Script
# -------------------------------------------------------------------------

def main():
    # 3a) Clean up CSV_OUTPUT_DIR before generating new files
    if os.path.exists(CSV_OUTPUT_DIR):
        shutil.rmtree(CSV_OUTPUT_DIR)

    # Also ensure the video output directory exists
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    # We'll recreate CSV_OUTPUT_DIR fresh
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    # Read the "ground_truth_events.csv" data
    rows = []
    with open(CSV_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    num_rows = len(rows)
    i = 0

    # We'll store all intervals in memory, keyed by (subfolder_name, camera_index).
    # That way, we can generate occurrences.csv and occurrences_per_minute.csv at the end.
    occurrences_storage = defaultdict(list)
    occurrences_storage_per_node = defaultdict(list)

    # We'll keep a global auto-increment ID across all intervals
    occurrence_id_counter = 1

    while i < num_rows:
        start_action_raw = rows[i]["Action"].strip()
        start_action = start_action_raw.lower()

        # If it's a recognized start action
        if start_action in ACTION_PAIRS:
            end_actions = ACTION_PAIRS[start_action]
            start_time = rows[i]["Timestamp"].strip()

            # Optional offset to get a slightly earlier cut
            start_time_seconds = parse_timestamp_to_seconds(start_time) - 0.25
            adjusted_start_str = f"{start_time_seconds:.3f}"

            j = i + 1
            matched_end_row = None
            found_end_action = None

            # Search forward for a matching end
            while j < num_rows:
                possible_end = rows[j]["Action"].strip().lower()
                for ea in end_actions:
                    if possible_end == ea:
                        matched_end_row = rows[j]
                        found_end_action = ea
                        break
                if found_end_action is not None:
                    break
                j += 1

            if matched_end_row:
                end_time = matched_end_row["Timestamp"].strip()
                start_id = rows[i]["ID"].strip()
                end_id = matched_end_row["ID"].strip()

                # Derive subfolder name from "start_action_raw"
                subfolder_name = start_action_raw
                if subfolder_name.lower().endswith(" starts"):
                    subfolder_name = subfolder_name[:-len(" starts")]

                video_action_folder = os.path.join(VIDEO_OUTPUT_DIR, subfolder_name).replace(" ", "_").replace("-", "_")
                stats_action_folder = os.path.join(CSV_OUTPUT_DIR, subfolder_name).replace(" ", "_").replace("-", "_")
                os.makedirs(video_action_folder, exist_ok=True)

                duration = calculate_duration(start_time, end_time)

                # For each camera
                for camera_index, camera_file in enumerate(CAMERA_FILES, start=1):
                    # ------------------------------------------------------------------
                    # 1) If GENERATE_VIDEOS=True, produce mp4 cut in the appropriate folder
                    # ------------------------------------------------------------------
                    if GENERATE_VIDEOS:
                        folder_video = os.path.join(video_action_folder, str(camera_index)).replace(" ", "_").replace("-", "_")
                        os.makedirs(folder_video, exist_ok=True)

                        output_filename = f"{start_id}_{end_id}.mp4"
                        output_path = os.path.join(folder_video, output_filename)

                        cmd_final = [
                            "ffmpeg",
                            "-y",
                            "-ss", adjusted_start_str,  # slightly earlier start
                            "-i", camera_file,
                            "-t", str(duration + 0.4),  # ensure no frame is missed
                            output_path,
                        ]
                        print("Running ffmpeg:", " ".join(cmd_final))
                        subprocess.run(cmd_final, check=True)

                    # ------------------------------------------------------------------
                    # 2) Store the interval in memory for CSV generation
                    # ------------------------------------------------------------------
                    # Iterate over keys that start with 'node'
                    if not occurrences_storage[subfolder_name] or occurrences_storage[subfolder_name][-1]["Start Timestamp"] !=  start_time:
                        occurrences_storage[subfolder_name].append({
                            "ID": occurrence_id_counter,
                            "Start Timestamp": start_time,
                            "End Timestamp": end_time,
                        })
                        for node_index in rows[i]:
                            if node_index.startswith('node'):
                                if rows[i][node_index] == '1':
                                    occurrences_storage_per_node[(subfolder_name, node_index)].append({
                                        "ID": occurrence_id_counter,
                                        "Start Timestamp": start_time,
                                        "End Timestamp": end_time,
                                    })

                # After processing all cameras for this interval, increment the ID
                occurrence_id_counter += 1

                # Skip past the matched end
                i = j + 1
                continue

        i += 1

    # ----------------------------------------------------------------------
    # 4) Now that we've collected all intervals in memory, generate the CSVs
    # ----------------------------------------------------------------------
    if GENERATE_CSV:
        for subfolder_name, intervals_list in occurrences_storage.items():
            stats_action_folder = os.path.join(CSV_OUTPUT_DIR, subfolder_name).replace(" ", "_").replace("-", "_")
            occurrences_folder = os.path.join(stats_action_folder)
            os.makedirs(occurrences_folder, exist_ok=True)

            # (A) Write occurrences.csv (ID,Start Timestamp,End Timestamp)
            occurrences_csv_path = os.path.join(occurrences_folder, "occurrences.csv")
            if not os.path.isfile(occurrences_csv_path):
                with open(occurrences_csv_path, mode="w", newline="", encoding="utf-8") as occ_file:
                    writer = csv.writer(occ_file)
                    writer.writerow(["ID", "Start Timestamp", "End Timestamp"])
                    for row_data in intervals_list:
                        writer.writerow([
                            row_data["ID"],
                            row_data["Start Timestamp"],
                            row_data["End Timestamp"]
                        ])
                # Process the first CSV file
                df = pd.read_csv(occurrences_csv_path)
                df['ID'] = range(0, len(df))
                # Move 'ID' to the first column
                df = df[['ID'] + [col for col in df.columns if col != 'ID']]
                df.to_csv(occurrences_csv_path, index=False)
            else:
                break


        for (subfolder_name, node_index), intervals_list in occurrences_storage_per_node.items():
            # Make sure the folder structure exists
            stats_action_folder = os.path.join(CSV_OUTPUT_DIR, subfolder_name)
            folder_stats = os.path.join(stats_action_folder, str(node_index)).replace(" ", "_").replace("-", "_")
            os.makedirs(folder_stats, exist_ok=True)


            # (B) Use pandas to produce occurrences_per_minute.csv
            #     -> 70 rows, each row "Start Timestamp,End Timestamp,Occurrences Count"
            pm_df = generate_per_minute_breakdown_pandas(intervals_list, total_minutes=70)
            occurrences_per_minute_csv_path = os.path.join(folder_stats, "occurrences_per_minute.csv").replace("-", "_")
            pm_df.to_csv(occurrences_per_minute_csv_path, index=False)

            # Process the second CSV file
            df = pd.read_csv(occurrences_per_minute_csv_path)
            df['ID'] = range(0, len(df))
            # Move 'ID' to the first column
            df = df[['ID'] + [col for col in df.columns if col != 'ID']]
            df.to_csv(occurrences_per_minute_csv_path, index=False)




if __name__ == "__main__":
    main()
