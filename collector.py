import time
import csv
import threading
import pandas as pd
import matplotlib.pyplot as plt
from pynput import keyboard

# Configuration
users = ['A', 'B', 'C']
user_colors = {'A': 'blue', 'B': 'green', 'C': 'red'}

# Globals for tracking state
press_times = {}
flight_times = {}
last_release_time = None
current_csv = ""
listener = None


def get_key_name(key):
    try:
        return key.char
    except AttributeError:
        return str(key).replace("Key.", "<") + ">"


def on_press(key):
    global last_release_time
    timestamp = time.time()
    key_name = get_key_name(key)

    if key_name not in press_times:
        press_times[key_name] = timestamp

        flight_time = 0.0
        if last_release_time:
            flight_time = timestamp - last_release_time

        flight_times[key_name] = flight_time


def on_release(key):
    global last_release_time
    release_time = time.time()
    key_name = get_key_name(key)

    if key_name in press_times:
        dwell_time = release_time - press_times[key_name]
        flight_time = flight_times.get(key_name, 0.0)

        # Save data to the CURRENT user's CSV file
        with open(current_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([key_name, f"{flight_time:.4f}", f"{dwell_time:.4f}"])

        del press_times[key_name]
        if key_name in flight_times:
            del flight_times[key_name]

    last_release_time = release_time


def stop_listener():
    """Stops the keyboard listener when the timer triggers."""
    print("\n⏱️ 10 seconds are up! Stopping collection...")
    if listener is not None:
        listener.stop()


# --- 1. Data Collection Phase ---
for user in users:
    current_csv = f"keystroke_data_User_{user}.csv"

    # Reset tracking variables for the new user
    press_times.clear()
    flight_times.clear()
    last_release_time = None

    # Initialize the new user's CSV
    with open(current_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Flight_Time_s", "Dwell_Time_s"])

    print("-" * 40)
    input(f"User {user}, press [ENTER] when you are at the keyboard and ready...")
    print(f"Start typing, User {user}! You have exactly 10 seconds...")

    # Start the strict 10-second timer
    timer = threading.Timer(10.0, stop_listener)
    timer.start()

    # Start listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
        listener = l
        listener.join()

print("\nAll data collected! Generating comparison graph...")

# --- 2. Data Visualization Phase ---
plt.figure(figsize=(14, 7))

for user in users:
    csv_file = f"keystroke_data_User_{user}.csv"

    try:
        df = pd.read_csv(csv_file)
        if not df.empty:
            # Dwell Time: Solid line with circle markers
            plt.plot(df.index, df['Dwell_Time_s'],
                     label=f'User {user} (Dwell)',
                     color=user_colors[user],
                     marker='o', linestyle='-', alpha=0.8)

            # Flight Time: Dashed line with 'x' markers
            plt.plot(df.index, df['Flight_Time_s'],
                     label=f'User {user} (Flight)',
                     color=user_colors[user],
                     marker='x', linestyle='--', alpha=0.8)
    except FileNotFoundError:
        print(f"Could not find data for User {user}.")

# Graph styling
plt.title('Keystroke Rhythm Comparison: Users A, B, and C')
plt.xlabel('Keystroke Sequence (Nth Key Typed)')
plt.ylabel('Time (Seconds)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Show the interactive graph (Code pauses here until you close the graph window)
plt.show()

# --- 3. Data Combination Phase ---
print("\nCombining individual CSVs into combined_keystroke_data.csv...")

all_dfs = []
for user in users:
    csv_file = f"keystroke_data_User_{user}.csv"
    try:
        # Read the file
        df = pd.read_csv(csv_file)
        if not df.empty:
            all_dfs.append(df)
    except FileNotFoundError:
        pass

if all_dfs:
    # Concatenate everything vertically
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Insert the SL.no. as the very first column
    combined_df.insert(0, 'SL.no.', range(1, len(combined_df) + 1))

    # Save the combined dataset without pandas row indices
    combined_df.to_csv("combined_keystroke_data.csv", index=False)
    print("✅ Successfully generated combined_keystroke_data.csv!")
else:
    print("❌ No data was recorded. Combination skipped.")