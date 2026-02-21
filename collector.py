import time
import csv
import threading
import pandas as pd
from pynput import keyboard

# Globals for tracking state
press_times = {}
flight_times = {}
last_release_time = None
current_csv = ""
listener = None
stop_session = False
esc_pressed = False


def get_key_name(key):
    try:
        return key.char
    except AttributeError:
        return str(key).replace("Key.", "<") + ">"


def on_press(key):
    global last_release_time, esc_pressed
    if key == keyboard.Key.esc:
        esc_pressed = True
        return False  # Stops listener immediately

    timestamp = time.time()
    key_name = get_key_name(key)
    if key_name not in press_times:
        press_times[key_name] = timestamp
        flight_times[key_name] = (timestamp - last_release_time) if last_release_time else 0.0


def on_release(key):
    global last_release_time
    release_time = time.time()
    key_name = get_key_name(key)
    if key_name in press_times:
        dwell_time = release_time - press_times[key_name]
        flight_time = flight_times.get(key_name, 0.0)
        with open(current_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([key_name, f"{flight_time:.4f}", f"{dwell_time:.4f}"])
        del press_times[key_name]
    last_release_time = release_time


def stop_listener():
    print("\n⏱️ 10 seconds are up!")
    if listener: listener.stop()


# --- 1. Data Collection Phase ---
u_id = 1
collected_users = []

print("Press [Esc] at any prompt or during typing to stop adding new users.")

while not esc_pressed:
    current_csv = f"keystroke_data_User_{u_id}.csv"
    press_times.clear()
    last_release_time = None

    with open(current_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Flight_Time_s", "Dwell_Time_s"])

    print(f"\n--- User {u_id} ---")
    input(f"User {u_id}, press [ENTER] to start 10s timer (or [Esc] then [Enter] to quit)...")

    if esc_pressed: break

    timer = threading.Timer(10.0, stop_listener)
    timer.start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
        listener = l
        listener.join()

    collected_users.append(u_id)
    u_id += 1

# --- 2. Data Combination Phase ---
print("\nCombining all files...")
all_dfs = []

for uid in collected_users:
    file_name = f"keystroke_data_User_{uid}.csv"
    try:
        df = pd.read_csv(file_name)
        if not df.empty:
            df['ID'] = uid
            all_dfs.append(df)
    except FileNotFoundError:
        continue

if all_dfs:
    full_data = pd.concat(all_dfs, ignore_index=True)

    # 1. Unordered File: No ID column, has Serial Number
    unordered = full_data.drop(columns=['ID']).copy()
    unordered.insert(0, 'SL.no.', range(1, len(unordered) + 1))
    unordered.to_csv("combined.csv", index=False)

    # 2. Ordered File: Has ID column at start, sorted by ID
    ordered = full_data.sort_values(by='ID').copy()
    cols = ['ID'] + [c for c in ordered.columns if c != 'ID']
    ordered = ordered[cols]
    ordered.to_csv("ord_combine.csv", index=False)

    print(f"✅ Created 'combined.csv' (Unordered, SL.no only)")
    print(f"✅ Created 'ord_combine.csv' (Ordered by ID)")
else:
    print("❌ No data collected.")
