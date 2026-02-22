import time
import csv
import threading
import pandas as pd
from pynput import keyboard
import glob
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

#phase 1, data collection.
print("\n" + "=" * 50)
print("INITIALIZING KEYSTROKE DYNAMICS ENGINE")
print("Passwordless Client-Side Authentication Pipeline")
print("=" * 50)

# Globals for tracking advanced state
active_keys = {}
last_release_time = 0.0
last_press_time = 0.0
last_key_pressed = ""
current_csv = ""
listener = None
esc_pressed = False


def get_key_name(key):
    try:
        return key.char
    except AttributeError:
        return str(key).replace("Key.", "<") + ">"


def on_press(key):
    global last_press_time, last_release_time, last_key_pressed, esc_pressed
    if key == keyboard.Key.esc:
        esc_pressed = True
        return False

    timestamp = time.time()
    key_name = get_key_name(key)

    if key_name not in active_keys:
        # Calculate Advanced Latency Metrics
        flight_dd = (timestamp - last_press_time) if last_press_time else 0.0
        flight_ud = (timestamp - last_release_time) if last_release_time else 0.0

        # Determine if this key was pressed before the previous one was released
        is_overlap = 1 if (last_release_time == 0.0 or last_press_time > last_release_time) else 0

        # Create Digraph (Key Pair)
        key_pair = f"{last_key_pressed}_{key_name}" if last_key_pressed else "START"

        # Store metrics at the exact moment of press
        active_keys[key_name] = {
            'press_time': timestamp,
            'flight_dd': flight_dd,
            'flight_ud': flight_ud,
            'is_overlap': is_overlap,
            'key_pair': key_pair
        }

        last_press_time = timestamp
        last_key_pressed = key_name


def on_release(key):
    global last_release_time
    release_time = time.time()
    key_name = get_key_name(key)

    if key_name in active_keys:
        data = active_keys[key_name]
        dwell_time = release_time - data['press_time']

        # Write the enriched data row
        with open(current_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                key_name,
                data['key_pair'],
                f"{data['flight_ud']:.4f}",
                f"{data['flight_dd']:.4f}",
                f"{dwell_time:.4f}",
                data['is_overlap']
            ])
        del active_keys[key_name]

    last_release_time = release_time


def stop_listener():
    print("\n⏱️ Time is up! Stopping collection...")
    if listener: listener.stop()


# Find next available ID
existing_files = glob.glob("keystroke_data_User_*.csv")
max_id = 0
for f in existing_files:
    match = re.search(r'User_(\d+)\.csv', f)
    if match: max_id = max(max_id, int(match.group(1)))

u_id = max_id + 1

print("\n--- PHASE 1: DATA GATHERING ---")
print("Press [Esc] at the prompt to end data collection and proceed to ML Training.")
if max_id > 0: print(f"📁 Found existing profiles up to User {max_id}.")

while not esc_pressed:
    current_csv = f"keystroke_data_User_{u_id}.csv"
    active_keys.clear()
    last_release_time = 0.0
    last_press_time = 0.0
    last_key_pressed = ""

    with open(current_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Key_Pair", "Flight_UD_s", "Flight_DD_s", "Dwell_Time_s", "Is_Overlap"])

    # UPGRADE: 60 Second Timer for robust data collection
    input(f"\nUser {u_id} Ready? Press [ENTER] to start 60s timer (or [Esc] then [Enter] to skip to ML)...")
    if esc_pressed:
        if os.path.exists(current_csv): os.remove(current_csv)
        break

    timer = threading.Timer(60.0, stop_listener)
    timer.start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
        listener = l
        listener.join()
    u_id += 1

#phase 2: combination
print("\n--- PHASE 2: DATA COMBINATION ---")
all_dfs = []
all_existing_files = glob.glob("keystroke_data_User_*.csv")

for file_name in all_existing_files:
    match = re.search(r'User_(\d+)\.csv', file_name)
    if match:
        try:
            df = pd.read_csv(file_name)
            if not df.empty:
                df['ID'] = int(match.group(1))
                all_dfs.append(df)
        except FileNotFoundError:
            continue

if not all_dfs:
    print("❌ No data found. Exiting pipeline.")
    exit()

full_data = pd.concat(all_dfs, ignore_index=True)

# Generate ordered and unordered master files
unordered = full_data.drop(columns=['ID']).copy()
unordered.insert(0, 'SL.no.', range(1, len(unordered) + 1))
unordered.to_csv("combined.csv", index=False)

ordered = full_data.sort_values(by='ID').copy()
cols = ['ID'] + [c for c in ordered.columns if c != 'ID']
ordered = ordered[cols]
ordered.to_csv("ord_combine.csv", index=False)
print("success, Master datasets generated (combined.csv, ord_combine.csv).")

# =====================================================================
# PHASE 3: RAW DATA VISUALIZATION
# =====================================================================

print("\n--- PHASE 3: RAW DATA VISUALIZATION ---")
print("Visualizing raw Dwell vs Flight (UD) signatures. (Close the window to proceed).")
plt.figure(figsize=(10, 6))
user_ids = ordered['ID'].unique()
for uid in user_ids:
    user_data = ordered[ordered['ID'] == uid]
    plt.scatter(user_data['Flight_UD_s'], user_data['Dwell_Time_s'], alpha=0.5, label=f'User {uid}', s=40)

plt.title('Raw Keystroke Signature: Dwell Time vs. Flight Time (UD)')
plt.xlabel('Up-to-Down Flight Time (s)')
plt.ylabel('Dwell Time (s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# =====================================================================
# PHASE 4: DETECTOR (MACHINE LEARNING)
# =====================================================================
print("\n--- PHASE 4: ML TRAINING (Random Forest) ---")
df_ml = pd.read_csv("ord_combine.csv").dropna()

# UPGRADE: Remove extreme human error (pauses > 1s, overly long dwells)
print("Filtering out human errors and extreme outliers...")
df_ml = df_ml[(df_ml['Flight_UD_s'] < 1.0) & (df_ml['Dwell_Time_s'] < 0.5)]

# Apply Rolling Averages
print("Calculating macro-rhythm trends...")
df_ml['Rolling_Flight'] = df_ml.groupby('ID')['Flight_UD_s'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean())
df_ml['Rolling_Dwell'] = df_ml.groupby('ID')['Dwell_Time_s'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean())

# One-Hot Encode structural data
print("Encoding Keyboard Digraphs and structural features...")
df_encoded = pd.get_dummies(df_ml, columns=['Key', 'Key_Pair'])

X = df_encoded.drop(columns=['ID'])
y = df_encoded['ID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training authentication model...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

test_predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)
print(f" Baseline ML Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(rf_model, 'advanced_keystroke_model.pkl')

# Generate Full Predictions for Auditing
raw_predictions = rf_model.predict(X)
output_df = df_ml.copy()

# UPGRADE: Apply a Rolling Majority Vote (Smoothing)
# Looks at groups of 7 keystrokes to filter out isolated errors/typos
pred_series = pd.Series(raw_predictions)
smoothed_predictions = pred_series.rolling(window=7, min_periods=1).apply(
    lambda x: x.mode()[0] if not x.mode().empty else x.iloc[-1]
).astype(int)

output_df['Predicted_User'] = [f"User_{pred}" for pred in smoothed_predictions]

output_df = output_df.drop(columns=['ID', 'Rolling_Flight', 'Rolling_Dwell'], errors='ignore')
cols = ['Predicted_User'] + [c for c in output_df.columns if c != 'Predicted_User']
output_df = output_df[cols]
output_df.to_csv("predicted.csv", index=False)

print("Displaying ML Confusion Matrix. (Close the window to proceed).")
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'User {i}' for i in rf_model.classes_],
            yticklabels=[f'User {i}' for i in rf_model.classes_])
plt.title(f"Authentication Accuracy: {accuracy * 100:.2f}%", fontweight='bold')
plt.xlabel("Algorithm Guess")
plt.ylabel("Actual User")
plt.tight_layout()
plt.show()

#part 5: final audiit
print("\n--- PHASE 5: SYSTEM AUDIT ---")
df_true = pd.read_csv("ord_combine.csv").dropna()
# Ensure the truth dataframe gets the exact same filtering as the ML dataframe
df_true = df_true[(df_true['Flight_UD_s'] < 1.0) & (df_true['Dwell_Time_s'] < 0.5)]

df_pred = pd.read_csv("predicted.csv")

# UPGRADE: Direct Assignment instead of float-merging to preserve 100% of rows
merged = df_pred.copy()
merged['Actual_ID'] = df_true['ID'].values

label_map = {}
for actual_id in merged['Actual_ID'].unique():
    id_data = merged[merged['Actual_ID'] == actual_id]
    if not id_data.empty:
        most_frequent_guess = id_data['Predicted_User'].value_counts().idxmax()
        label_map[most_frequent_guess] = actual_id

merged['Mapped_Predicted_ID'] = merged['Predicted_User'].map(label_map)
merged['Is_Error'] = merged['Actual_ID'] != merged['Mapped_Predicted_ID']

errors = merged[merged['Is_Error'] == True]
correct = merged[merged['Is_Error'] == False]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.scatter(correct['Flight_UD_s'], correct['Dwell_Time_s'], c='blue', alpha=0.4, s=30, label='Authorized')
ax1.set_title('Successful Authentications', fontsize=14, color='blue')
ax1.set_xlabel('Flight Time (UD)')
ax1.set_ylabel('Dwell Time (s)')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

ax2.scatter(errors['Flight_UD_s'], errors['Dwell_Time_s'], c='red', alpha=1.0, s=100, marker='X', edgecolors='black',
            label='Intrusion / Mismatch')
ax2.set_title('Authentication Failures', fontsize=14, color='red')
ax2.set_xlabel('Flight Time (UD)')
ax2.set_ylabel('Dwell Time (s)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()

print("=" * 40)
print("FINAL PIPELINE REPORT")
print(f"Total Keystrokes Analyzed: {len(merged)}")
print(f"Successful Identifications: {len(correct)}")
print(f"Security Mismatches: {len(errors)}")
print(f"Final Smoothed Pipeline Accuracy: {(len(correct) / len(merged)) * 100 if len(merged) > 0 else 0:.2f}%")
print("=" * 40)

print("Displaying final audit graphs...")
plt.show()