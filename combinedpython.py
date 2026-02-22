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
import math
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# =====================================================================
# GLOBALS & BIOMETRIC ALGORITHMS
# =====================================================================
print("\n" + "=" * 50)
print("🚀 INITIALIZING KEYSTROKE DYNAMICS ENGINE")
print("Passwordless Client-Side Authentication Pipeline")
print("=" * 50)

active_keys = {}
last_release_time = 0.0
last_press_time = 0.0
last_key_pressed = ""
current_csv = ""
listener = None
esc_pressed = False

session_data = []
session_start_time = 0.0
typed_sentence = ""

# Define the Keyboard Grid for Physical Distance Tracking
key_coords = {
    'q':(0,0), 'w':(0,1), 'e':(0,2), 'r':(0,3), 't':(0,4), 'y':(0,5), 'u':(0,6), 'i':(0,7), 'o':(0,8), 'p':(0,9),
    'a':(1,0.5), 's':(1,1.5), 'd':(1,2.5), 'f':(1,3.5), 'g':(1,4.5), 'h':(1,5.5), 'j':(1,6.5), 'k':(1,7.5), 'l':(1,8.5),
    'z':(2,0.7), 'x':(2,1.7), 'c':(2,2.7), 'v':(2,3.7), 'b':(2,4.7), 'n':(2,5.7), 'm':(2,6.7)
}

def calculate_grid_distance(key_pair):
    """Calculates physical distance between two keys using Euclidean math"""
    if pd.isna(key_pair) or key_pair == "START" or "_" not in str(key_pair):
        return 0.0
    try:
        k1, k2 = key_pair.split('_')
        k1, k2 = k1.lower(), k2.lower()
        if k1 in key_coords and k2 in key_coords:
            x1, y1 = key_coords[k1]
            x2, y2 = key_coords[k2]
            return round(math.dist((x1, y1), (x2, y2)), 2)
    except Exception:
        pass
    return 0.0

def get_key_name(key):
    try:
        return key.char
    except AttributeError:
        return str(key).replace("Key.", "<") + ">"

def on_press(key):
    global last_press_time, last_release_time, last_key_pressed, esc_pressed, session_start_time, typed_sentence
    if key == keyboard.Key.esc:
        esc_pressed = True
        return False

    timestamp = time.time()
    key_name = get_key_name(key)

    if key_name == ',':
        return

    if hasattr(key, 'char') and key.char is not None and key.char != ',':
        typed_sentence += key.char
    elif key == keyboard.Key.space:
        typed_sentence += " "
    elif key == keyboard.Key.backspace:
        typed_sentence = typed_sentence[:-1]

    if session_start_time == 0.0:
        session_start_time = timestamp

    if key_name not in active_keys:
        flight_dd = (timestamp - last_press_time) if last_press_time else 0.0
        flight_ud = (timestamp - last_release_time) if last_release_time else 0.0
        is_overlap = 1 if (last_release_time == 0.0 or last_press_time > last_release_time) else 0
        key_pair = f"{last_key_pressed}_{key_name}" if last_key_pressed else "START"

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

    if key_name == ',':
        return

    if key_name in active_keys:
        data = active_keys[key_name]
        dwell_time = release_time - data['press_time']

        session_data.append([
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

# =====================================================================
# PHASE 1: DATA GATHERING
# =====================================================================
print("\n--- PHASE 1: DATA GATHERING ---")
existing_files = glob.glob("keystroke_data_User_*.csv")
existing_ids = []
for f in existing_files:
    match = re.search(r'User_(\d+)\.csv', f)
    if match: existing_ids.append(int(match.group(1)))

if existing_ids:
    print(f"📁 Found existing profiles for Users: {sorted(existing_ids)}")

while not esc_pressed:
    user_input = input("\n>> Enter your user id (or 'q' to quit to ML Training): ").strip()

    if user_input.lower() == 'q':
        break

    if not user_input.isdigit():
        print("❌ Please enter a valid numeric ID.")
        continue

    u_id = int(user_input)
    current_csv = f"keystroke_data_User_{u_id}.csv"

    active_keys.clear()
    session_data.clear()
    last_release_time = 0.0
    last_press_time = 0.0
    last_key_pressed = ""
    session_start_time = 0.0
    typed_sentence = ""

    file_exists = os.path.exists(current_csv)
    with open(current_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["Key", "Key_Pair", "Flight_UD_s", "Flight_DD_s", "Dwell_Time_s", "Is_Overlap", "Session_WPM"])

    input(f">> Hello User {u_id}, press [ENTER] to begin typing (20s timer)...")
    print("User: ", end="", flush=True)

    timer = threading.Timer(20.0, stop_listener)
    timer.start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
        listener = l
        listener.join()

    if session_data:
        session_end_time = time.time()
        duration_min = max((session_end_time - session_start_time) / 60.0, 0.001)
        actual_chars = len(typed_sentence)
        wpm = round((actual_chars / 5.0) / duration_min, 2)

        print(f"\n📝 Sentence Captured: '{typed_sentence}'")
        print(f"📊 Session complete! Calculated WPM: {wpm} (Time: {round(duration_min * 60, 1)}s)")

        with open(current_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in session_data:
                writer.writerow(row + [wpm])

        print(f">> Saved to {current_csv}")

    if esc_pressed:
        break

# =====================================================================
# PHASE 2: COMBINATION
# =====================================================================
print("\n--- PHASE 2: DATA COMBINATION ---")
all_dfs = []
all_existing_files = glob.glob("keystroke_data_User_*.csv")

for file_name in all_existing_files:
    match = re.search(r'User_(\d+)\.csv', file_name)
    if match:
        try:
            df = pd.read_csv(file_name, on_bad_lines='skip')
            if not df.empty:
                df['ID'] = int(match.group(1))
                all_dfs.append(df)
        except FileNotFoundError:
            continue

if not all_dfs:
    print("❌ No data found. Exiting pipeline.")
    exit()

full_data = pd.concat(all_dfs, ignore_index=True)

unordered = full_data.drop(columns=['ID']).copy()
unordered.insert(0, 'SL.no.', range(1, len(unordered) + 1))
unordered.to_csv("combined.csv", index=False)

ordered = full_data.sort_values(by='ID').copy()
cols = ['ID'] + [c for c in ordered.columns if c != 'ID']
ordered = ordered[cols]
ordered.to_csv("ord_combine.csv", index=False)
print("✅ Master datasets generated (combined.csv, ord_combine.csv).")

# =====================================================================
# PHASE 3: RAW DATA VISUALIZATION
# =====================================================================
print("\n--- PHASE 3: RAW DATA VISUALIZATION ---")
print("Visualizing raw Dwell vs Flight (UD) signatures. (Close the window to proceed).")
plt.figure(figsize=(10, 6))
user_ids = ordered['ID'].unique()
for uid in user_ids:
    user_data = ordered[ordered['ID'] == uid]
    plt.scatter(user_data['Flight_UD_s'], user_data['Dwell_Time_s'], alpha=0.5, label=f'User {uid}', s=14)

plt.title('Raw Keystroke Signature: Dwell Time vs. Flight Time (UD)')
plt.xlabel('Up-to-Down Flight Time (s)')
plt.ylabel('Dwell Time (s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# =====================================================================
# PHASE 3.5: BIOMETRIC DECAY ANALYSIS
# =====================================================================
print("\n--- PHASE 3.5: BIOMETRIC DECAY ANALYSIS ---")
df_ml = pd.read_csv("ord_combine.csv").dropna()

# Extract Physical Keyboard Distance & Instantaneous Speed
df_ml['Flight_DD_s'] = pd.to_numeric(df_ml['Flight_DD_s'], errors='coerce')
df_ml['Grid_Distance'] = df_ml['Key_Pair'].apply(calculate_grid_distance)
df_ml['Instant_WPM'] = 60 / (df_ml['Flight_DD_s'].clip(lower=0.01) * 5)

decay_plot_df = df_ml[df_ml['Grid_Distance'] > 0]


plt.figure(figsize=(10, 6))
sns.regplot(data=decay_plot_df, x='Grid_Distance', y='Instant_WPM',
            scatter_kws={'alpha':0.3, 'color':'teal'}, line_kws={'color':'red', 'linewidth':2})

plt.title("Biometric Decay: WPM vs. Keyboard Reach Distance", fontweight='bold')
plt.xlabel("Physical Keyboard Reach Distance (Grid Units)")
plt.ylabel("Effective Instant WPM")
plt.grid(True, linestyle='--', alpha=0.6)

plt.text(decay_plot_df['Grid_Distance'].max()*0.4, decay_plot_df['Instant_WPM'].max()*0.85,
         "Steep Slope = Small Hand Span\nFlat Slope = Large Hand Span",
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

print("Displaying Biometric Decay Graph. (Close the window to proceed).")
plt.show()

# =====================================================================
# PHASE 4: DETECTOR (MACHINE LEARNING)
# =====================================================================
print("\n--- PHASE 4: ML TRAINING (Random Forest) ---")
print("Filtering out human errors using Isolation Forest (Per User)...")

clean_dfs = []

for uid in df_ml['ID'].unique():
    user_data = df_ml[df_ml['ID'] == uid].copy()

    if len(user_data) < 10:
        clean_dfs.append(user_data)
        continue

    # Include WPM and physical distance into anomaly detection
    features = user_data[['Flight_UD_s', 'Dwell_Time_s', 'Session_WPM', 'Grid_Distance']]

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    inliers = iso_forest.fit_predict(features)

    clean_user_data = user_data[inliers == 1]
    clean_dfs.append(clean_user_data)

df_ml = pd.concat(clean_dfs).sort_index()
valid_indices = df_ml.index

print("Calculating macro-rhythm trends...")
df_ml['Rolling_Flight'] = df_ml.groupby('ID')['Flight_UD_s'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean())
df_ml['Rolling_Dwell'] = df_ml.groupby('ID')['Dwell_Time_s'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean())

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
print(f"✅ Baseline ML Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(rf_model, 'advanced_keystroke_model.pkl')
# CRITICAL FOR LIVE PREDICTOR: Save the exact training columns
joblib.dump(X_train.columns, 'model_columns.pkl')

raw_predictions = rf_model.predict(X)
output_df = df_ml.copy()

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

# =====================================================================
# PHASE 5: COMPARATOR AUDIT
# =====================================================================
print("\n--- PHASE 5: SYSTEM AUDIT ---")
df_true = pd.read_csv("ord_combine.csv").dropna()
df_true = df_true.loc[valid_indices]

df_pred = pd.read_csv("predicted.csv")

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