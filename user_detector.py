import time
import pandas as pd
from pynput import keyboard
from sklearn.ensemble import RandomForestClassifier
import collections
import os
import math

# Define file names
ORD_FILE = "ord_combine.csv"
COMBINED_FILE = "combined.csv"

# =======================================================
# BIOMETRIC ALGORITHMS: Distance & Hand Size
# =======================================================
key_coords = {
    'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4), 'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8),
    'p': (0, 9),
    'a': (1, 0.5), 's': (1, 1.5), 'd': (1, 2.5), 'f': (1, 3.5), 'g': (1, 4.5), 'h': (1, 5.5), 'j': (1, 6.5),
    'k': (1, 7.5), 'l': (1, 8.5),
    'z': (2, 0.7), 'x': (2, 1.7), 'c': (2, 2.7), 'v': (2, 3.7), 'b': (2, 4.7), 'n': (2, 5.7), 'm': (2, 6.7)
}


def calculate_grid_distance(key_pair):
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


def calculate_estimated_hand_size(df):
    if df.empty:
        return 0.0
    avg_flight_dd = df['Flight_DD_s'].astype(float).mean()
    overlap_ratio = df['Is_Overlap'].astype(float).mean()
    estimated_cm = 22.4 - (avg_flight_dd * 4.5) + (overlap_ratio * 3.5)
    return round(max(14.0, min(26.0, estimated_cm)), 2)


# =======================================================
# PHASE 1: Train the Supervised AI (Advanced Metrics)
# =======================================================
try:
    print(f"🧠 Loading {ORD_FILE} and training AI...")
    df = pd.read_csv(ORD_FILE, on_bad_lines='skip')

    # STRICT TYPE CASTING
    numeric_cols = ["Flight_UD_s", "Flight_DD_s", "Dwell_Time_s", "Is_Overlap", "Session_WPM"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing WPM for older profiles
    if 'Session_WPM' in df.columns:
        df['Session_WPM'] = df['Session_WPM'].fillna(df['Session_WPM'].mean())

    df = df.dropna()

    # INJECT BIOMECHANICAL FEATURES BEFORE TRAINING
    df['Grid_Distance'] = df['Key_Pair'].apply(calculate_grid_distance)
    df['Instant_WPM'] = 60 / (df['Flight_DD_s'].clip(lower=0.01) * 5)

    y_train = df['ID']
    df_features = df.drop(columns=['ID'])

    X_train = pd.get_dummies(df_features)
    training_columns = X_train.columns

    rf_model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
    rf_model.fit(X_train, y_train)

    print(f"✅ AI Trained! Known Users in Database: {list(y_train.unique())}\n")

except FileNotFoundError:
    print(f"❌ Error: Could not find '{ORD_FILE}'. Make sure you ran the collector first.")
    exit()

# =======================================================
# PHASE 2: Interactive Typing Box (Advanced Tracking)
# =======================================================
active_keys = {}
last_release_time = 0.0
last_press_time = 0.0
last_key_pressed = ""
live_data = []

session_start_time = 0.0
typed_sentence = ""


def get_key_name(key):
    try:
        return key.char
    except AttributeError:
        return str(key).replace("Key.", "<") + ">"


def on_press(key):
    global last_release_time, last_press_time, last_key_pressed, session_start_time, typed_sentence
    if key == keyboard.Key.enter or key == keyboard.Key.esc:
        return False

    timestamp = time.time()
    key_name = get_key_name(key)

    if key_name == ',':
        return

    # Track actual sentence for accurate WPM calculation
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
    if key == keyboard.Key.enter or key == keyboard.Key.esc:
        return False

    release_time = time.time()
    key_name = get_key_name(key)

    if key_name == ',':
        return

    if key_name in active_keys:
        data = active_keys[key_name]
        dwell_time = release_time - data['press_time']

        live_data.append([
            key_name,
            data['key_pair'],
            data['flight_ud'],
            data['flight_dd'],
            dwell_time,
            data['is_overlap']
        ])
        del active_keys[key_name]

    last_release_time = release_time


print("=" * 50)
print("⌨️  INTERACTIVE KEYSTROKE PREDICTOR (ADVANCED)")
print("=" * 50)
print("Type a random sentence below. Press ENTER when finished.")

session_start_time = 0.0
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

user_input = input("\n📝 Textbox: ")
listener.join()

# =======================================================
# PHASE 3: Prediction & Feedback Loop
# =======================================================
if not live_data:
    print("\n❌ No typing detected! Run the script again.")
else:
    # ACCURATE WPM CALCULATION
    session_end_time = time.time()
    duration_min = max((session_end_time - session_start_time) / 60.0, 0.001)
    actual_chars = len(typed_sentence)
    wpm = round((actual_chars / 5.0) / duration_min, 2)

    print(f"\n📝 Sentence Captured: '{typed_sentence}'")
    print(f"📊 Session complete! Calculated WPM: {wpm} (Time: {round(duration_min * 60, 1)}s)")

    # Convert live data to DataFrame
    live_df = pd.DataFrame(live_data,
                           columns=["Key", "Key_Pair", "Flight_UD_s", "Flight_DD_s", "Dwell_Time_s", "Is_Overlap"])
    live_df["Session_WPM"] = wpm

    # STRICT TYPE CASTING
    for col in numeric_cols:
        if col in live_df.columns:
            live_df[col] = pd.to_numeric(live_df[col], errors='coerce')

    # INJECT BIOMECHANICAL FEATURES BEFORE PREDICTING
    live_df['Grid_Distance'] = live_df['Key_Pair'].apply(calculate_grid_distance)
    live_df['Instant_WPM'] = 60 / (live_df['Flight_DD_s'].clip(lower=0.01) * 5)

    # Make a clean copy of standard columns to save to CSVs later
    save_df = live_df[
        ["Key", "Key_Pair", "Flight_UD_s", "Flight_DD_s", "Dwell_Time_s", "Is_Overlap", "Session_WPM"]].copy()

    live_encoded = pd.get_dummies(live_df)
    X_live = live_encoded.reindex(columns=training_columns, fill_value=0)

    predictions = rf_model.predict(X_live)
    counter = collections.Counter(predictions)
    predicted_user, votes = counter.most_common(1)[0]
    confidence = (votes / len(predictions)) * 100

    print("\n" + "=" * 50)
    print(f"🤖 AI PREDICTION: You are USER {predicted_user} (Confidence: {confidence:.1f}%)")
    print("=" * 50)

    is_correct = input(f"Was this prediction correct? (y/n): ").strip().lower()

    if is_correct == 'y':
        print(f"✅ Great! Saving your new data to User {predicted_user}'s profile...")
        actual_id = predicted_user
    else:
        actual_id_input = input("❌ Whoops! What is your actual User ID? (Enter a number): ").strip()
        actual_id = int(actual_id_input) if actual_id_input.isdigit() else 999
        print(f"🔄 Understood. Learning from mistake. Saving data to User {actual_id}...")

    # =======================================================
    # PHASE 4: Update the Datasets (Continuous Learning)
    # =======================================================

    # We save 'save_df' so we don't pollute the CSVs with dynamically generated Grid math
    user_filename = f"keystroke_data_User_{actual_id}.csv"
    file_exists = os.path.exists(user_filename)
    save_df.to_csv(user_filename, mode='a', header=not file_exists, index=False)

    ord_update_df = save_df.copy()
    ord_update_df.insert(0, 'ID', actual_id)
    ord_update_df.to_csv(ORD_FILE, mode='a', header=False, index=False)

    if os.path.exists(COMBINED_FILE):
        combined_df = pd.read_csv(COMBINED_FILE)
        start_sl = combined_df['SL.no.'].max() + 1 if not combined_df.empty else 1
        comb_update_df = save_df.copy()
        comb_update_df.insert(0, 'SL.no.', range(start_sl, start_sl + len(comb_update_df)))
        comb_update_df.to_csv(COMBINED_FILE, mode='a', header=False, index=False)

    estimated_span = calculate_estimated_hand_size(save_df)
    hand_filename = f"HandSize_User_{actual_id}.csv"
    hand_file_exists = os.path.exists(hand_filename)

    hand_data = pd.DataFrame([{
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Session_Keys_Typed": len(save_df),
        "Session_WPM": wpm,
        "Estimated_Hand_Span_cm": estimated_span
    }])

    hand_data.to_csv(hand_filename, mode='a', header=not hand_file_exists, index=False)

    print(f"💾 Databases updated! Data appended to {user_filename}.")
    print(f"🖐️ Hand size calculation logged to {hand_filename} (Estimated: {estimated_span} cm).")