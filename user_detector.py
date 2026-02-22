import time
import pandas as pd
from pynput import keyboard
from sklearn.ensemble import RandomForestClassifier
import collections
import os

# Define file names
ORD_FILE = "ord_combine.csv"
COMBINED_FILE = "combined.csv"


#hand size algorithm
def calculate_estimated_hand_size(df):
    """
    Calculates an estimated hand span (in cm) based on typing dynamics.
    Faster reach across the keyboard (lower Flight_DD) and higher key overlap
    typically correlate with longer fingers and larger hand spans.
    """
    if df.empty:
        return 0.0

    # Calculate macro trends for the session
    avg_flight_dd = df['Flight_DD_s'].mean()
    overlap_ratio = df['Is_Overlap'].mean()

    # Base baseline hand span is ~19.0 cm.
    # Algorithm adjusts based on latency and simultaneous key presses.
    estimated_cm = 20.0 - (avg_flight_dd * 4.5) + (overlap_ratio * 3.5)

    # Constrain to realistic human limits (14cm to 26cm)
    return round(max(14.0, min(26.0, estimated_cm)), 2)


#training the supervised AI model.
try:
    print(f"🧠 Loading {ORD_FILE} and training AI...")
    df = pd.read_csv(ORD_FILE).dropna()

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

 
# PHASE 2: Interactive Typing Box (Advanced Tracking)

active_keys = {}
last_release_time = 0.0
last_press_time = 0.0
last_key_pressed = ""
live_data = []


def get_key_name(key):
    try:
        return key.char
    except AttributeError:
        return str(key).replace("Key.", "<") + ">"


def on_press(key):
    global last_release_time, last_press_time, last_key_pressed
    if key == keyboard.Key.enter or key == keyboard.Key.esc:
        return False

    timestamp = time.time()
    key_name = get_key_name(key)

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
    live_df = pd.DataFrame(live_data,
                           columns=["Key", "Key_Pair", "Flight_UD_s", "Flight_DD_s", "Dwell_Time_s", "Is_Overlap"])

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

    # A) Update individual user keystroke file
    user_filename = f"keystroke_data_User_{actual_id}.csv"
    file_exists = os.path.exists(user_filename)
    live_df.to_csv(user_filename, mode='a', header=not file_exists, index=False)

    # B) Update ord_combine.csv
    ord_update_df = live_df.copy()
    ord_update_df.insert(0, 'ID', actual_id)
    ord_update_df.to_csv(ORD_FILE, mode='a', header=False, index=False)

    # C) Update combined.csv
    if os.path.exists(COMBINED_FILE):
        combined_df = pd.read_csv(COMBINED_FILE)
        start_sl = combined_df['SL.no.'].max() + 1 if not combined_df.empty else 1
        comb_update_df = live_df.copy()
        comb_update_df.insert(0, 'SL.no.', range(start_sl, start_sl + len(comb_update_df)))
        comb_update_df.to_csv(COMBINED_FILE, mode='a', header=False, index=False)

    # D) Calculate and log the Hand Size

    estimated_span = calculate_estimated_hand_size(live_df)
    hand_filename = f"HandSize_User_{actual_id}.csv"
    hand_file_exists = os.path.exists(hand_filename)

    # Store timestamp, keystroke count for the session, and the calculated span
    hand_data = pd.DataFrame([{
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Session_Keys_Typed": len(live_df),
        "Estimated_Hand_Span_cm": estimated_span
    }])

    hand_data.to_csv(hand_filename, mode='a', header=not hand_file_exists, index=False)

    print(f"💾 Databases updated! Data appended to {user_filename}.")
    print(f"🖐️ Hand size calculation logged to {hand_filename} (Estimated: {estimated_span} cm).")