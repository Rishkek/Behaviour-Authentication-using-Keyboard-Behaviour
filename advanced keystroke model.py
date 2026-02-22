import time
import pandas as pd
import joblib
from pynput import keyboard
from collections import deque

# Load the trained model and features
print("Loading Keystroke Model...")
rf_model = joblib.load('advanced_keystroke_model.pkl')

# Configuration
SMOOTHING_WINDOW = 7
metrics_buffer = deque(maxlen=SMOOTHING_WINDOW)
active_keys = {}
last_press_time = 0.0
last_release_time = 0.0
last_key_pressed = ""

def get_key_name(key):
    try: return key.char
    except AttributeError: return str(key).replace("Key.", "<") + ">"

def on_press(key):
    global last_press_time, last_release_time, last_key_pressed
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
    release_time = time.time()
    key_name = get_key_name(key)

    if key_name in active_keys:
        data = active_keys[key_name]
        dwell_time = release_time - data['press_time']
        
        # Prepare feature vector
        process_and_predict(key_name, data, dwell_time)
        del active_keys[key_name]

    last_release_time = release_time
    if key == keyboard.Key.esc:
        return False

def process_and_predict(key_name, data, dwell):
    # This must match your training features exactly
    # Note: In a production environment, you would need to handle 
    # the One-Hot Encoding alignment using the training columns.
    
    # Placeholder for the prediction logic
    # For a full implementation, you would:
    # 1. Create a DataFrame with the new keystroke
    # 2. Reindex it to match the training X.columns
    # 3. Predict and update the rolling majority
    pass

print("\n--- REAL-TIME AUTH ACTIVE ---")
print("Type naturally. Press [Esc] to stop.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
