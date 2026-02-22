from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import csv
import glob
import re

app = FastAPI()

# =========================
# 🔥 ABSOLUTE PATH SETUP
# =========================
# This guarantees files are saved exactly next to this Python script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 🔥 CORS CONFIGURATION
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 🔹 DATA MODELS
# =========================

class KeystrokeEntry(BaseModel):
    Key: str
    Key_Pair: str
    Flight_UD_s: float
    Flight_DD_s: float
    Is_Overlap: int
    Dwell_Time_s: Optional[float] = None


class SentenceSession(BaseModel):
    sentence: str
    keystrokes: List[KeystrokeEntry]
    WPM: float
    Accuracy: float
    Mean_Dwell: float
    Mean_Flight_UD: float
    Mean_Flight_DD: float
    Overlap_Rate: float
    Burst_Pause_Count: int


class TrainingRequest(BaseModel):
    user: str
    sessions: List[SentenceSession]


# =========================
# 🔹 HELPER: USER MAPPING & CONFLICT PREVENTION
# =========================

def get_or_create_user_id(username: str) -> int:
    """Checks the mapping CSV and filesystem to guarantee a conflict-free numeric ID."""
    mapping_file = os.path.join(BASE_DIR, "user_mapping.csv")

    # Create the file with headers if it doesn't exist
    if not os.path.exists(mapping_file):
        with open(mapping_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["UserID", "Username"])

    # Read the mapping to find the user
    try:
        df = pd.read_csv(mapping_file)
        if username in df['Username'].values:
            # User exists, return their mapped numeric ID safely
            return int(df.loc[df['Username'] == username, 'UserID'].iloc[0])
        else:
            # USER IS NEW: Calculate the absolute highest ID across all files to prevent conflicts

            # 1. Check highest ID in the mapping file
            max_map_id = int(df['UserID'].max()) if not df.empty else 0

            # 2. Check highest ID in existing personal CSV files on the disk
            max_file_id = 0
            search_pattern = os.path.join(BASE_DIR, "keystroke_data_User_*.csv")
            existing_files = glob.glob(search_pattern)
            for f in existing_files:
                match = re.search(r'User_(\d+)\.csv', f)
                if match:
                    max_file_id = max(max_file_id, int(match.group(1)))

            # 3. Check highest ID inside the ord_combine.csv database
            max_ord_id = 0
            ord_file_path = os.path.join(BASE_DIR, "ord_combine.csv")
            if os.path.exists(ord_file_path):
                try:
                    df_ord = pd.read_csv(ord_file_path, usecols=['ID'], on_bad_lines='skip')
                    if not df_ord.empty:
                        max_ord_id = int(df_ord['ID'].max())
                except Exception:
                    pass

            # Guarantee the new ID does not conflict with ANY existing data
            new_id = max(max_map_id, max_file_id, max_ord_id) + 1

            with open(mapping_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([new_id, username])
            return new_id

    except Exception as e:
        print(f"⚠️ Error reading {mapping_file}: {e}. Resetting mapping file safely.")

        # Failsafe: Re-calculate the highest file ID even if the mapping file breaks
        max_file_id = 0
        search_pattern = os.path.join(BASE_DIR, "keystroke_data_User_*.csv")
        for f in glob.glob(search_pattern):
            match = re.search(r'User_(\d+)\.csv', f)
            if match: max_file_id = max(max_file_id, int(match.group(1)))

        new_id = max_file_id + 1

        with open(mapping_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["UserID", "Username"])
            writer.writerow([new_id, username])
        return new_id


# =========================
# 🔹 TRAIN ENDPOINT
# =========================

@app.post("/train")
def train(request: TrainingRequest):
    print("\n" + "=" * 40)
    print(f"📥 Received training for username: {request.user}")

    # 1) Resolve the Username into a Conflict-Free Numeric ID
    user_id = get_or_create_user_id(request.user)
    print(f"🆔 Mapped to Numeric User ID: {user_id}")
    print(f"📦 Number of sessions: {len(request.sessions)}")

    # =========================================================
    # TASK 1: SAVE THE HIGH-LEVEL SESSION SUMMARY (TEXT INPUTS)
    # =========================================================
    session_file_path = os.path.join(BASE_DIR, f"session_summary_User_{user_id}.csv")
    session_exists = os.path.isfile(session_file_path)

    with open(session_file_path, mode="a", newline="", encoding="utf-8") as f_sess:
        writer_sess = csv.writer(f_sess)

        # Write headers if file is new
        if not session_exists:
            writer_sess.writerow([
                "Sentence", "Mean_Dwell", "Mean_Flight_UD", "Mean_Flight_DD",
                "Overlap_Rate", "WPM", "Accuracy", "Burst_Pause_Count"
            ])

        # Write the high level metrics
        for session in request.sessions:
            writer_sess.writerow([
                session.sentence,
                session.Mean_Dwell,
                session.Mean_Flight_UD,
                session.Mean_Flight_DD,
                session.Overlap_Rate,
                session.WPM,
                session.Accuracy,
                session.Burst_Pause_Count
            ])

    print(f"✅ Session summary saved to: {session_file_path}")

    # =========================================================
    # TASK 2: SAVE THE RAW KEYSTROKES TO THE ML PIPELINE
    # =========================================================
    indiv_file_path = os.path.join(BASE_DIR, f"keystroke_data_User_{user_id}.csv")
    ord_file_path = os.path.join(BASE_DIR, "ord_combine.csv")
    comb_file_path = os.path.join(BASE_DIR, "combined.csv")

    indiv_exists = os.path.isfile(indiv_file_path)
    ord_exists = os.path.isfile(ord_file_path)
    comb_exists = os.path.isfile(comb_file_path)

    # Calculate the starting SL.no. for combined.csv
    start_sl = 1
    if comb_exists:
        try:
            df_comb = pd.read_csv(comb_file_path, usecols=['SL.no.'])
            if not df_comb.empty:
                start_sl = int(df_comb['SL.no.'].max()) + 1
        except Exception:
            pass  # Failsafe defaults to 1

    total_keystrokes_written = 0

    # Open all three files simultaneously to append data efficiently
    with open(indiv_file_path, mode="a", newline="", encoding="utf-8") as f_indiv, \
            open(ord_file_path, mode="a", newline="", encoding="utf-8") as f_ord, \
            open(comb_file_path, mode="a", newline="", encoding="utf-8") as f_comb:

        writer_indiv = csv.writer(f_indiv)
        writer_ord = csv.writer(f_ord)
        writer_comb = csv.writer(f_comb)

        base_columns = ["Key", "Key_Pair", "Flight_UD_s", "Flight_DD_s", "Dwell_Time_s", "Is_Overlap", "Session_WPM"]

        # Write headers if the files were just created
        if not indiv_exists:
            writer_indiv.writerow(base_columns)
        if not ord_exists:
            writer_ord.writerow(["ID"] + base_columns)
        if not comb_exists:
            writer_comb.writerow(["SL.no."] + base_columns)

        # Iterate through every session and keystroke
        for session in request.sessions:
            session_wpm = round(session.WPM, 2)

            for ks in session.keystrokes:
                dwell = ks.Dwell_Time_s if ks.Dwell_Time_s is not None else 0.0

                # The core data block
                row_data = [
                    ks.Key,
                    ks.Key_Pair,
                    f"{ks.Flight_UD_s:.4f}",
                    f"{ks.Flight_DD_s:.4f}",
                    f"{dwell:.4f}",
                    ks.Is_Overlap,
                    session_wpm
                ]

                # Append to individual user CSV
                writer_indiv.writerow(row_data)

                # Append to ordered Master CSV (Prefix with User ID)
                writer_ord.writerow([user_id] + row_data)

                # Append to anonymous Master CSV (Prefix with SL.no.)
                writer_comb.writerow([start_sl] + row_data)

                start_sl += 1
                total_keystrokes_written += 1

    print(f"✅ Granular keystroke data written to: {indiv_file_path}")
    print(f"✅ Master databases ({ord_file_path}, {comb_file_path}) updated.")
    print(f"📊 Total Keystrokes Processed: {total_keystrokes_written}")

    return {
        "status": "csv saved and synced successfully",
        "username": request.user,
        "numeric_user_id": user_id,
        "sessions_processed": len(request.sessions),
        "keystrokes_written": total_keystrokes_written,
        "individual_file": indiv_file_path,
        "session_summary_file": session_file_path
    }