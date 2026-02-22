from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import csv

app = FastAPI()

# =========================
# 🔥 CORS CONFIGURATION
# =========================
# For hackathon simplicity, allow all origins.
# You can restrict later.
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
# 🔹 TRAIN ENDPOINT
# =========================

@app.post("/train")
def train(request: TrainingRequest):

    print("\n==============================")
    print("Received training for user:", request.user)
    print("Number of sessions:", len(request.sessions))

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    file_path = f"data/{request.user}.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header if file does not exist
        if not file_exists:
            writer.writerow([
                "Mean_Dwell",
                "Mean_Flight_UD",
                "Mean_Flight_DD",
                "Overlap_Rate",
                "WPM",
                "Accuracy",
                "Burst_Pause_Count"
            ])

        # Write one row per session
        for session in request.sessions:
            writer.writerow([
                session.Mean_Dwell,
                session.Mean_Flight_UD,
                session.Mean_Flight_DD,
                session.Overlap_Rate,
                session.WPM,
                session.Accuracy,
                session.Burst_Pause_Count
            ])

    print(f"CSV successfully saved at: {file_path}")

    return {
        "status": "csv saved successfully",
        "user": request.user,
        "sessions_written": len(request.sessions),
        "file_path": file_path
    }