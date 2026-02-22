# Behaviour-Authentication-using-Keyboard-Behaviour
🚀 MorphoType: Continuous Behavioral Biometric Authentication
MorphoType is a sophisticated keystroke dynamics engine that moves beyond "what you know" (passwords) to "how you act" (behavior). By analyzing the micro-rhythms of a user's typing patterns and the physics of their hand movement, we provide a seamless, invisible layer of security.

🔬 The Core Innovation: "The Physical Reach Proxy"
Unlike traditional keystroke trackers, MorphoType includes a Biometric Decay Analysis.

We utilize a custom Euclidean Grid Mapping of the QWERTY layout to calculate the relationship between WPM and Key Distance.

The Insight: Users with larger hand spans maintain high velocity across distal keys (e.g., 'Q' to 'P'), while smaller hands show a steeper "velocity decay." This allows our model to verify identity based on the physical constraints of the human hand.

🛠️ Technical Pipeline
1. Multi-Dimensional Feature Extraction

We capture three critical temporal signatures for every keystroke:

Dwell Time (T 
d
​	
 ): Muscle latency and key-press habit.

Flight Time (T 
f
​	
 ): The travel interval between key release and next press.

Key Rollover (Overlap): Detecting fluid "rolling" typing vs. "staccato" typing.

2. Autonomous Data Hygiene

Real-world data is noisy. We implement an Isolation Forest (Anomaly Detection) per user profile to automatically strip away:

Accidental double-taps.

Sudden pauses (distractions).

"Fat-finger" errors.
This ensures the Random Forest Classifier trains only on the user's "Golden Rhythm."

3. Identity Smoothing

To prevent "False Rejections," the engine utilizes a 7-keystroke rolling window. Authentication isn't decided by a single key but by a consensus of the last 7 actions, providing a stable, jitter-free security score.

📊 Performance & Visualization
The pipeline generates three high-impact audit reports:

Rhythm Signature: A scatter plot mapping Dwell vs. Flight profiles.

Biometric Decay: A regression plot showing the WPM-to-Distance correlation.

Security Audit: A confusion matrix and intrusion detection map identifying "Authorized" vs. "Mismatch" attempts.

🚀 Installation & Usage
📦 Dependencies

Bash
pip install pynput pandas seaborn scikit-learn joblib matplotlib
🏃 How to Run

Data Collection: Run the script and enter a User ID. Type for 20 seconds to build your profile.

ML Training: Enter q to trigger the automated cleaning and training pipeline.

Audit: Review the generated predicted.csv and the Audit Graphs to verify the system's security accuracy.

⚠️ Challenges & Limitations
Hardware Bias: Mechanical vs. Membrane keyboards affect Dwell Time.

Cognitive State: Significant fatigue or caffeine intake can shift rhythmic baselines.

Future Scope: Implementing a Hardware Normalization Layer to allow profiles to roam between different devices seamlessly.

🏆 Hackathon Impact
Continuous Auth: Protects against "Session Hijacking" after the initial login.

Proctoring: Verifies student identity throughout the duration of digital exams.

Zero-Touch UX: Security that works in the background without interrupting the workflow.
