import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

csv_filename = "combined_keystroke_data.csv"

try:
    # 1. Load the dataset
    df = pd.read_csv(csv_filename).dropna()

    # 2. Extract features
    X = df[['Flight_Time_s', 'Dwell_Time_s']]

    # 3. Filter out "Noise" using an Isolation Forest
    # This acts as a bouncer, kicking out weird keystrokes (like 5-second pauses)
    print("Detecting and removing erratic keystrokes...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Assume 5% of data is noise
    good_data_mask = iso_forest.fit_predict(X) == 1

    # Keep only the clean data
    df_clean = df[good_data_mask].copy()
    X_clean = df_clean[['Flight_Time_s', 'Dwell_Time_s']]
    print(f"Removed {len(df) - len(df_clean)} outlier keystrokes.")

    # 4. Standardize the clean data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # 5. Apply Gaussian Mixture Model (GMM)
    # covariance_type='full' allows each user's cluster to be its own unique elliptical shape
    print("Calculating probabilistic user distributions (GMM)...")
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, n_init=5)

    # Predict which user each keystroke belongs to
    df_clean['Detected_User'] = gmm.fit_predict(X_scaled)

    # 6. Visualize the Advanced Separation
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'red']

    # Plot the clean, separated data
    for cluster_num in range(3):
        cluster_data = df_clean[df_clean['Detected_User'] == cluster_num]
        plt.scatter(cluster_data['Flight_Time_s'], cluster_data['Dwell_Time_s'],
                    c=colors[cluster_num], label=f'User Profile {cluster_num + 1}',
                    alpha=0.6, edgecolors='w', s=60)

    # Plot the mathematical centers (Means of the Gaussian distributions)
    centroids = scaler.inverse_transform(gmm.means_)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', marker='*', s=500,
                edgecolors='black', label='User Rhythm Centers')

    # Styling the plot
    plt.title('Advanced Biometric Separation via Gaussian Mixture Models')
    plt.xlabel('Flight Time (Seconds)')
    plt.ylabel('Dwell Time (Seconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: Please ensure '{csv_filename}' is in the same folder as this script.")
except KeyError as e:
    print(f"Error: Missing expected column in the CSV - {e}")