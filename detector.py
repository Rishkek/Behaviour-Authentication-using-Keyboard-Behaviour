import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

csv_filename = "combined.csv"

try:
    # 1. Load the dataset
    df = pd.read_csv(csv_filename).dropna()

    # 2. Extract features
    X = df[['Flight_Time_s', 'Dwell_Time_s']]

    # 3. Filter out "Noise" - Low contamination to keep 1.2s points
    iso_forest = IsolationForest(contamination=0.05, random_state=52)
    good_data_mask = iso_forest.fit_predict(X) == 1
    df_clean = df[good_data_mask].copy()
    X_clean = df_clean[['Flight_Time_s', 'Dwell_Time_s']]

    # 4. Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # --- 5. DYNAMIC USER DETECTION (BIC) ---
    print("Finding the optimal number of users...")
    bic_scores = []
    n_components_range = range(1, 11)  # Test for 1 to 10 users

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=42).fit(X_scaled)
        bic_scores.append(gmm.bic(X_scaled))

    # The best n is the one with the lowest BIC score
    optimal_n = n_components_range[np.argmin(bic_scores)]
    print(f"✅ Algorithm detected {optimal_n} unique user profiles.")

    # 6. Apply GMM with the detected number of users
    gmm = GaussianMixture(n_components=optimal_n, covariance_type='full', random_state=42, n_init=5)
    clusters = gmm.fit_predict(X_scaled)
    df_clean['Predicted_User'] = [f"User_{c + 1}" for c in clusters]

    # --- 7. Save to predicted.csv ---
    if 'SL.no.' in df_clean.columns:
        df_clean = df_clean.drop(columns=['SL.no.'])
    cols = ['Predicted_User'] + [c for c in df_clean.columns if c != 'Predicted_User']
    df_clean[cols].to_csv("predicted.csv", index=False)

    # 8. Visualize with dynamic colors
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('tab10')  # Supports up to 10 distinct colors

    for i in range(optimal_n):
        label_name = f"User_{i + 1}"
        cluster_data = df_clean[df_clean['Predicted_User'] == label_name]
        plt.scatter(cluster_data['Flight_Time_s'], cluster_data['Dwell_Time_s'],
                    color=cmap(i), label=f'Profile {label_name}',
                    alpha=0.6, edgecolors='w', s=60)

    # Plot Rhythm Centers
    centroids = scaler.inverse_transform(gmm.means_)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', marker='*', s=500,
                edgecolors='black', label='User Rhythm Centers')

    plt.title(f'Detected {optimal_n} Unique User Profiles via BIC & GMM')
    plt.xlabel('Flight Time (Seconds)')
    plt.ylabel('Dwell Time (Seconds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")