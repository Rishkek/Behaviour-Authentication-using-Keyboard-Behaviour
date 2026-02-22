# Create a 'Local WPM' feature for every 5-character window
df_clean['Local_WPM'] = 60 / ((df_clean['Flight_Time_s'] + df_clean['Dwell_Time_s']) * 5)

# Update your feature set for the GMM
X_refined = df_clean[['Flight_Time_s', 'Dwell_Time_s', 'Local_WPM']]
X_scaled = scaler.fit_transform(X_refined)
