# New logic: Calculate 'Functional Reach Index' (FRI)
# 1.0 is the 'ideal' fast typist. Lower numbers = larger physical reach.
user_fri = []
for center in centroids:
    # Use the ratio of flight to dwell as a proxy for 'effort'
    # Smaller hands usually have a higher Flight/Dwell ratio because 
    # they spent more time 'traveling' than 'pressing'.
    fri = (center[0] / center[1]) * (100 / gross_wpm)
    user_fri.append(round(fri, 3))

# Update labels for the graph
df_clean['Predicted_User'] = [f"Profile {c+1} (FRI: {user_fri[c]})" for c in clusters]
