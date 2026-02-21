import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    # 1. Load both files
    df_true = pd.read_csv("ord_combine.csv")
    df_pred = pd.read_csv("predicted.csv")

    # 2. Merge to align rows precisely
    merged = pd.merge(df_true, df_pred, on=['Key', 'Flight_Time_s', 'Dwell_Time_s'], how='inner')

    # 3. Label Alignment Logic
    label_map = {}
    for actual_id in merged['ID'].unique():
        id_data = merged[merged['ID'] == actual_id]
        if not id_data.empty:
            most_frequent_guess = id_data['Predicted_User'].value_counts().idxmax()
            label_map[most_frequent_guess] = actual_id

    merged['Mapped_Predicted_ID'] = merged['Predicted_User'].map(label_map)

    # 4. Filter into Correct and Incorrect
    merged['Is_Error'] = merged['ID'] != merged['Mapped_Predicted_ID']
    errors = merged[merged['Is_Error'] == True]
    correct = merged[merged['Is_Error'] == False]

    # --- 5. Visualization: Two Separate Graphs ---
    # Create a figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # GRAPH 1: SUCCESSES
    ax1.scatter(correct['Flight_Time_s'], correct['Dwell_Time_s'],
                c='blue', alpha=0.5, s=40, label='Matches')
    ax1.set_title('Graph 1: Correct Profile Matches', fontsize=14, color='blue')
    ax1.set_xlabel('Flight Time (s)')
    ax1.set_ylabel('Dwell Time (s)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # GRAPH 2: ERRORS

    ax2.scatter(errors['Flight_Time_s'], errors['Dwell_Time_s'],
                c='red', alpha=1.0, s=120, marker='X', edgecolors='black', label='Mismatches')
    ax2.set_title('Graph 2: Profile Mismatches (Errors)', fontsize=14, color='red')
    ax2.set_xlabel('Flight Time (s)')
    ax2.set_ylabel('Dwell Time (s)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    # Print Accuracy Stats
    accuracy = (len(correct) / len(merged)) * 100 if len(merged) > 0 else 0
    print("-" * 30)
    print(f"Comparison Result:")
    print(f"Correct Matches: {len(correct)}")
    print(f"Errors Found:    {len(errors)}")
    print(f"System Accuracy: {accuracy:.2f}%")
    print("-" * 30)

    plt.show()

except FileNotFoundError as e:
    print(f"Error: Missing file - {e}")
except Exception as e:
    print(f"An error occurred: {e}")