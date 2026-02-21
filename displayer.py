import pandas as pd
import matplotlib.pyplot as plt


def plot_dwell_vs_flight(filename):
    try:
        # 1. Load the data
        df = pd.read_csv(filename)

        if df.empty:
            print("No data found in the file.")
            return

        plt.figure(figsize=(12, 7))

        # 2. Get unique users
        user_ids = df['ID'].unique()

        # 3. Create a scatter plot for each user
        for uid in user_ids:
            user_data = df[df['ID'] == uid]

            plt.scatter(
                user_data['Flight_Time_s'],
                user_data['Dwell_Time_s'],
                alpha=0.6,
                label=f'User {uid}',
                edgecolors='w',
                s=60  # Size of the dots
            )

        # 4. Styling the graph
        plt.title('Keystroke Signature: Dwell Time vs. Flight Time', fontsize=14)
        plt.xlabel('Flight Time (Seconds between keys)', fontsize=12)
        plt.ylabel('Dwell Time (Seconds key was held)', fontsize=12)

        # Add a horizontal/vertical line at 0 for clarity
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)

        plt.legend(title="User ID")
        plt.grid(True, linestyle='--', alpha=0.5)

        # 5. Show plot
        print("Generating Scatter Plot...")
        plt.show()

    except FileNotFoundError:
        print(f"Error: {filename} not found.")


if __name__ == "__main__":
    plot_dwell_vs_flight('ord_combine.csv')