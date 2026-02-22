import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define the Keyboard Grid (Relative Units)
key_coords = {
    'q':(0,0), 'w':(0,1), 'e':(0,2), 'r':(0,3), 't':(0,4), 'y':(0,5),
    'a':(1,0.5), 's':(1,1.5), 'd':(1,2.5), 'f':(1,3.5), 'g':(1,4.5),
    'z':(2,0.7), 'x':(2,1.7), 'c':(2,2.7), 'v':(2,3.7), 'b':(2,4.7)
}

def calculate_wpm_decay(df):
    # Simulated mapping of 'Reach Distance' for the dataset
    # In a real app, you'd calculate this based on the actual keys pressed
    # Here we categorize 'Flight_Time' into Distance Buckets
    
    df['Key_Distance'] = pd.cut(df['Flight_Time_s'], bins=[0, 0.15, 0.3, 0.6], labels=[1, 2, 3])
    df['Instant_WPM'] = 60 / (df['Flight_Time_s'] * 5)
    
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='Flight_Time_s', y='Instant_WPM', 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    
    plt.title("Biometric Decay: WPM vs. Reach Distance")
    plt.xlabel("Travel Time (Distance Proxy)")
    plt.ylabel("Effective WPM")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Text annotation for judges
    plt.text(0.4, df['Instant_WPM'].max()*0.8, 
             "Steep Slope = Small Hand Span\nFlat Slope = Large Hand Span", 
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()

# Call this with your df_clean
# calculate_wpm_decay(df_clean)
