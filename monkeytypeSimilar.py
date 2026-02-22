import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_csv('results.csv')

# 2. Preprocessing
# Convert timestamp (milliseconds) to datetime for time-series analysis
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

# 3. Summary Statistics
print("Summary Statistics:")
print(df[['wpm', 'acc', 'consistency', 'rawWpm']].describe())

# 4. Visualization - WPM Trend over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='wpm', data=df.sort_values('date'), color='#2ecc71')
plt.title('Typing Speed (WPM) Progress Over Time')
plt.xlabel('Date')
plt.ylabel('Words Per Minute (WPM)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('wpm_trend.png')

# 5. Visualization - WPM Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['wpm'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Typing Speed')
plt.xlabel('WPM')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('wpm_distribution.png')

# 6. Visualization - WPM vs Accuracy
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wpm', y='acc', data=df, alpha=0.5, color='#e74c3c')
plt.title('Speed (WPM) vs. Accuracy (%)')
plt.xlabel('WPM')
plt.ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig('wpm_vs_accuracy.png')

# 7. Average WPM by Language (Top 10)
plt.figure(figsize=(10, 6))
avg_wpm_lang = df.groupby('language')['wpm'].mean().sort_values(ascending=False).head(10)
avg_wpm_lang.plot(kind='bar', color='coral')
plt.title('Average Speed by Language (Top 10)')
plt.xlabel('Language')
plt.ylabel('Mean WPM')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('avg_wpm_language.png')

print("Analysis complete. Visualizations saved as PNG files.")
