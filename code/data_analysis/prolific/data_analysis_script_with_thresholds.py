
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load the data
file_path = 'data/anonymous_data.csv'  # Replace with the path to your data file
df = pd.read_csv(file_path)

# Define the Time taken thresholds for analysis
time_thresholds = [0, 60, 120, 180, 240, 300]

# Initialize dictionaries to store means and p-values for each threshold
mean_values_booking = {threshold: [] for threshold in time_thresholds}
mean_values_tripadvisor = {threshold: [] for threshold in time_thresholds}
p_values = {threshold: [] for threshold in time_thresholds}

# Extract Booking and Tripadvisor related columns
booking_columns = [col for col in df.columns if 'Booking' in col]
tripadvisor_columns = [col for col in df.columns if 'Tripadvisor' in col]

# Calculate means and p-values for each threshold and each quarter
for threshold in time_thresholds:
    df_threshold = df[df['Time taken'] >= threshold]
    for i in range(1, 5):  # For Q1, Q2, Q3, and Q4
        booking_col = [col for col in booking_columns if col.startswith(f'Q{i}')][0]
        tripadvisor_col = [col for col in tripadvisor_columns if col.startswith(f'Q{i}')][0]
        
        # Extract the data for the test
        booking_data = df_threshold[booking_col].dropna()
        tripadvisor_data = df_threshold[tripadvisor_col].dropna()
        
        # Compute means
        mean_values_booking[threshold].append(booking_data.mean())
        mean_values_tripadvisor[threshold].append(tripadvisor_data.mean())
        
        # Perform the one-sided Mann-Whitney U test (Booking < Tripadvisor) and store p-values
        _, p_value = mannwhitneyu(booking_data, tripadvisor_data, alternative='less')
        p_values[threshold].append(p_value)

# Plot the means for each threshold
plt.rcParams.update({'font.size': 14, "axes.labelsize": 14})  # Increase global font size
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# fig.suptitle('Means for Booking and Tripadvisor Questions Across Time Thresholds', fontsize=16)

q_ids = [3,4,15,16]
for i, ax in enumerate(axes.flatten()):
    booking_means = [mean_values_booking[threshold][i] for threshold in time_thresholds]
    tripadvisor_means = [mean_values_tripadvisor[threshold][i] for threshold in time_thresholds]
    
    ax.plot(time_thresholds, booking_means, marker='o', label='Booking', linestyle='-', color='b')
    ax.plot(time_thresholds, tripadvisor_means, marker='o', label='Tripadvisor', linestyle='-', color='r')
    
    # Highlight statistically significant points
    for j, threshold in enumerate(time_thresholds):
        p_value = p_values[threshold][i]
        if p_value < 0.05:
            # ax.plot(threshold, booking_means[j], 'bo', markersize=10)
            # ax.plot(threshold, tripadvisor_means[j], 'ro', markersize=10)
            ax.annotate(f'p: {p_value:.3f}', xy=(threshold,booking_means[j]), xytext=(threshold,booking_means[j]+0.05), textcoords="data", ha='center', va='center', fontsize=14, color='blue')
        if p_value > 1-0.05:
            # ax.plot(threshold, booking_means[j], 'bo', markersize=10)
            # ax.plot(threshold, tripadvisor_means[j], 'ro', markersize=10)
            ax.annotate(f'p: {1-p_value:.3f}', xy=(threshold,tripadvisor_means[j]), xytext=(threshold,tripadvisor_means[j]+0.05), textcoords="data", ha='center', va='center', fontsize=14, color='red')
    
    ax.set_title(f'Q{q_ids[i]}')
    if i > 1:
        ax.set_xlabel('Min. Seconds Spent')
    if i%2 == 0:
        ax.set_ylabel('Mean Score')
    if i == 0:
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('means_when_changing_thresholds.png')

