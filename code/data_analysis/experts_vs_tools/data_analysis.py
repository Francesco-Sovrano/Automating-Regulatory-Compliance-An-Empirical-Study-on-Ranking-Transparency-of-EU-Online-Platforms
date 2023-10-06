import pandas as pd
import zipfile
import os
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

extraction_folder = 'tables/'

# Create a folder to extract the files into
if not os.path.exists(extraction_folder):
    os.makedirs(extraction_folder)

# List the extracted files
extracted_files = os.listdir(extraction_folder)

# Initialize an empty dictionary to hold the processed DataFrames
processed_dfs = {}

# Loop through each extracted CSV file to process the data
for file in extracted_files:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(extraction_folder, file))
        numerical_columns = ['Expert 1', 'Expert 2', 'Expert 3', 'Explanatory Relevance Score', 'DoX Score', 'Pertinence Score']
        df[numerical_columns] = df[numerical_columns].fillna(0)
        df.fillna('N', inplace=True)
        df['DoX-based'] = df['DoXpert'].apply(lambda x: x[0] if pd.notna(x) else 'N')
        df['ChatGPT 3.5'] = df['ChatGPT 3.5'].apply(lambda x: x[0] if pd.notna(x) else 'N')
        df['ChatGPT 4'] = df['ChatGPT 4'].apply(lambda x: x[0] if pd.notna(x) else 'N')
        df['Disagreement'] = df[['Expert 1', 'Expert 2', 'Expert 3']].var(axis=1)
        df['Majority Answer'] = df[['Expert 1', 'Expert 2', 'Expert 3']].median(axis=1)
        df['Normalised Majority Answer'] = df['Majority Answer'].apply(lambda x: 'Y' if x >= 3 else 'N')
        processed_dfs[file] = df

        average_disagreement_variance = df['Disagreement'].mean()
        print(file, f'average_disagreement_variance: {average_disagreement_variance:.2f}')

# Merge all processed DataFrames
merged_df = pd.concat(processed_dfs.values(), ignore_index=True)

merged_df = merged_df.rename(columns={
    'Explanatory Relevance Score': 'Explanatory Relevance',
    'Pertinence Score': 'Pertinence',
    'DoX Score': 'DoX'
})

# Split the merged DataFrame based on 'Normalised Majority Answer'
group_Y = merged_df[merged_df['Normalised Majority Answer'] == 'Y']
group_N = merged_df[merged_df['Normalised Majority Answer'] == 'N']

# Initialize a dictionary to hold test results
test_results = {'Explanatory Relevance': {}, 'Pertinence': {}, 'DoX': {}}

# Perform the Mann-Whitney U tests and calculate rank biserial correlation
for column in ['Explanatory Relevance', 'Pertinence', 'DoX']:
    u_stat, p_val = mannwhitneyu(group_Y[column], group_N[column], alternative='greater')
    rank_biserial = 1 - 2 * u_stat / (len(group_Y) * len(group_N))
    test_results[column]['U Statistic'] = u_stat
    test_results[column]['P-Value'] = p_val
    test_results[column]['Rank Biserial Correlation'] = rank_biserial

# Plotting the results
sns.set(style="whitegrid")
# plt.rcParams.update({'font.size': 14, "axes.labelsize": 14})  # Increase global font size
plt.figure(figsize=(15, 10))
plot_order = ['Explanatory Relevance', 'Pertinence', 'DoX']
palette = {'Y': 'lightgreen', 'N': 'lightcoral'}

ax = sns.boxplot(
    x="variable", 
    y="value", 
    hue="Normalised Majority Answer", 
    data=pd.melt(merged_df, id_vars=['Normalised Majority Answer'], value_vars=plot_order), 
    hue_order=['N', 'Y'],
    order=plot_order, 
    palette=palette
)
# ax.set_title('Comparison of Scores by Normalized Majority Answer', fontsize=22)
ax.set_xlabel('', fontsize=22)
ax.set_ylabel('Value', fontsize=22)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
# plt.legend(fontsize=20)

# Annotate with p-values and correlation coefficients (r)
for i, column in enumerate(plot_order):
    p_val = test_results[column]['P-Value']
    rank_biserial = test_results[column]['Rank Biserial Correlation']
    ax.text(i, merged_df[column].max() + 0.05, f'p = {p_val:.2E}\nr = {rank_biserial:.3f}', horizontalalignment='center', size='x-large', color='black', weight='semibold')

# Update the legend labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=["No (< 3)", "Yes (≥ 3)"], fontsize=22)

# plt.show()
plt.savefig('scores_alignment.png')

# Calculate the agreement with the Normalised Majority Answer for each system
systems = ['ChatGPT 3.5', 'ChatGPT 4', 'DoX-based']
agreements = {}
for system in systems:
    agreements[system] = len(merged_df[merged_df[system] == merged_df['Normalised Majority Answer']])/len(merged_df[system])*100

# Baseline: Random Function
np.random.seed(41)
merged_df['Random'] = np.random.choice(['Y', 'N'], size=len(merged_df))
agreements['Random'] = len(merged_df[merged_df['Random'] == merged_df['Normalised Majority Answer']])/len(merged_df['Random'])*100

# Baseline: Constant Function (Always Yes)
merged_df['Always Yes'] = 'Y'
agreements['Always Yes'] = len(merged_df[merged_df['Always Yes'] == merged_df['Normalised Majority Answer']])/len(merged_df['Always Yes'])*100

# # Determine which system/baseline has the highest agreement
# highest_agreement = max(agreements, key=agreements.get)

# # Plot the agreement bar chart
# fig, ax = plt.subplots(figsize=(12, 7))
# colors = ['blue' if system != highest_agreement else 'gold' for system in agreements.keys()]
# ax.bar(agreements.keys(), agreements.values(), color=colors)

# ax.set_title('Agreement with Normalised Majority Answer')
# ax.set_ylabel('Number of Agreements')
# ax.set_xlabel('System')

# plt.savefig('experts_vs_tools.png')

# Convert the dictionary to a DataFrame for display
# Initialize a dictionary to store agreement percentages by system and table
agreements_by_table = {system: {} for system in systems + ['Random', 'Always Yes']}

# Loop through each processed DataFrame
for file, df in processed_dfs.items():
    total_count = len(df)

    for system in systems:
        agree_count = len(df[df[system] == df['Normalised Majority Answer']])
        agreement_percentage = (agree_count / total_count) * 100
        agreements_by_table[system][file] = agreement_percentage

    # Compute Random baseline for the current table
    np.random.seed(42)  # Set the seed for reproducibility
    df['Random'] = np.random.choice(['Y', 'N'], size=total_count)
    random_agree_count = len(df[df['Random'] == df['Normalised Majority Answer']])
    agreements_by_table['Random'][file] = (random_agree_count / total_count) * 100

    # Compute Always Yes baseline for the current table
    df['Always Yes'] = 'Y'
    always_yes_agree_count = len(df[df['Always Yes'] == df['Normalised Majority Answer']])
    agreements_by_table['Always Yes'][file] = (always_yes_agree_count / total_count) * 100

# Add the merged data to the agreement table
for system in systems + ['Random', 'Always Yes']:
    agreements_by_table[system]['Merged Data'] = agreements[system]

# Convert the nested dictionary to a DataFrame for display
agreement_df_by_table = pd.DataFrame(agreements_by_table).reset_index()
agreement_df_by_table.columns = ['Table'] + list(agreement_df_by_table.columns[1:])

# Rename tables for readability
def get_readable_name(filename):
    if filename.endswith('.csv'):
        name = filename[:-4]  # remove .csv
        parts = name.split('-')
        if len(parts) == 2 and parts[0] == parts[1]:
            return parts[0].capitalize()
    return filename

agreement_df_by_table['Table'] = agreement_df_by_table['Table'].apply(get_readable_name)

print(agreement_df_by_table.to_string(index=False))