import pandas as pd
from sklearn.metrics import r2_score

# Read data from files
try:
    nuclei_counts_df = pd.read_csv(r'results\segmentation_results\nuclei_counts.txt', delimiter=':', names=['ID', 'Count'])
    manual_count_df = pd.read_csv(r'results\segmentation_results\mannual_counts.txt', delimiter=':', names=['ID', 'Count'])

except FileNotFoundError:
    print("One or both of the files couldn't be found.")
    exit()

# Ensure data is properly loaded
if nuclei_counts_df.empty or manual_count_df.empty:
    print("One or both of the files are empty.")
    exit()

# Assuming both files have a column named 'Count' representing the counts
y_true = manual_count_df['Count']
y_pred = nuclei_counts_df['Count']

# Calculate R2 score
r2 = r2_score(y_true, y_pred)
print("R2 Score:", r2)

# R2 Score: 0.7563761082935017