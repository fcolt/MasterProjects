import pandas as pd

# Load CSV files
sanity_check_file = 'val_sanity_check.csv'
train_file = 'val.csv'

sanity_check_df = pd.read_csv(sanity_check_file)
train_df = pd.read_csv(train_file)

# Merge dataframes based on the 'Image' column
merged_df = pd.merge(sanity_check_df, train_df, on='Image')

# Compare classes and calculate accuracy
correct_predictions = (merged_df['Class_x'] == merged_df['Class_y']).sum()
total_samples = len(merged_df)

accuracy = correct_predictions / total_samples * 100

print(f'Accuracy: {accuracy:.2f}%')
