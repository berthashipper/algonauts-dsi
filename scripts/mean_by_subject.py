import pandas as pd

# Load the data
df = pd.read_csv('accuracy_vs_pca_final.csv')

# Verify data loading
print("First few rows of data:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Calculate mean accuracy for each subject by PCA components and modality
# Using reset_index() to ensure proper structure
subject_means = df.groupby(['modality', 'subject', 'pca_components'])['accuracy'].mean().reset_index()

# Pivot the table for better readability
subject_means_pivot = subject_means.pivot_table(
    index=['modality', 'subject'],
    columns='pca_components',
    values='accuracy'
)

# Display the results
print("\nAverage Accuracy by Subject, Modality, and PCA Components:")
print("="*70)
print(subject_means_pivot)

# Save both formats to CSV
subject_means.to_csv('subject_accuracy_means_long_format.csv', index=False)
subject_means_pivot.to_csv('subject_accuracy_means_pivot_format.csv')

print("\nResults saved to:")
print("- 'subject_accuracy_means_long_format.csv' (long format)")
print("- 'subject_accuracy_means_pivot_format.csv' (pivot format)")
