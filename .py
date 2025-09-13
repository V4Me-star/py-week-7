
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
df = sns.load_dataset('iris')

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Check the data types and non-null values
print("\nDataset Info:")
df.info()

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Compute basic statistics for numerical columns
print("Descriptive Statistics:")
print(df.describe())

# Group by species and compute the mean of numerical columns
print("\nMean of numerical columns grouped by species:")
print(df.groupby('species').mean())


# Set a modern seaborn style for the plots
sns.set_theme(style="whitegrid")

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Visualizations', fontsize=20, y=1.02)

# Bar chart of average petal length by species
avg_petal_length = df.groupby('species')['petal_length'].mean().reset_index()
sns.barplot(x='species', y='petal_length', data=avg_petal_length, ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title('Average Petal Length by Species', fontsize=14)
axes[0, 0].set_ylabel('Average Petal Length (cm)')
axes[0, 0].set_xlabel('Species')

# Histogram of petal length
sns.histplot(df['petal_length'], bins=20, kde=True, ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Distribution of Petal Length', fontsize=14)
axes[0, 1].set_xlabel('Petal Length (cm)')
axes[0, 1].set_ylabel('Frequency')

# Scatter plot of sepal length vs. petal length, colored by species
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df, ax=axes[1, 0], s=80)
axes[1, 0].set_title('Sepal Length vs. Petal Length', fontsize=14)
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Petal Length (cm)')
axes[1, 0].legend(title='Species')

# Box plot of sepal width distribution by species
sns.boxplot(x='species', y='sepal_width', data=df, ax=axes[1, 1], palette="coolwarm")
axes[1, 1].set_title('Distribution of Sepal Width by Species', fontsize=14)
axes[1, 1].set_xlabel('Species')
axes[1, 1].set_ylabel('Sepal Width (cm)')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

