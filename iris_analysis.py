import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visualization
sns.set_style("whitegrid")

def load_iris_data():
    """Load Iris dataset and convert to pandas DataFrame"""
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_data(df):
    """Explore dataset structure and basic statistics"""
    print("Dataset Preview:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Group by species and calculate means
    print("\nMean measurements by species:")
    print(df.groupby('species').mean())

def create_visualizations(df):
    """Create required visualizations"""
    # 1. Line chart (mean measurements across species)
    plt.figure(figsize=(10, 6))
    for column in df.columns[:-1]:
        plt.plot(df.groupby('species').mean()[column], label=column, marker='o')
    plt.title('Mean Measurements Across Iris Species')
    plt.xlabel('Species')
    plt.ylabel('Measurement (cm)')
    plt.legend()
    plt.savefig('line_chart.png')
    plt.close()

    # 2. Bar chart (average sepal length by species)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='sepal length (cm)', data=df)
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length (cm)')
    plt.savefig('bar_chart.png')
    plt.close()

    # 3. Histogram (petal length distribution)
    plt.figure(figsize=(8, 6))
    plt.hist(df['petal length (cm)'], bins=20, edgecolor='black')
    plt.title('Distribution of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Count')
    plt.savefig('histogram.png')
    plt.close()

    # 4. Scatter plot (sepal length vs petal length)
    plt.figure(figsize=(8, 6))
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal length (cm)'], 
                   species_data['petal length (cm)'], 
                   label=species, alpha=0.6)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.savefig('scatter_plot.png')
    plt.close()

def main():
    # Load data
    df = load_iris_data()
    
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Explore data
    print("=== Data Exploration ===")
    explore_data(df)
    
    # Handle missing values (not necessary for Iris dataset, but included for completeness)
    try:
        df = df.dropna()  # Drop any missing values
        if df.empty:
            print("Dataset is empty after cleaning.")
            return
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    create_visualizations(df)
    
    # Findings and observations
    print("\n=== Findings and Observations ===")
    print("1. Setosa species has distinctly smaller petals compared to versicolor and virginica.")
    print("2. Petal length shows a clear separation between species, with setosa having the shortest.")
    print("3. Sepal length varies less across species compared to petal measurements.")
    print("4. The scatter plot shows clear clustering of species based on sepal and petal length.")

if __name__ == "__main__":
    main()